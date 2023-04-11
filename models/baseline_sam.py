import sys
sys.path.insert(0, '.')
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from solver import build_optimizer, build_scheduler
from .base import BaseModel
from models.head import *
from .bonder import CrossAttnBlock_v1, CrossAttnBlock_sam
from .backbone import ImageEncoderViT, MODEL_PARAMS

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.simple_tokenizer import MySimpleTokenizer
from datasets.templates import get_templates
from functools import partial
import numpy as np

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BASELINE.N_CTX
        ctx_init = cfg.TRAINER.BASELINE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]    # vitb: 512   rn50: 512 (text encoder dim)
        vis_dim = clip_model.visual.output_dim      # vitb: 512  rn50: 1024
        vocab_size = clip_model.vocab_size
        transformer_width = clip_model.transformer_width
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        device = 'cuda'

        num_query_token = 32
        hidden_act = 'gelu'
        self.bonder = CrossAttnBlock_sam(ctx_dim, num_heads=8)
        self.query = nn.Parameter(torch.zeros(1, num_query_token, ctx_dim))
        self.query.data.normal_(mean=0.0, std=0.02)
        print(f"Number of queries: {num_query_token}")
        # self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.vocab_head = nn.Linear(transformer_width, vocab_size, bias=False)
        # self.vocab_head = BertOnlyMLMHead(hidden_size=ctx_dim, hidden_act=hidden_act, vocab_size=vocab_size)

        # query用"a photo of" + [cls]初始化
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            self.query.data[:, :n_ctx] = ctx_vectors[:, :]  # "a photo of a " initialization
            print(f"Initialize query with: {ctx_init}")
        else:
            print(f"Random initialization for query!")

        if cfg.TRAINER.PREC == "fp16":
            self.bonder.half()

        classnames = [name.replace("_", " ").replace("(", "").replace(")", "") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        pseudo_query = [" X" * num_query_token + '.']   # only used to locate <eos>, and store prefix suffix
        label_sentence = ["a photo of a " + n for n in classnames]

        tokenized_query = clip.tokenize(pseudo_query)  # [1,77,512]   only used to locate <eos>, and store prefix suffix
        tokenized_label = clip.tokenize(label_sentence)     # [n_cls, 77],  [n_cls, 77]
        tokenized_label = tokenized_label[:, 1:1 + num_query_token]
        # mask = mask.unsqueeze(-1).expand(-1, -1, ctx_dim)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_query).type(dtype)  # [1,77,512]
            # label_embed = clip_model.token_embedding(tokenized_label).type(dtype)  # [n_cls, 77, 512]

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + num_query_token:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_query = tokenized_query.to(device)
        self.label = tokenized_label.masked_fill(tokenized_label == 0, -100).to(device)   # "remove" pad token, 0 is pad_token_ids
        self.name_lens = torch.tensor(name_lens)
        self.num_query_token = num_query_token
        self.vocab_size = vocab_size
        self.criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)     # todo:label smoothing?

    def forward(self, im_features, im_cls, target=None):  # [B,512] -> [B,5,512]
        prefix = self.token_prefix.expand(im_features.size(0), -1, -1)  # [B,1,512]
        suffix = self.token_suffix.expand(im_features.size(0), -1, -1)  # [B,77-num_query,512]

        # [cls]
        # todo: must discard cls token, since the dim is different
        query = self.query.expand(im_features.size(0), -1, -1).clone()
        # query.data[:, self.n_ctx] = im_cls[:, :]
        query_output = self.bonder(self.query, im_features)   # [B, num_query, dim]
        query_output_vocab = self.vocab_head(query_output)   # [B, num_query, vocab_size]
        # label [B,num_query]    head [B,num_query,vocab_size]
        if self.training:
            target = self.label[target]  # [B,num_query]
            loss_prompts = self.criterion(
                query_output_vocab.view(-1, self.vocab_size),
                target.view(-1)
            )
        else:
            loss_prompts = 0    # dummy

        prompts = torch.cat([prefix, query_output, suffix], dim=1)  # [B, 77, 512]

        return prompts, loss_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        # self.image_encoder = clip_model.visual
        # Use SAM image encoder
        params = MODEL_PARAMS[cfg.MODEL.SAM.NAME]
        self.image_encoder = ImageEncoderViT(
            depth=params['encoder_depth'],
            embed_dim=params['encoder_embed_dim'],
            img_size=cfg.INPUT.SIZE[0],       # params['image_size'],
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=params['encoder_num_heads'],
            patch_size=params['vit_patch_size'],
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=params['encoder_global_attn_indexes'],
            window_size=14,
            out_chans=params['prompt_embed_dim'],
        )
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        vis_dim = params['prompt_embed_dim']
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.cls_head = ClsHead_sam(classnames, vis_dim, ctx_dim)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

        with open(cfg.MODEL.SAM.CHECKPOINT, "rb") as f:
            state_dict = torch.load(f)

        # equal interval sampling
        pos_embed_idx = np.linspace(0, 63, 14)
        pos_embed_idx = list(map(int, pos_embed_idx))
        rel_pos_idx = np.linspace(0, 126, 27)
        rel_pos_idx = list(map(int, rel_pos_idx))
        for k, v in state_dict.items():
            if k == 'pos_embed':
                v = v[:, pos_embed_idx, :, :]
                state_dict[k] = v[:, :, pos_embed_idx, :]
            elif ('rel_pos_w' in k) or ('rel_pos_h' in k):
                if v.shape[0] >=127:
                    state_dict[k] = v[rel_pos_idx, :]
        self.image_encoder.load_state_dict(state_dict)

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features = self.image_encoder(image.type(self.dtype))     # [B, 256, 14, 14]
        image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B, 196, 256]
        image_features = image_features[:, :self.prompt_learner.num_query_token, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_cls = image_features[:, 0, :]     # [B, 256]
        image_cls = image_cls / image_cls.norm(dim=-1, keepdim=True)
        # image_features = image_features.float()

        prompts, loss_prompts = self.prompt_learner(image_features, image_cls, target)  # [B, 77, 1024]

        text_features = self.text_encoder(prompts, tokenized_prompts)   # [B,1024]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features, image_cls = text_features.float(), image_cls.float()
        logits = self.cls_head(image_cls, text_features)

        return logits, loss_prompts


class Baseline_sam(BaseModel):
    def __init__(self, cfg, classnames=None):
        super().__init__()
        self.logger = logging.getLogger(cfg.TRAINER.NAME)
        self.check_cfg(cfg)
        self.cfg = cfg
        self.test_freq = cfg.TRAIN.TEST_FREQ

        self.logger.info(f"Loading CLIP for text encoder")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        for param in clip_model.parameters():
            param.requires_grad = False

        self.logger.info("Building Baseline_sam")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        self.logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = ["prompt_learner", "cls_head"]

        for name, param in self.model.named_parameters():
            if (name_to_update[0] in name) or (name_to_update[1] in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        # Fix the weights of vocab head
        self.model.prompt_learner.vocab_head.weight.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        self.logger.info(f"Parameters to be updated: {enabled}")

        # not necessary
        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        # NOTE: only give prompt_learner,cls_head to the optimizer
        self.optim = build_optimizer([self.model.prompt_learner, self.model.cls_head], cfg.OPTIM)
        self.sched = build_scheduler(self.optim, cfg.OPTIM)

        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("cls_head", self.model.cls_head)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def forward(self, image, label=None):
        return self.model(image, label)     # logits