import sys
sys.path.insert(0, '.')
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from solver import build_optimizer, build_scheduler, build_scheduler_iter
from .base import BaseModel
from models.head import *
from .bonder import CrossAttnBlock, CrossAttnBlock_nx, CrossAttnBlock_projkv,\
    CrossAttnBlock_nx_projkv, CrossAttnBlock_single_query

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.simple_tokenizer import MySimpleTokenizer
from datasets.templates import get_templates

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

    def forward(self, prompts, tokenized_prompts):      # [b,cls, num_query, dim]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner_coophead(nn.Module):
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

        num_query_token = cfg.MODEL.BONDER.NUM_Q
        depth = cfg.MODEL.BONDER.DEPTH
        self.bonder = CrossAttnBlock_projkv(ctx_dim, num_heads=8)
        self.query = nn.Parameter(torch.zeros(1, num_query_token, ctx_dim))
        self.query.data.normal_(mean=0.0, std=0.02)
        print(f"Number of queries: {num_query_token}")
        print(f"Depth of bonder: {depth}")
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
        pseudo_prompts = [" ".join(["X"] * num_query_token) + " " + name + "." for name in classnames]
        label_sentence = ["a photo of a " + n for n in classnames]

        tokenized_prompts = clip.tokenize(pseudo_prompts)  # [n_cls,77,512]   only used to locate <eos>, and store prefix suffix

        tokenized_label = clip.tokenize(label_sentence)     # [n_cls, 77],  [n_cls, 77]
        tokenized_label = tokenized_label[:, 1:1 + num_query_token]
        with torch.no_grad():
            embedding_prompts = clip_model.token_embedding(tokenized_prompts).type(dtype)  # [n_cls, 77, 512]

        self.register_buffer("token_prefix_prompts", embedding_prompts[:, :1, :])  # SOS
        self.register_buffer("token_suffix_prompts", embedding_prompts[:, 1 + num_query_token:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts.to(device)
        self.label = tokenized_label.masked_fill(tokenized_label == 0, -100).to(device)   # "remove" pad token, 0 is pad_token_ids
        self.name_lens = torch.tensor(name_lens)
        self.num_query_token = num_query_token
        self.vocab_size = vocab_size
        self.criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)

    def forward(self, im_features, im_cls, target=None):
        prefix = self.token_prefix_prompts.unsqueeze(0).expand(im_cls.size(0), -1, -1, -1)      # [cls, 1, 512]
        suffix = self.token_suffix_prompts.unsqueeze(0).expand(im_cls.size(0), -1, -1, -1)      # [cls, 72, 512]
        # note: cat [cls] or not?
        query = self.query.expand(im_cls.size(0), -1, -1).clone()   # [b, num_q, dim]
        query.data[:, self.n_ctx] = im_cls[:, :]
        query_output = self.bonder(self.query, im_features)  # [B, num_query, dim]
        if self.training:
            query_output_vocab = self.vocab_head(query_output)
            target = self.label[target]  # [B,num_query]
            loss_prompts = self.criterion(
                query_output_vocab.view(-1, self.vocab_size),
                target.view(-1)
            )
        else:
            loss_prompts = 0    # dummy

        query_output = query_output.unsqueeze(1)
        query_output = query_output.expand(-1, self.n_cls, -1, -1)  # [b, cls, num_query, dim]
        prompts = torch.cat([prefix, query_output, suffix], dim=2)
        return prompts, loss_prompts


class CustomCLIP_coophead(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        device = 'cuda'
        self.prompt_learner = PromptLearner_coophead(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME
        self.logit_scale = clip_model.logit_scale

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]
        prompts, loss_prompts = self.prompt_learner(image_features, image_cls, target)  # [B, 77, 1024]

        logits = []
        image_cls = image_cls / image_cls.norm(dim=-1, keepdim=True)    # [b,dim]
        for pts_i, im_i in zip(prompts, image_cls):     # [dim]
            text_features = self.text_encoder(pts_i, tokenized_prompts)     # [cls,dim]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * im_i @ text_features.t()    # [cls]
            logits.append(l_i)
        logits = torch.stack(logits)
        # print("image shape:", image_cls.shape)
        # print("text shape:", text_features.shape)
        # print("logit shape:", logits.shape)

        return logits, loss_prompts


class Baseline_cattn_coophead(BaseModel):
    def __init__(self, cfg, classnames=None):
        super().__init__()
        self.logger = logging.getLogger(cfg.TRAINER.NAME)
        self.check_cfg(cfg)
        self.cfg = cfg
        self.test_freq = cfg.TRAIN.TEST_FREQ

        self.logger.info(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        for param in clip_model.parameters():
            param.requires_grad = False

        self.logger.info("Building Baseline_cattn_coophead")
        self.model = CustomCLIP_coophead(cfg, classnames, clip_model)

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

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_scheduler_iter(self.optim, cfg.OPTIM)

        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def forward(self, image, label=None):
        return self.model(image, label)     # logits