import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from solver import build_optimizer, build_scheduler
from .base import BaseModel
from models.head.cls_heads import ClsHead_v2
from .bonder import CrossAttnBlock

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

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
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        device = 'cuda'

        num_query_token = 32
        self.bonder = CrossAttnBlock(ctx_dim, num_heads=8)
        self.query = nn.Parameter(torch.zeros(1, num_query_token, ctx_dim))
        self.query.data.normal_(mean=0.0, std=0.02)
        print(f"Number of queries: {num_query_token}")

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
        tokenized_label, mask = clip.tokenize_with_mask(label_sentence)     # [n_cls, 77]
        mask = mask.unsqueeze(-1).expand(-1, -1, ctx_dim)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_query).type(dtype)  # [1,77,512]
            label_embed = clip_model.token_embedding(tokenized_label).type(dtype)  # [n_cls, 77, 512]

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + num_query_token:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_query = tokenized_query  # torch.Tensor
        # self.label_embed = label_embed[:, 1:1 + 10, :].to(device)   # [n_cls, 5, 512]
        self.label_embed = label_embed[:, 1:1+num_query_token, :].to(device)  # [n_cls, 32, 512]
        self.mask = mask[:, 1:1+num_query_token, :].to(device)   # [n_cls, 32, 512]
        self.name_lens = torch.tensor(name_lens)
        self.num_query_token = num_query_token

    def forward(self, im_features, im_cls, target=None):  # [B,512] -> [B,5,512]
        prefix = self.token_prefix.expand(im_features.size(0), -1, -1)  # [B,1,512]
        suffix = self.token_suffix.expand(im_features.size(0), -1, -1)  # [B,77-num_query,512]

        # todo: 不加[cls]可以给self-attention加速
        # self.query.data = self.query.expand(im_features.size(0), -1, -1).clone()
        # self.query.data[:, self.n_ctx] = im_cls[:, :]
        query = self.query.expand(im_features.size(0), -1, -1).clone()    # 梯度可以传给self.query更新，但是self.query还是单行向量
        query.data[:, self.n_ctx] = im_cls[:, :]
        query_output = self.bonder(query, im_features)   # [B, num_query, dim]
        if self.training:
            target_embed = self.label_embed[target]  # [B,num_query,512]
            mask = self.mask[target]  # [B,num_query,512]
            loss_prompts = F.kl_div((query_output * mask).softmax(dim=-1).log(), (target_embed * mask).softmax(dim=-1),reduction="sum")
            # loss_prompts = (1 - F.cosine_similarity(pseudo_category_prompt_embed * mask, target_embed * mask, dim=-1)).mean()
        else:
            loss_prompts = 0    # dummy

        prompts = torch.cat([prefix, query_output, suffix], dim=1)  # [B, 77, 512]

        return prompts, loss_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.cls_head = ClsHead_v2(cfg, classnames, clip_model)

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]
        # todo: resampler
        image_features = image_features[:, :self.prompt_learner.num_query_token, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_cls = image_cls / image_cls.norm(dim=-1, keepdim=True)
        # image_features = image_features.float()

        prompts, loss_prompts = self.prompt_learner(image_features, image_cls, target)  # [B, 77, 1024]

        text_features = self.text_encoder(prompts, tokenized_prompts)   # [B,1024]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features, image_cls = text_features.float(), image_cls.float()
        logits = self.cls_head(image_cls, text_features)

        return logits, loss_prompts


class Baseline_cattn(BaseModel):
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

        self.logger.info("Building Baseline_cattn")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        self.logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = ["prompt_learner", "cls_head"]

        for name, param in self.model.named_parameters():
            if (name_to_update[0] in name) or (name_to_update[1] in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

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
