import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

from solver import build_optimizer, build_scheduler
from .base import BaseModel
from models.head import ClsHead_v1, ClsHead_v2

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


class TextEncoder_v2(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, name_len):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        a = tokenized_prompts.argmax(dim=-1) + name_len
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)+name_len] @ self.text_projection

        return x


class PromptLearner_v1(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BASELINE.N_CTX
        ctx_init = cfg.TRAINER.BASELINE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        # todo: encoder design
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),  # (512,32)
            ("leakyrelu", nn.LeakyReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))  # (32,512)
        ]))

        if cfg.TRAINER.PREC == "fp16":
            self.meta_net.half()

        # classnames = [name.replace("_", " ") for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompts = prompt_prefix

        # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        tokenized_prompts = clip.tokenize(prompts)    # (1, n_tkn)  only one prompt for each image
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)   # [1, n_tkn, 512]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):  # [B,512]
        prefix = self.token_prefix.expand(im_features.shape[0], -1, -1)  # [B,1,512]
        suffix = self.token_suffix.expand(im_features.shape[0], -1, -1)  # [B,72,512]  suffix contain padding tokens
        ctx = self.ctx  # (n_ctx, ctx_dim) [4,512]
        bias = self.meta_net(im_features)  # (batch, ctx_dim) [B,512]
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim) [B,1,512]    per image, 100 images in total
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim) [1,4,512]      shared for all images
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim) [B,4,512]

        # prompts = []
        # for ctx_shifted_i in ctx_shifted:
        #     ctx_i = ctx_shifted_i.unsqueeze(0)  # [1,4,512]
        #     pts_i = self.construct_prompts(ctx_i, prefix, suffix)
        #     prompts.append(pts_i)
        # prompts = torch.stack(prompts)  # [B,1,77,512]
        prompts = self.construct_prompts(ctx_shifted, prefix, suffix).unsqueeze(1)

        return prompts


class CustomCLIP_v1(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_v1(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.cls_head = ClsHead_v1(cfg, classnames, clip_model)

    def forward(self, image, label=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [50,77]  50 base classes
        logit_scale = self.logit_scale.exp()  # [B]

        image_features = self.image_encoder(image.type(self.dtype))  # [B,512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [B,512]
        # image_features = image_features.float()

        prompts = self.prompt_learner(image_features)  # [B,50,77,512]->[B,1,77,512]

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # [B,512]

            text_features, imf_i = text_features.float(), imf_i.float()
            logits.append(self.cls_head(text_features, imf_i.unsqueeze(0)))   # cat / ele_mul
        logits = torch.stack(logits)    # [B,1,50]
        logits = logits.squeeze(1)

        return logits


class PromptLearner_v2(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BASELINE.N_CTX
        ctx_init = cfg.TRAINER.BASELINE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim      # 1024
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        device = 'cuda'

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        # todo: encoder design
        # self.meta_net = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(vis_dim*2, vis_dim*2 // 16)),  # (512,32)
        #     ("leakyrelu", nn.LeakyReLU(inplace=True)),
        #     ("linear2", nn.Linear(vis_dim*2 // 16, ctx_dim))  # (32,512)
        # ]))
        if cfg.MODEL.BACKBONE.NAME.startswith('RN'):
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(vis_dim * 2, ctx_dim))
            ]))
        else:
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(vis_dim, ctx_dim))
            ]))

        if cfg.TRAINER.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ").replace("(", "").replace(")", "") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # print("max length:", max(name_lens))
        # prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompts = prompt_prefix

        tokenized_prompts = clip.tokenize(prompts)  # [n_cls, 77]->[1,77,512]
        tokenized_label = clip.tokenize(classnames)     # [n_cls, 77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # [n_cls, 77, 512]->[1,77,512]
            label_embed = clip_model.token_embedding(tokenized_label).type(dtype)  # [n_cls, 77, 512]

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:67, :])  # EOS    # todo: replace 72 = 77-5

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.label_embed = label_embed[:, 1:1 + 10, :].to(device)   # [n_cls, 5, 512]      # todo: replace 5
        mask = []
        for i in range(len(name_lens)):
            if name_lens[i] > 10:
                print(name_lens[i])
                raise ValueError(f"Length of class {classnames[i]} is larger than 10.")
            else:
                mask_one = torch.ones(name_lens[i], ctx_dim)
                if name_lens[i] == 10:
                    mask.append(mask_one)
                else:
                    mask_zero = torch.zeros(10 - name_lens[i], ctx_dim)
                    mask.append(torch.cat([mask_one, mask_zero], dim=0))
        self.mask = torch.stack(mask).to(device)   # [n_cls, 5, 512]
        self.name_lens = torch.tensor(name_lens)

    def forward(self, im_features, target=None):  # [B,512] -> [B,5,512]
        prefix = self.token_prefix.expand(im_features.size(0), -1, -1)  # [B,1,512]
        suffix = self.token_suffix.expand(im_features.size(0), -1, -1)  # [B,72,512]
        ctx = self.ctx  # (n_ctx, ctx_dim) [4,512]

        pseudo_category_prompt_embed = self.meta_net(im_features)   # [B,5,512]
        if self.training:
            target_embed = self.label_embed[target]  # [B,5,512]
            mask = self.mask[target]  # [B,5,512]
            loss_prompts = F.kl_div((pseudo_category_prompt_embed * mask).softmax(dim=-1).log(), (target_embed * mask).softmax(dim=-1),reduction="sum")
            # loss_prompts = (1 - F.cosine_similarity(pseudo_category_prompt_embed * mask, target_embed * mask, dim=-1)).mean()
        else:
            loss_prompts = 0    # dummy
        ctx = ctx.unsqueeze(0).expand(im_features.shape[0], -1, -1)  # [B, 4, 512]
        ctx_new = torch.cat([ctx, pseudo_category_prompt_embed], dim=1)     # [B, 4+5, 512]

        prompts = torch.cat([prefix, ctx_new, suffix], dim=1)    # [B, 77, 512]

        return prompts, loss_prompts


class CustomCLIP_v2(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_v2(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_v2(clip_model)
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
        image_features = image_features[:, :10, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_cls = image_cls / image_cls.norm(dim=-1, keepdim=True)
        # image_features = image_features.float()

        prompts, loss_prompts = self.prompt_learner(image_features, target)  # [B, 77, 1024]

        # [B,77,1024] [B,5,2048]
        # if self.prompt_learner.training:
        #     name_len = self.prompt_learner.name_lens[target]
        #     # name_len = 5
        # else:
        #     name_len = 5
        name_len = 10
        text_features = self.text_encoder(prompts, tokenized_prompts, name_len)   # [B,1024]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features, image_cls = text_features.float(), image_cls.float()
        logits = self.cls_head(image_cls, text_features)

        return logits, loss_prompts


class Baseline(BaseModel):
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

        self.logger.info("Building custom CLIP")
        self.model = CustomCLIP_v2(cfg, classnames, clip_model)

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
