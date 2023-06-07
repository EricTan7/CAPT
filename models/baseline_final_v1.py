import sys
sys.path.insert(0, '.')
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from solver import build_optimizer, build_scheduler, build_scheduler_iter
from .base import BaseModel
from models.head import *
from .bonder import CrossAttnBlock, CrossAttnBlock_nx, CrossAttnBlock_projkv, CrossAttnBlock_nx_projkv

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.simple_tokenizer import MySimpleTokenizer
from datasets.templates import get_templates
from tools.model import load_pretrained_weights

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


class PromptLearner_shembed_zsinit_lscale_wiseft(nn.Module):
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
        self.bonder = CrossAttnBlock(ctx_dim, num_heads=8)
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
        label_sentence = ["a photo of a " + n for n in classnames]  # todo: use template ensemble

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
        self.criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)

    def forward(self, im_features, im_cls, target=None):  # [B,512] -> [B,5,512]
        prefix = self.token_prefix.expand(im_features.size(0), -1, -1)  # [B,1,512]
        suffix = self.token_suffix.expand(im_features.size(0), -1, -1)  # [B,77-num_query,512]

        # [cls]
        query = self.query.expand(im_features.size(0), -1, -1).clone()
        query.data[:, self.n_ctx] = im_cls[:, :]
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


class PromptLearner_template_ensemble(nn.Module):
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
        self.bonder = CrossAttnBlock(ctx_dim, num_heads=8)
        self.query = nn.Parameter(torch.zeros(1, num_query_token, ctx_dim))
        self.query.data.normal_(mean=0.0, std=0.02)
        print(f"Number of queries: {num_query_token}")
        # self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.vocab_head = nn.Linear(transformer_width, vocab_size, bias=False)
        # self.vocab_head = BertOnlyMLMHead(hidden_size=ctx_dim, hidden_act=hidden_act, vocab_size=vocab_size)

        # query用"a photo of" + [cls]初始化
        # if ctx_init:
        #     # use given words to initialize context vectors
        #     ctx_init = ctx_init.replace("_", " ")
        #     n_ctx = len(ctx_init.split(" "))
        #     prompt = clip.tokenize(ctx_init)
        #     with torch.no_grad():
        #         embedding = clip_model.token_embedding(prompt).type(dtype)
        #     ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        #     self.query.data[:, :n_ctx] = ctx_vectors[:, :]  # "a photo of a " initialization
        #     print(f"Initialize query with: {ctx_init}")
        # else:
        #     print(f"Random initialization for query!")

        if cfg.TRAINER.PREC == "fp16":
            self.bonder.half()

        classnames = [name.replace("_", " ").replace("(", " ").replace(")", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        pseudo_query = [" X" * num_query_token + '.']   # only used to locate <eos>, and store prefix suffix
        # label_sentence = ["a photo of a " + n for n in classnames]  # todo: use template ensemble
        text_templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)
        template_len = len(text_templates)
        label_sentence = []     # [n_cls, temp]
        for n in classnames:
            label_sentence.extend([template.format(n) for template in text_templates])

        tokenized_query = clip.tokenize(pseudo_query)  # [1,77,512]   only used to locate <eos>, and store prefix suffix
        tokenized_label = clip.tokenize(label_sentence)     # [n_cls, 77] [n_cls*temp, 77]
        tokenized_label = tokenized_label[:, 1:1 + num_query_token]
        tokenized_label = tokenized_label.reshape(-1, template_len, num_query_token)    # [n_cls, temp, num_q]
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
        self.criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)

    def forward(self, im_features, im_cls, target=None):  # [B,512] -> [B,5,512]
        prefix = self.token_prefix.expand(im_features.size(0), -1, -1)  # [B,1,512]
        suffix = self.token_suffix.expand(im_features.size(0), -1, -1)  # [B,77-num_query,512]

        # [cls]
        query = self.query.expand(im_features.size(0), -1, -1).clone()
        query.data[:, self.n_ctx] = im_cls[:, :]
        query_output = self.bonder(self.query, im_features)   # [B, num_query, dim]
        query_output_vocab = self.vocab_head(query_output)   # [B, num_query, vocab_size]
        # label [B,num_query]    head [B,num_query,vocab_size]
        if self.training:
            target = self.label[target]  # [B, temp, num_q]
            # expand to parallel among all templates
            query_output_vocab = query_output_vocab.unsqueeze(1).expand(-1, target.size(1),-1, -1).contiguous()   # [b, temp, num_q, vocab_size]
            loss_prompts = self.criterion(
                query_output_vocab.view(-1, self.vocab_size),   # [b*temp*num_q, vocab_size]
                target.view(-1)  # [b*temp*num_q]
            )
        else:
            loss_prompts = 0    # dummy

        prompts = torch.cat([prefix, query_output, suffix], dim=1)  # [B, 77, 512]

        return prompts, loss_prompts
    # note: 现在是一次前向，所有templates算一次loss；还可以一个templates更新一次梯度


# ================================ final v1 ===================================
class CustomCLIP_shembed_zsinit_lscale_wiseft(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        device = 'cuda'
        self.prompt_learner = PromptLearner_shembed_zsinit_lscale_wiseft(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = torch.tensor(4.60517)
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.cls_head = ClsHead_cat_lscale(classnames, clip_model, self.logit_scale)
        self.wiseft_head = ClsHead_cat_lscale(classnames, clip_model, self.logit_scale)
        self.wiseft_head2 = ClsHead_cat_lscale(classnames, clip_model, self.logit_scale)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

        self.text_templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)
        self.zs_weights = self.get_zero_shot_weights(classnames, clip_model).to(device)    # check if require grad: false
        self.cls_head.fc.weight.data = self.zs_weights.clone()

    def get_zero_shot_weights(self, classnames, clip_model, device="cuda"):
        num_classes = len(classnames)
        self.text_encoder.to(device)
        with torch.no_grad():
            weights = torch.empty_like(self.cls_head.fc.weight.data)
            for label in range(num_classes):
                text_prompts = [template.format(classnames[label]) for template in self.text_templates]
                text_tokenized = clip.tokenize(text_prompts)
                text_embedding = clip_model.token_embedding(text_tokenized).type(self.dtype)
                text_embedding = text_embedding.to(device)

                text_features = self.text_encoder(text_embedding, text_tokenized)
                text_features = text_features.mean(dim=0)  # average across all templates
                text_features = torch.cat([text_features, text_features])
                weights[label] = text_features
            weights.data = F.normalize(weights, dim=1)
        return weights

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]

        prompts, loss_prompts = self.prompt_learner(image_features, image_cls, target)  # [B, 77, 1024]

        text_features = self.text_encoder(prompts, tokenized_prompts)   # [B,1024]
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features, image_cls = text_features.float(), image_cls.float()
        fused_fea = torch.cat([image_cls, text_features], dim=1)
        logits = self.cls_head(fused_fea)
        if not self.prompt_learner.training:
            logits_wiseft = self.wiseft_head(fused_fea)
            # logits_wiseft2 = self.wiseft_head2(fused_fea)
            # return logits, logits_wiseft, logits_wiseft2
            return logits, logits_wiseft

        return logits, loss_prompts


class Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft")
        self.model = CustomCLIP_shembed_zsinit_lscale_wiseft(cfg, classnames, clip_model)

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

        self.optim = build_optimizer([self.model.prompt_learner, self.model.cls_head], cfg.OPTIM)
        self.sched = build_scheduler_iter(self.optim, cfg.OPTIM)

        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("cls_head", self.model.cls_head)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def forward(self, image, label=None):
        return self.model(image, label)     # logits


# =============================== independent fc =========================================
class CustomCLIP_shembed_zsinit_lscale_wiseft_idfc(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        device = 'cuda'
        self.prompt_learner = PromptLearner_shembed_zsinit_lscale_wiseft(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = torch.tensor(4.60517)
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        # self.cls_head = ClsHead_cat_lscale(classnames, clip_model, self.logit_scale)
        self.img_head = ClsHead_lscale(classnames, clip_model, self.logit_scale)
        self.text_head = ClsHead_lscale(classnames, clip_model, self.logit_scale)
        self.img_wiseft = ClsHead_lscale(classnames, clip_model, self.logit_scale)
        self.text_wiseft = ClsHead_lscale(classnames, clip_model, self.logit_scale)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

        self.text_templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)
        self.zs_weights = self.get_zero_shot_weights(classnames, clip_model).to(device)    # check if require grad: false
        self.img_head.fc.weight.data = self.zs_weights.clone()
        self.text_head.fc.weight.data = self.zs_weights.clone()

    def get_zero_shot_weights(self, classnames, clip_model, device="cuda"):
        num_classes = len(classnames)
        self.text_encoder.to(device)
        with torch.no_grad():
            weights = torch.empty_like(self.img_head.fc.weight.data)
            for label in range(num_classes):
                text_prompts = [template.format(classnames[label]) for template in self.text_templates]
                text_tokenized = clip.tokenize(text_prompts)
                text_embedding = clip_model.token_embedding(text_tokenized).type(self.dtype)
                text_embedding = text_embedding.to(device)

                text_features = self.text_encoder(text_embedding, text_tokenized)
                text_features = text_features.mean(dim=0)  # average across all templates
                # text_features = torch.cat([text_features, text_features])
                weights[label] = text_features
            weights.data = F.normalize(weights, dim=1)
        return weights

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]

        prompts, loss_prompts = self.prompt_learner(image_features, image_cls, target)  # [B, 77, 1024]

        text_features = self.text_encoder(prompts, tokenized_prompts)   # [B,1024]
        text_features, image_cls = text_features.float(), image_cls.float()
        # fused_fea = torch.cat([image_cls, text_features], dim=1)
        img_logits = self.img_head(image_cls)
        text_logits = self.text_head(text_features)
        if not self.prompt_learner.training:
            img_wiseft_logits = self.img_wiseft(image_cls)
            text_wiseft_logits = self.text_wiseft(text_features)
            return img_logits, text_logits, img_wiseft_logits, text_wiseft_logits

        return img_logits, text_logits, loss_prompts


class Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_idfc(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_idfc")
        self.model = CustomCLIP_shembed_zsinit_lscale_wiseft_idfc(cfg, classnames, clip_model)

        self.logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = ["prompt_learner", "img_head", "text_head"]

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

        self.optim = build_optimizer([self.model.prompt_learner, self.model.img_head, self.model.text_head], cfg.OPTIM)
        self.sched = build_scheduler_iter(self.optim, cfg.OPTIM)

        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("img_head", self.model.img_head)
        self.register_model("text_head", self.model.text_head)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def forward(self, image, label=None):
        return self.model(image, label)     # logits


# ============================== fusion: add =======================================
class CustomCLIP_shembed_zsinit_lscale_wiseft_add(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        device = 'cuda'
        self.prompt_learner = PromptLearner_shembed_zsinit_lscale_wiseft(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = torch.tensor(4.60517)
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.cls_head = ClsHead_add_lscale(classnames, clip_model, self.logit_scale)
        self.wiseft_head = ClsHead_add_lscale(classnames, clip_model, self.logit_scale)
        self.wiseft_head2 = ClsHead_add_lscale(classnames, clip_model, self.logit_scale)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

        self.text_templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)
        self.zs_weights = self.get_zero_shot_weights(classnames, clip_model).to(device)    # check if require grad: false
        self.cls_head.fc.weight.data = self.zs_weights.clone()

    def get_zero_shot_weights(self, classnames, clip_model, device="cuda"):
        num_classes = len(classnames)
        self.text_encoder.to(device)
        with torch.no_grad():
            weights = torch.empty_like(self.cls_head.fc.weight.data)
            for label in range(num_classes):
                text_prompts = [template.format(classnames[label]) for template in self.text_templates]
                text_tokenized = clip.tokenize(text_prompts)
                text_embedding = clip_model.token_embedding(text_tokenized).type(self.dtype)
                text_embedding = text_embedding.to(device)

                text_features = self.text_encoder(text_embedding, text_tokenized)
                text_features = text_features.mean(dim=0)  # average across all templates
                # text_features = torch.cat([text_features, text_features])
                weights[label] = text_features
            weights.data = F.normalize(weights, dim=1)
        return weights

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]

        prompts, loss_prompts = self.prompt_learner(image_features, image_cls, target)  # [B, 77, 1024]

        text_features = self.text_encoder(prompts, tokenized_prompts)   # [B,1024]
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features, image_cls = text_features.float(), image_cls.float()
        # fused_fea = torch.cat([image_cls, text_features], dim=1)
        fused_fea = (image_cls + text_features) / 2.
        logits = self.cls_head(fused_fea)
        if not self.prompt_learner.training:
            logits_wiseft = self.wiseft_head(fused_fea)
            logits_wiseft2 = self.wiseft_head2(fused_fea)
            return logits, logits_wiseft, logits_wiseft2
            # return logits, logits_wiseft

        return logits, loss_prompts


class Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_add(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_add")
        self.model = CustomCLIP_shembed_zsinit_lscale_wiseft_add(cfg, classnames, clip_model)

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

        self.optim = build_optimizer([self.model.prompt_learner, self.model.cls_head], cfg.OPTIM)
        self.sched = build_scheduler_iter(self.optim, cfg.OPTIM)

        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("cls_head", self.model.cls_head)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def forward(self, image, label=None):
        return self.model(image, label)     # logits


# ============================== supervision: templates ensemble =======================================
class CustomCLIP_template_ensemble(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        device = 'cuda'
        self.prompt_learner = PromptLearner_template_ensemble(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = torch.tensor(4.60517)
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.cls_head = ClsHead_cat_lscale(classnames, clip_model, self.logit_scale)
        self.wiseft_head = ClsHead_cat_lscale(classnames, clip_model, self.logit_scale)
        self.wiseft_head2 = ClsHead_cat_lscale(classnames, clip_model, self.logit_scale)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

        self.text_templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)
        self.zs_weights = self.get_zero_shot_weights(classnames, clip_model).to(device)    # check if require grad: false
        self.cls_head.fc.weight.data = self.zs_weights.clone()

    def get_zero_shot_weights(self, classnames, clip_model, device="cuda"):
        num_classes = len(classnames)
        self.text_encoder.to(device)
        with torch.no_grad():
            weights = torch.empty_like(self.cls_head.fc.weight.data)
            for label in range(num_classes):
                text_prompts = [template.format(classnames[label]) for template in self.text_templates]
                text_tokenized = clip.tokenize(text_prompts)
                text_embedding = clip_model.token_embedding(text_tokenized).type(self.dtype)
                text_embedding = text_embedding.to(device)

                text_features = self.text_encoder(text_embedding, text_tokenized)
                text_features = text_features.mean(dim=0)  # average across all templates
                text_features = torch.cat([text_features, text_features])
                weights[label] = text_features
            weights.data = F.normalize(weights, dim=1)
        return weights

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]

        prompts, loss_prompts = self.prompt_learner(image_features, image_cls, target)  # [B, 77, 1024]

        text_features = self.text_encoder(prompts, tokenized_prompts)   # [B,1024]

        text_features, image_cls = text_features.float(), image_cls.float()
        fused_fea = torch.cat([image_cls, text_features], dim=1)
        # fused_fea = image_cls + text_features
        logits = self.cls_head(fused_fea)
        if not self.prompt_learner.training:
            logits_wiseft = self.wiseft_head(fused_fea)
            logits_wiseft2 = self.wiseft_head2(fused_fea)
            return logits, logits_wiseft, logits_wiseft2
            # return logits, logits_wiseft

        return logits, loss_prompts


class Baseline_cattn_wiseft_template_ensemble(BaseModel):
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

        self.logger.info("Building Baseline_cattn_wiseft_template_ensemble")
        self.model = CustomCLIP_template_ensemble(cfg, classnames, clip_model)

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

        self.optim = build_optimizer([self.model.prompt_learner, self.model.cls_head], cfg.OPTIM)
        self.sched = build_scheduler_iter(self.optim, cfg.OPTIM)

        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("cls_head", self.model.cls_head)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def forward(self, image, label=None):
        return self.model(image, label)     # logits

    # def load_model(self, directory, epoch=None):
    #     # directory
    #     load_pretrained_weights(self.model.prompt_learner, "")
    #     load_pretrained_weights(self.model.cls_head, "")
    #     load_pretrained_weights(self.model.wiseft_head, "")

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)