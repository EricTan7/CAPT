import sys
sys.path.insert(0, '.')
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from solver import build_optimizer, build_scheduler
from .base import BaseModel
from models.head import *
from .bonder import CrossAttnBlock_v1, CrossAttnBlock_v2, CrossAttnBlock_v2_pe

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.simple_tokenizer import MySimpleTokenizer
from datasets.templates import get_templates
import torch.distributed as dist

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
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        device = 'cuda'

        num_query_token = 32
        hidden_act = 'gelu'
        self.bonder = CrossAttnBlock_v1(ctx_dim, num_heads=8)
        self.query = nn.Parameter(torch.zeros(1, num_query_token, ctx_dim))
        self.query.data.normal_(mean=0.0, std=0.02)
        print(f"Number of queries: {num_query_token}")
        self.vocab_head = BertOnlyMLMHead(hidden_size=ctx_dim, hidden_act=hidden_act, vocab_size=vocab_size)

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
            # loss_prompts = F.kl_div((query_output * mask).softmax(dim=-1).log(), (target_embed * mask).softmax(dim=-1),reduction="sum")
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


class Baseline_cattn_vocabloss(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss")
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


# ============================================ Shared Embedding mul =======================================
class PromptLearner_shembed_mul(nn.Module):
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
        self.bonder = CrossAttnBlock_v1(ctx_dim, num_heads=8)
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


class CustomCLIP_shembed_mul(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_shembed_mul(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        # self.cls_head = ClsHead_v1(cfg, classnames, clip_model)
        self.cls_head = ClsHead_fea_scale(cfg, classnames, clip_model)
        # self.cls_head = ClsHead_logit_scale(cfg, classnames, clip_model)
        self.fea_scale = cfg.TRAINER.BASELINE.FEA_SCALE
        self.log_scale = cfg.TRAINER.BASELINE.LOG_SCALE

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]
        image_features = image_features[:, :self.prompt_learner.num_query_token, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_cls = image_cls / image_cls.norm(dim=-1, keepdim=True)
        # image_features = image_features.float()

        prompts, loss_prompts = self.prompt_learner(image_features, image_cls, target)  # [B, 77, 1024]

        text_features = self.text_encoder(prompts, tokenized_prompts)   # [B,1024]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features, image_cls = text_features.float(), image_cls.float()

        # logit scale happen before "mul", so it should be employed here
        # if here, the logit scale will be mul by 2 times
        # image_cls = logit_scale * image_cls
        # text_features = logit_scale * text_features

        logits = self.cls_head(image_cls, text_features, self.fea_scale)
        # logits = self.cls_head(image_cls, text_features, self.log_scale)

        return logits, loss_prompts


class Baseline_cattn_vocabloss_shembed_mul(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_shembed_mul")
        self.model = CustomCLIP_shembed_mul(cfg, classnames, clip_model)

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


# ============================================ Shared Embedding =======================================
class PromptLearner_shembed(nn.Module):
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
        self.bonder = CrossAttnBlock_v1(ctx_dim, num_heads=8)
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


class CustomCLIP_shembed(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_shembed(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.cls_head = ClsHead_v2(cfg, classnames, clip_model)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]
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


class Baseline_cattn_vocabloss_shembed(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_shembed")
        self.model = CustomCLIP_shembed(cfg, classnames, clip_model)

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


# ============================================ Compact vocabulary ======================================
class PromptLearner_cpvocab(nn.Module):
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
        hidden_act = 'gelu'
        self.bonder = CrossAttnBlock_v1(ctx_dim, num_heads=8)
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
        classnames.extend(['a', 'photo', 'of', '.'])
        mytokenzier = MySimpleTokenizer(classnames)
        self.vocab_size = mytokenzier.vocab_size
        self.vocab_head = BertOnlyMLMHead(hidden_size=ctx_dim, hidden_act=hidden_act, vocab_size=self.vocab_size)

        pseudo_query = [" X" * num_query_token + '.']
        label_sentence = ["a photo of a " + n for n in classnames]
        tokenized_query = clip.tokenize(pseudo_query)  # original token id
        tokenized_label = mytokenzier.forward(label_sentence)     # compact token id
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
        self.num_query_token = num_query_token
        self.criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)     # todo:label smoothing?

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
            # loss_prompts = F.kl_div((query_output * mask).softmax(dim=-1).log(), (target_embed * mask).softmax(dim=-1),reduction="sum")
            # loss_prompts = (1 - F.cosine_similarity(pseudo_category_prompt_embed * mask, target_embed * mask, dim=-1)).mean()
        else:
            loss_prompts = 0    # dummy

        prompts = torch.cat([prefix, query_output, suffix], dim=1)  # [B, 77, 512]

        return prompts, loss_prompts


class CustomCLIP_cpvocab(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_cpvocab(cfg, classnames, clip_model)
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


class Baseline_cattn_vocabloss_cpvocab(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_cpvocab")
        self.model = CustomCLIP_cpvocab(cfg, classnames, clip_model)

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


# ============================================ Ablation model ======================================
class PromptLearner_wotextloss(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BASELINE.N_CTX
        ctx_init = cfg.TRAINER.BASELINE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]    # vitb: 512   rn50: 512 (text encoder dim)
        vis_dim = clip_model.visual.output_dim      # vitb: 512  rn50: 1024
        vocab_size = clip_model.vocab_size
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        device = 'cuda'

        num_query_token = 32
        hidden_act = 'gelu'
        self.bonder = CrossAttnBlock_v1(ctx_dim, num_heads=8)
        self.query = nn.Parameter(torch.zeros(1, num_query_token, ctx_dim))
        self.query.data.normal_(mean=0.0, std=0.02)
        print(f"Number of queries: {num_query_token}")
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
        # label_sentence = ["a photo of a " + n for n in classnames]

        tokenized_query = clip.tokenize(pseudo_query)  # [1,77,512]   only used to locate <eos>, and store prefix suffix
        # tokenized_label = clip.tokenize(label_sentence)     # [n_cls, 77],  [n_cls, 77]
        # tokenized_label = tokenized_label[:, :num_query_token]
        # mask = mask.unsqueeze(-1).expand(-1, -1, ctx_dim)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_query).type(dtype)  # [1,77,512]
            # label_embed = clip_model.token_embedding(tokenized_label).type(dtype)  # [n_cls, 77, 512]

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + num_query_token:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_query = tokenized_query.to(device)
        # self.label = tokenized_label.masked_fill(tokenized_label == 0, -100).to(device)   # "remove" pad token, 0 is pad_token_ids
        self.name_lens = torch.tensor(name_lens)
        self.num_query_token = num_query_token
        # self.vocab_size = vocab_size
        # self.criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)     # todo:label smoothing?

    def forward(self, im_features, im_cls, target=None):  # [B,512] -> [B,5,512]
        prefix = self.token_prefix.expand(im_features.size(0), -1, -1)  # [B,1,512]
        suffix = self.token_suffix.expand(im_features.size(0), -1, -1)  # [B,77-num_query,512]

        # [cls]
        query = self.query.expand(im_features.size(0), -1, -1).clone()
        query.data[:, self.n_ctx] = im_cls[:, :]
        query_output = self.bonder(self.query, im_features)   # [B, num_query, dim]
        # query_output_vocab = self.vocab_head(query_output)   # [B, num_query, vocab_size]
        # label [B,num_query]    head [B,num_query,vocab_size]
        # if self.training:
        #     target = self.label[target]  # [B,num_query]
        #     loss_prompts = self.criterion(
        #         query_output_vocab.view(-1, self.vocab_size),
        #         target.view(-1)
        #     )
        #     # loss_prompts = F.kl_div((query_output * mask).softmax(dim=-1).log(), (target_embed * mask).softmax(dim=-1),reduction="sum")
        #     # loss_prompts = (1 - F.cosine_similarity(pseudo_category_prompt_embed * mask, target_embed * mask, dim=-1)).mean()
        # else:
        #     loss_prompts = 0    # dummy
        loss_prompts = torch.tensor(0)

        prompts = torch.cat([prefix, query_output, suffix], dim=1)  # [B, 77, 512]

        return prompts, loss_prompts


class CustomCLIP_wotextloss(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_wotextloss(cfg, classnames, clip_model)
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


class Baseline_cattn_vocabloss_wotextloss(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_wotextloss")
        self.model = CustomCLIP_wotextloss(cfg, classnames, clip_model)

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



# ===================================== zeroshot init =======================
class PromptLearner_shembed_zsinit(nn.Module):
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
        self.bonder = CrossAttnBlock_v1(ctx_dim, num_heads=8)
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


class CustomCLIP_shembed_zsinit(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_shembed_zsinit(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.cls_head = ClsHead_v2(cfg, classnames, clip_model)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

        self.text_templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)
        self.zero_shot_init_head(classnames, clip_model)

    def zero_shot_init_head(self, classnames, clip_model, device="cuda"):
        # imagenet_templates = [
        #     "itap of a {}.",
        #     "a bad photo of the {}.",
        #     "a origami {}.",
        #     "a photo of the large {}.",
        #     "a {} in a video game.",
        #     "art of the {}.",
        #     "a photo of the small {}."
        # ]

        # classname:["dog", "dsfa", ...]
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
        self.cls_head.fc.weight.data = weights

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]
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


class Baseline_cattn_vocabloss_shembed_zsinit(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_shembed_zsinit")
        self.model = CustomCLIP_shembed_zsinit(cfg, classnames, clip_model)

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
        if dist.get_rank() == 0:
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



# ===================================== zeroshot init and fixed =======================
class PromptLearner_shembed_zsinit_fixed(nn.Module):
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
        self.bonder = CrossAttnBlock_v1(ctx_dim, num_heads=8)
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


class CustomCLIP_shembed_zsinit_fixed(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_shembed_zsinit_fixed(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.cls_head = ClsHead_v2(cfg, classnames, clip_model)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

        self.text_templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)
        self.zero_shot_init_head(classnames, clip_model)

    def zero_shot_init_head(self, classnames, clip_model, device="cuda"):
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
        self.cls_head.fc.weight.data = weights

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]
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


class Baseline_cattn_vocabloss_shembed_zsinit_fixed(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_shembed_zsinit_fixed")
        self.model = CustomCLIP_shembed_zsinit_fixed(cfg, classnames, clip_model)

        self.logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = ["prompt_learner", "cls_head"]

        for name, param in self.model.named_parameters():
            if (name_to_update[0] in name) or (name_to_update[1] in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        # Fix the weights of vocab head
        self.model.prompt_learner.vocab_head.weight.requires_grad_(False)

        # Fix the weights of fc
        self.model.cls_head.fc.weight.requires_grad_(False)

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



# ============================================ Shared Embedding, cat and logit_scale=======================================
class PromptLearner_shembed_lscale(nn.Module):
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
        self.bonder = CrossAttnBlock_v1(ctx_dim, num_heads=8)
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


class CustomCLIP_shembed_lscale(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_shembed_lscale(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.cls_head = ClsHead_v2_lscale(cfg, classnames, clip_model)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]
        image_features = image_features[:, :self.prompt_learner.num_query_token, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_cls = image_cls / image_cls.norm(dim=-1, keepdim=True)
        # image_features = image_features.float()

        prompts, loss_prompts = self.prompt_learner(image_features, image_cls, target)  # [B, 77, 1024]

        text_features = self.text_encoder(prompts, tokenized_prompts)   # [B,1024]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features, image_cls = text_features.float(), image_cls.float()
        logits = self.cls_head(image_cls, text_features, self.logit_scale)

        return logits, loss_prompts


class Baseline_cattn_vocabloss_shembed_lscale(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_shembed_lscale")
        self.model = CustomCLIP_shembed_lscale(cfg, classnames, clip_model)

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



# ===================================== zeroshot init =======================
class PromptLearner_shembed_zsinit_optimfc(nn.Module):
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
        self.bonder = CrossAttnBlock_v1(ctx_dim, num_heads=8)
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


class CustomCLIP_shembed_zsinit_optimfc(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_shembed_zsinit_optimfc(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.cls_head = ClsHead_v2(cfg, classnames, clip_model)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

        self.text_templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)
        self.zero_shot_init_head(classnames, clip_model)

    def zero_shot_init_head(self, classnames, clip_model, device="cuda"):
        # imagenet_templates = [
        #     "itap of a {}.",
        #     "a bad photo of the {}.",
        #     "a origami {}.",
        #     "a photo of the large {}.",
        #     "a {} in a video game.",
        #     "art of the {}.",
        #     "a photo of the small {}."
        # ]

        # classname:["dog", "dsfa", ...]
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
        self.cls_head.fc.weight.data = weights

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]
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


class Baseline_cattn_vocabloss_shembed_zsinit_optimfc(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_shembed_zsinit_optimfc")
        self.model = CustomCLIP_shembed_zsinit_optimfc(cfg, classnames, clip_model)

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
        # set lr for different groups
        # [{"params": params, "lr": lr}]
        # param_groups = [{"params": self.model.prompt_learner.parameters(), "lr": cfg.OPTIM.LR},
        #                 {"params": self.model.cls_head.parameters(), "lr": cfg.OPTIM.LR * cfg.OPTIM.LR_FC_RATIO}]
        # self.optim = build_optimizer(param_groups, cfg.OPTIM)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.optim_fc = build_optimizer([{"params": self.model.cls_head.parameters(), "lr": cfg.OPTIM.LR * cfg.OPTIM.LR_FC_RATIO}], cfg.OPTIM)    # no scheduler
        self.sched = build_scheduler(self.optim, cfg.OPTIM)

        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("cls_head", self.model.cls_head, self.optim_fc)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def forward(self, image, label=None):
        return self.model(image, label)     # logits


# ===================================== zeroshot init + 2x cattn =======================
class PromptLearner_shembed_zsinit_2xcattn(nn.Module):
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
        self.bonder = CrossAttnBlock_v2(ctx_dim, num_heads=8, depth=2)
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


class CustomCLIP_shembed_zsinit_2xcattn(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_shembed_zsinit_2xcattn(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.cls_head = ClsHead_v2(cfg, classnames, clip_model)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

        self.text_templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)
        self.zero_shot_init_head(classnames, clip_model)

    def zero_shot_init_head(self, classnames, clip_model, device="cuda"):
        # imagenet_templates = [
        #     "itap of a {}.",
        #     "a bad photo of the {}.",
        #     "a origami {}.",
        #     "a photo of the large {}.",
        #     "a {} in a video game.",
        #     "art of the {}.",
        #     "a photo of the small {}."
        # ]

        # classname:["dog", "dsfa", ...]
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
        self.cls_head.fc.weight.data = weights

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]
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


class Baseline_cattn_vocabloss_shembed_zsinit_2xcattn(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_shembed_zsinit_2xcattn")
        self.model = CustomCLIP_shembed_zsinit_2xcattn(cfg, classnames, clip_model)

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



# ===================================== zeroshot init + 2x cattn + pe =======================
class PromptLearner_shembed_zsinit_2xcattn_pe(nn.Module):
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
        self.bonder = CrossAttnBlock_v2_pe(ctx_dim, input_size=(num_query_token, ctx_dim), num_heads=8, depth=2)
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


class CustomCLIP_shembed_zsinit_2xcattn_pe(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_shembed_zsinit_2xcattn_pe(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.cls_head = ClsHead_v2(cfg, classnames, clip_model)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

        self.text_templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)
        self.zero_shot_init_head(classnames, clip_model)

    def zero_shot_init_head(self, classnames, clip_model, device="cuda"):
        # imagenet_templates = [
        #     "itap of a {}.",
        #     "a bad photo of the {}.",
        #     "a origami {}.",
        #     "a photo of the large {}.",
        #     "a {} in a video game.",
        #     "art of the {}.",
        #     "a photo of the small {}."
        # ]

        # classname:["dog", "dsfa", ...]
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
        self.cls_head.fc.weight.data = weights

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]
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


class Baseline_cattn_vocabloss_shembed_zsinit_2xcattn_pe(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_shembed_zsinit_2xcattn_pe")
        self.model = CustomCLIP_shembed_zsinit_2xcattn_pe(cfg, classnames, clip_model)

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



# ===================================== zeroshot init =======================
class PromptLearner_shembed_zsinit_fixedfirst(nn.Module):
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
        self.bonder = CrossAttnBlock_v1(ctx_dim, num_heads=8)
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


class CustomCLIP_shembed_zsinit_fixedfirst(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_shembed_zsinit_fixedfirst(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.cls_head = ClsHead_v2(cfg, classnames, clip_model)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

        self.text_templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)
        self.zero_shot_init_head(classnames, clip_model)

    def zero_shot_init_head(self, classnames, clip_model, device="cuda"):
        # imagenet_templates = [
        #     "itap of a {}.",
        #     "a bad photo of the {}.",
        #     "a origami {}.",
        #     "a photo of the large {}.",
        #     "a {} in a video game.",
        #     "art of the {}.",
        #     "a photo of the small {}."
        # ]

        # classname:["dog", "dsfa", ...]
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
        self.cls_head.fc.weight.data = weights

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]
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


class Baseline_cattn_vocabloss_shembed_zsinit_fixedfirst(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_shembed_zsinit")
        self.model = CustomCLIP_shembed_zsinit_fixedfirst(cfg, classnames, clip_model)

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


# ===================================== zeroshot init =======================
class PromptLearner_shembed_zsinit_textaug(nn.Module):
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
        self.bonder = CrossAttnBlock_v1(ctx_dim, num_heads=8)
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
        templates = get_templates(cfg.DATASET.NAME, cfg.INPUT.TEXT_AUG)

        # label_sentence = ["a photo of a " + n for n in classnames]
        # label_sentence = [templates[0].format(n) for n in classnames]
        self.label = []
        for i in range(len(templates)):
            label_sentence = [templates[i].format(n) for n in classnames]
            tokenized_label = clip.tokenize(label_sentence)
            tokenized_label = tokenized_label[:, 1:1 + num_query_token]
            self.label.append(tokenized_label.masked_fill(tokenized_label == 0, -100).to(device))

        tokenized_query = clip.tokenize(pseudo_query)  # [1,77,512]   only used to locate <eos>, and store prefix suffix
        # mask = mask.unsqueeze(-1).expand(-1, -1, ctx_dim)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_query).type(dtype)  # [1,77,512]
            # label_embed = clip_model.token_embedding(tokenized_label).type(dtype)  # [n_cls, 77, 512]

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + num_query_token:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_query = tokenized_query.to(device)
        self.name_lens = torch.tensor(name_lens)
        self.num_query_token = num_query_token
        self.vocab_size = vocab_size
        self.criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)     # todo:label smoothing?
        self.templates = templates

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
            loss_prompts = 0
            for i in range(len(self.label)):
                target_i = self.label[i][target]  # [B,num_query]
                loss_prompts_i = self.criterion(
                    query_output_vocab.view(-1, self.vocab_size),
                    target_i.view(-1)
                )
                loss_prompts += loss_prompts_i
            loss_prompts /= len(self.label)     # average across all templates
        else:
            loss_prompts = 0    # dummy

        prompts = torch.cat([prefix, query_output, suffix], dim=1)  # [B, 77, 512]

        return prompts, loss_prompts


class CustomCLIP_shembed_zsinit_textaug(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_shembed_zsinit_textaug(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_query
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.cls_head = ClsHead_v2(cfg, classnames, clip_model)

        # shared weights and frozen it
        self.prompt_learner.vocab_head.weight.data = clip_model.token_embedding.weight.data.clone()

        self.text_templates = self.prompt_learner.templates
        self.zero_shot_init_head(classnames, clip_model)

    def zero_shot_init_head(self, classnames, clip_model, device="cuda"):
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
        self.cls_head.fc.weight.data = weights

    def forward(self, image, target=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]
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


class Baseline_cattn_vocabloss_shembed_zsinit_textaug(BaseModel):
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

        self.logger.info("Building Baseline_cattn_vocabloss_shembed_zsinit_textaug")
        self.model = CustomCLIP_shembed_zsinit_textaug(cfg, classnames, clip_model)

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
