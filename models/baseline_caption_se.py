import sys
sys.path.insert(0, '.')
import logging
import os.path as osp
from tools.model import load_checkpoint

import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath, Mlp

from solver import build_optimizer, build_scheduler, build_scheduler_iter
from .base import BaseModel
from models.head import *
from .bonder import CrossAttnBlock, CrossAttnBlock_nx, CrossAttnBlock_projkv, CrossAttnBlock_nx_projkv, \
    CrossAttnBlock_nx_pe, CrossAttnBlock_nx_pe_auxi, CrossAttnBlock_nx_projk_pe_auxi, CrossAttnBlock_nx_projk_pe, \
    CrossAttnBlock_projkv_pe_rn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.simple_tokenizer import MySimpleTokenizer
from datasets.templates import get_templates
from tools.model import load_pretrained_weights

_tokenizer = _Tokenizer()

dataset_name = {
    "OxfordPets": "oxford_pets",
    "OxfordFlowers": "oxford_flowers",
    "FGVCAircraft": "fgvc_aircraft",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "StanfordCars": "stanford_cars",
    "Food101": "food-101",
    "SUN397": "sun397",
    "Caltech101": "caltech-101",
    "UCF101": "ucf101",
    "ImageNet": "imagenet",
    "ImageNetV2": "imagenetv2",
    "ImageNetSketch": "imagenet-sketch",
    "ImageNetA": "imagenet-adversarial",
    "ImageNetR": "imagenet-rendition"
}


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
    

class PromptLearner_caption(nn.Module):
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

        num_q_category = 32
        num_q_instance = cfg.MODEL.BONDER.NUM_Q
        self.bonder_category = CrossAttnBlock(ctx_dim, num_heads=8)
        self.bonder_instance = CrossAttnBlock(ctx_dim, num_heads=8)
        self.query_category = nn.Parameter(torch.zeros(1, num_q_category, ctx_dim))     # NOTE category-wise prompt use 32 query
        self.query_instance = nn.Parameter(torch.zeros(1, num_q_instance, ctx_dim))
        self.query_category.data.normal_(mean=0.0, std=0.02)
        self.query_instance.data.normal_(mean=0.0, std=0.02)
        print("Using 2 prompting streams.")
        print(f"Number of category-wise queries: {num_q_category}")
        print(f"Number of instance-wise queries: {num_q_instance}")
        self.vocab_head = nn.Linear(transformer_width, vocab_size, bias=False)

        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        self.query_category.data[:, :n_ctx] = ctx_vectors[:, :]  # "a photo of a " initialization
        print(f"Initialize category query with: {ctx_init}")

        print(f"Random initialization for instance query!")

        if cfg.TRAINER.PREC == "fp16":
            self.bonder_category.half()
            self.bonder_instance.half()

        classnames = [name.replace("_", " ").replace("(", " ").replace(")", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        pseudo_query_category = ["X" + " X" * (num_q_category - 1) + '.']   # only used to locate <eos>, and store prefix suffix
        pseudo_query_instance = ["X" + " X" * (num_q_instance - 1) + '.']
        label_sentence = [f"a photo of a {n}." for n in classnames]

        tokenized_query_category = clip.tokenize(pseudo_query_category)  # [1,77,512]   only used to locate <eos>, and store prefix suffix
        tokenized_query_instance = clip.tokenize(pseudo_query_instance)
        tokenized_label = clip.tokenize(label_sentence)     # category
        tokenized_label = tokenized_label[:, 1:1 + num_q_category]
        with torch.no_grad():
            embedding_category = clip_model.token_embedding(tokenized_query_category).type(dtype)  # [1,77,512]
            embedding_instance = clip_model.token_embedding(tokenized_query_instance).type(dtype)  # [1,77,512]

        self.register_buffer("token_prefix_cat", embedding_category[:, :1, :])  # SOS
        self.register_buffer("token_suffix_cat", embedding_category[:, 1 + num_q_category:, :])  # EOS
        self.register_buffer("token_prefix_ins", embedding_instance[:, :1, :])  # SOS
        self.register_buffer("token_suffix_ins", embedding_instance[:, 1 + num_q_instance:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.device = device
        self.tokenized_query_category = tokenized_query_category.to(device)
        self.tokenized_query_instance = tokenized_query_instance.to(device)
        self.label = tokenized_label.masked_fill(tokenized_label == 0, -100).to(device)
        self.name_lens = torch.tensor(name_lens)
        self.num_query_token = num_q_instance
        self.vocab_size = vocab_size
        self.criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)

    def forward(self, im_features, im_cls, target=None, caption=None):  # [B,512] -> [B,5,512]
        prefix_cat = self.token_prefix_cat.expand(im_features.size(0), -1, -1)  # [B,1,512]
        suffix_cat = self.token_suffix_cat.expand(im_features.size(0), -1, -1)  # [B,77-num_query,512]
        prefix_ins = self.token_prefix_ins.expand(im_features.size(0), -1, -1)  # [B,1,512]
        suffix_ins = self.token_suffix_ins.expand(im_features.size(0), -1, -1)  # [B,77-num_query,512]

        # category-wise
        query_category = self.query_category.expand(im_features.size(0), -1, -1).clone()
        query_category.data[:, self.n_ctx] = im_cls[:, :]
        query_output_category = self.bonder_category(self.query_category, im_features)  # [B, num_query, dim]
        query_output_vocab_category = self.vocab_head(query_output_category)  # [B, num_query, vocab_size]

        # instance-wise
        query_instance = self.query_instance.expand(im_features.size(0), -1, -1).clone()
        query_instance.data[:, self.n_ctx] = im_cls[:, :]
        query_output_instance = self.bonder_instance(self.query_instance, im_features)  # [B, num_query, dim]
        query_output_vocab_instance = self.vocab_head(query_output_instance)  # [B, num_query, vocab_size]

        if self.training:
            # category-wise
            category_target = self.label[target]  # [B,num_q]
            loss_category = self.criterion(
                query_output_vocab_category.view(-1, self.vocab_size),
                category_target.view(-1)
            )

            # instance-wise
            caption = caption[:, 1:1 + self.num_query_token]    # [B, num_q]
            instance_target = caption.masked_fill(caption == 0, -100).to(self.device)
            loss_instance = self.criterion(
                query_output_vocab_instance.view(-1, self.vocab_size),
                instance_target.view(-1)
            )
            # loss_prompts = loss_category + loss_instance
        else:
            loss_category, loss_instance = 0, 0    # dummy

        prompts_category = torch.cat([prefix_cat, query_output_category, suffix_cat], dim=1)  # [B, 77, 512]
        prompts_instance = torch.cat([prefix_ins, query_output_instance, suffix_ins], dim=1)

        return prompts_category, prompts_instance, loss_category, loss_instance


class CustomCLIP_caption_se_pre_all(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        device = 'cuda'
        self.prompt_learner = PromptLearner_caption(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = torch.tensor(4.60517)
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        # TODO if use, merge this into cls_head
        self.se_img = se_layer(clip_model.visual.output_dim, reduction=16)
        self.se_text = se_layer(clip_model.visual.output_dim, reduction=16)

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

    def forward(self, image, target=None, caption=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts_category = self.prompt_learner.tokenized_query_category  # [1, 77]
        tokenized_prompts_instance = self.prompt_learner.tokenized_query_instance  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]

        prompts_category, prompts_instance, loss_category, loss_instance = self.prompt_learner(image_features, image_cls, target, caption)  # [B, 77, 1024]

        text_features_category = self.text_encoder(prompts_category, tokenized_prompts_category)   # [B,1024]
        text_features_instance = self.text_encoder(prompts_instance, tokenized_prompts_instance)  # [B,1024]
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_category, text_features_instance = text_features_category.float(), text_features_instance.float()
        image_cls = image_cls.float()

        # NOTE ensemble over embedding space
        text_features = (text_features_category + text_features_instance) / 2.
        image_cls = self.se_img(image_cls)
        text_features = self.se_text(text_features)

        fused_fea = torch.cat([image_cls, text_features], dim=1)
        logits = self.cls_head(fused_fea)
        if not self.prompt_learner.training:
            logits_wiseft = self.wiseft_head(fused_fea)
            logits_wiseft2 = self.wiseft_head2(fused_fea)
            return logits, logits_wiseft, logits_wiseft2

        return logits, loss_category, loss_instance


class Baseline_caption_wiseft_se_pre_all(BaseModel):
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

        self.logger.info("Building Baseline_caption_wiseft_se_pre_all")
        self.model = CustomCLIP_caption_se_pre_all(cfg, classnames, clip_model)

        self.logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = ["prompt_learner", "cls_head", "se_img", "se_text"]

        for name, param in self.model.named_parameters():
            if (name_to_update[0] in name) or (name_to_update[1] in name) or (name_to_update[2] in name) or (name_to_update[3] in name):
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
        self.register_model("se_img", self.model.se_img)
        self.register_model("se_text", self.model.se_text)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def forward(self, image, label=None, caption=None):
        return self.model(image, label, caption)     # logits



class CustomCLIP_caption_se_post(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        device = 'cuda'
        self.prompt_learner = PromptLearner_caption(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = torch.tensor(4.60517)
        self.dtype = clip_model.dtype
        self.backbone = cfg.MODEL.BACKBONE.NAME

        self.se = se_layer(clip_model.visual.output_dim*2, reduction=16)

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

    def forward(self, image, target=None, caption=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts_category = self.prompt_learner.tokenized_query_category  # [1, 77]
        tokenized_prompts_instance = self.prompt_learner.tokenized_query_instance  # [1, 77]
        logit_scale = self.logit_scale.exp()  # [B]

        image_features, image_cls = self.image_encoder(image.type(self.dtype))
        # rn50: [B,2048,7,7]
        if self.backbone.startswith('RN'):
            image_features = image_features.view(image_features.size(0), image_features.size(1), -1).permute(0,2,1)     # [B,49,2048]
        # vitb16: [B,197,512]

        prompts_category, prompts_instance, loss_category, loss_instance = self.prompt_learner(image_features, image_cls, target, caption)  # [B, 77, 1024]

        text_features_category = self.text_encoder(prompts_category, tokenized_prompts_category)   # [B,1024]
        text_features_instance = self.text_encoder(prompts_instance, tokenized_prompts_instance)  # [B,1024]
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_category, text_features_instance = text_features_category.float(), text_features_instance.float()
        image_cls = image_cls.float()

        # NOTE ensemble over embedding space
        text_features = (text_features_category + text_features_instance) / 2.

        fused_fea = torch.cat([image_cls, text_features], dim=1)
        fused_fea = self.se(fused_fea)
        logits = self.cls_head(fused_fea)
        if not self.prompt_learner.training:
            logits_wiseft = self.wiseft_head(fused_fea)
            logits_wiseft2 = self.wiseft_head2(fused_fea)
            return logits, logits_wiseft, logits_wiseft2

        return logits, loss_category, loss_instance


class Baseline_caption_wiseft_se_post(BaseModel):
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

        self.logger.info("Building Baseline_caption_wiseft_se_post")
        self.model = CustomCLIP_caption_se_post(cfg, classnames, clip_model)

        self.logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = ["prompt_learner", "cls_head", "se"]

        for name, param in self.model.named_parameters():
            if (name_to_update[0] in name) or (name_to_update[1] in name) or (name_to_update[2] in name):
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

        self.optim = build_optimizer([self.model.prompt_learner, self.model.cls_head, self.model.se], cfg.OPTIM)
        self.sched = build_scheduler_iter(self.optim, cfg.OPTIM)

        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("cls_head", self.model.cls_head)
        self.register_model("se", self.model.se)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def forward(self, image, label=None, caption=None):
        return self.model(image, label, caption)     # logits