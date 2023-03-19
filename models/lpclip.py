import logging
import torch
import torch.nn as nn

from solver import build_optimizer, build_scheduler
from .base import BaseModel
from .heads import ClsHead_v1, ClsHead_v2

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


class Linear_Probe(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.cls_head = ClsHead_v1(cfg, classnames, clip_model)

    def forward(self, image):  # image: [B,3,224,224]  label:[B]
        image_features = self.image_encoder(image.type(self.dtype))  # [B,512]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # According to new repo: no L2 norm

        logits = self.cls_head(image_features)  # [B,num_cls]

        return logits


class lpclip(BaseModel):
    def __init__(self, cfg, classnames=None):
        super().__init__()
        self.logger = logging.getLogger(cfg.TRAINER.NAME)   # lpclip
        self.check_cfg(cfg)
        self.cfg = cfg
        self.test_freq = cfg.TRAIN.TEST_FREQ

        self.logger.info(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        self.logger.info("Building Linear Probe CLIP")
        self.model = Linear_Probe(cfg, classnames, clip_model)

        self.logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = ["cls_head"]

        for name, param in self.model.named_parameters():
            if name_to_update[0] in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        self.logger.info(f"Parameters to be updated: {enabled}")

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.cls_head, cfg.OPTIM)
        self.sched = build_scheduler(self.optim, cfg.OPTIM)

        self.register_model("cls_head", self.model.cls_head, self.optim, self.sched)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def forward(self, image):
        return self.model(image)     # logits
