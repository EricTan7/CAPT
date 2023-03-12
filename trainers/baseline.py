import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

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

        if cfg.TRAINER.BASELINE.PREC == "fp16":
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

    def forward(self, im_features):  # [1,512]
        prefix = self.token_prefix  # [50,1,512]->[1,1,512]
        suffix = self.token_suffix  # [50,72,512]->[1,72,512]  suffix contain padding tokens
        ctx = self.ctx  # (n_ctx, ctx_dim) [4,512]
        bias = self.meta_net(im_features)  # (batch, ctx_dim) [1,512]
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim) [B,1,512]    per image, 100 images in total
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim) [1,4,512]      shared for all images
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim) [B,4,512]

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0)  # [1,4,512]
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)  # [B,1,77,512]

        return prompts


class ClsHead_v1(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.fuse = cfg.TRAINER.BASELINE.FUSE
        assert self.fuse in ["cat", "mul"]
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)

        if self.fuse == 'cat':
            in_fea = vis_dim * 2
        elif self.fuse == 'mul':
            in_fea = vis_dim
        self.fc = nn.Linear(in_fea, n_cls, bias=True)
        # self.fc = nn.Sequential(
        #     nn.Linear(in_fea, in_fea//2, bias=True),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_fea//2, n_cls, bias=True),
        # )

        # if cfg.TRAINER.BASELINE.PREC == "fp16":
        #     self.fc.half()
        # todo: fc design

    def forward(self, text_fea, image_fea):   # [1,512] [1,512]
        # 1. fuse features
        if self.fuse == 'cat':
            fused_fea = torch.cat([text_fea, image_fea], dim=1)     # [1,1024]
        elif self.fuse == 'mul':
            fused_fea = text_fea * image_fea    # [1,512]

        # 2. classification
        logits = self.fc(fused_fea)
        # out = F.softmax(logits)

        return logits


class ClsHead_v2(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.fuse = cfg.TRAINER.BASELINE.FUSE
        assert self.fuse in ["cat", "mul"]
        vis_dim = clip_model.visual.output_dim
        n_cls = len(classnames)

        if self.fuse == 'cat':
            in_fea = vis_dim * 2
        elif self.fuse == 'mul':
            in_fea = vis_dim
        self.fc = nn.Sequential(
            nn.Linear(in_fea, in_fea//2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_fea//2, n_cls, bias=True),
        )

    def forward(self, text_fea, image_fea):   # [1,512] [1,512]
        # 1. fuse features
        if self.fuse == 'cat':
            fused_fea = torch.cat([text_fea, image_fea], dim=1)     # [1,1024]
        elif self.fuse == 'mul':
            fused_fea = text_fea * image_fea    # [1,512]

        # 2. classification
        logits = self.fc(fused_fea)
        # out = F.softmax(logits)

        return logits


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

        # n_cls = len(classnames)
        # self.test_cls_head = nn.Linear(512, n_cls)
        # self.test_cls_head.half()

    def forward(self, image, label=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [50,77]  50 base classes
        logit_scale = self.logit_scale.exp()  # [B]

        image_features = self.image_encoder(image.type(self.dtype))  # [B,512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [B,512]
        # image_features = image_features.float()
        # logits = self.test_cls_head(image_features)  # [B,50]

        prompts = self.prompt_learner(image_features)  # [B,50,77,512]->[B,1,77,512]

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # [B,512]

            text_features, imf_i = text_features.float(), imf_i.float()
            logits.append(self.cls_head(text_features, imf_i.unsqueeze(0)))   # cat / ele_mul
        logits = torch.stack(logits)    # [B,1,50]
        logits = logits.squeeze(1)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


class CustomCLIP_v2(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_v1(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.cls_head = ClsHead_v2(cfg, classnames, clip_model)

    def forward(self, image, label=None):  # image: [B,3,224,224]  label:[B]
        tokenized_prompts = self.tokenized_prompts  # [50,77]  50 base classes
        logit_scale = self.logit_scale.exp()  # [B]

        image_features = self.image_encoder(image.type(self.dtype))  # [B,512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [B,512]

        prompts = self.prompt_learner(image_features)  # [B,50,77,512]->[B,1,77,512]
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # [B,512]

            text_features, imf_i = text_features.float(), imf_i.float()
            logits.append(self.cls_head(text_features, imf_i.unsqueeze(0)))   # cat / ele_mul
        logits = torch.stack(logits)    # [B,1,50]
        logits = logits.squeeze(1)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


@TRAINER_REGISTRY.register()
class Baseline(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BASELINE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.test_freq = cfg.TRAIN.TEST_FREQ

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BASELINE.PREC == "fp32" or cfg.TRAINER.BASELINE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP_v2(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
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
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        # todo: remember to modify optim
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.optim2 = build_optimizer(self.model.cls_head, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.sched2 = build_lr_scheduler(self.optim2, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("cls_head", self.model.cls_head, self.optim2, self.sched2)

        self.scaler = GradScaler() if cfg.TRAINER.BASELINE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        # self.log_ctx = []
        # self.log_meta = []
        # self.log_head = []

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        optim2 = self.optim2
        scaler = self.scaler

        prec = self.cfg.TRAINER.BASELINE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            optim2.zero_grad()
            loss.backward()
            optim.step()
            optim2.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        if (self.batch_idx + 1) == self.num_batches and (self.epoch+1) % self.test_freq == 0:
            self.test()
            self.set_model_mode("train")

        # if (self.batch_idx + 1) == self.num_batches:
        #     self.log_ctx.append(self.model.prompt_learner.ctx[0, :16].clone().detach())
        #     self.log_meta.append(self.model.prompt_learner.meta_net.linear1.weight.data[0, :16].clone().detach())
        #     self.log_head.append(self.model.cls_head.fc.weight.data[0, :16].clone().detach())
        #     with open('ctx_baseline_addhead.txt', 'w') as f:
        #         for item in self.log_ctx:
        #             f.write(str(item)+'\n')
        #
        #     with open('meta_baseline_addhead.txt', 'w') as f:
        #         for item in self.log_meta:
        #             f.write(str(item)+'\n')
        #
        #     with open('head_baseline_addhead.txt', 'w') as f:
        #         for item in self.log_head:
        #             f.write(str(item)+'\n')

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

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
