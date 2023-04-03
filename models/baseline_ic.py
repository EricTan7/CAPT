import logging

import torch
import torch.nn as nn

from solver import build_optimizer, build_scheduler
from .base import BaseModel
from models.head.cls_heads import ClsHead_v2

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .bonder import BertLMHeadModel
from transformers import BertConfig, BertTokenizer

from transformers import BertLMHeadModel

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


class TextEncoder_v3(nn.Module):
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

        # x.shape = [batch_size, n_ctx, transformer.width] [B, 77, 512]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)+name_len] @ self.text_projection

        return x


class PromptLearner_v3(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BASELINE.N_CTX
        ctx_init = cfg.TRAINER.BASELINE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim      # transformer输出的维度（512）  rn50:1024
        vis_witdth = clip_model.visual.width        # transfomer中间的隐层维度（768）  rn50:64，需要乘32变成2048
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
        num_query_token = 32
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vis_witdth      # todo: vision_width?
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        self.bonder = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.max_txt_len = 32

        self.clip_proj = nn.Linear(
            self.bonder.config.hidden_size, clip_model.embed_dim
        )

        if cfg.TRAINER.PREC == "fp16":
            self.bonder.half()

        classnames = [name.replace("_", " ").replace("(", "").replace(")", "") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # print("max length:", max(name_lens))
        # prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompts = prompt_prefix
        template = 'a photo of a '
        self.text = [template+n for n in classnames]

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

    def forward(self, im_features, text=None):  # [B,512] -> [B,5,512]
        prefix = self.token_prefix.expand(im_features.size(0), -1, -1)  # [B,1,512]
        suffix = self.token_suffix.expand(im_features.size(0), -1, -1)  # [B,72,512]
        ctx = self.ctx  # (n_ctx, ctx_dim) [4,512]

        text = self.text

        # 1.im_features, queries   forward
        # todo: im_features should be what? cls token
        image_atts = torch.ones(im_features.size()[:-1], dtype=torch.long)
        query_tokens = self.query_tokens.expand(im_features.shape[0], -1, -1)

        if self.training:
            query_output = self.bonder.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=im_features,  # cross_attention input, serve as decoder
                encoder_attention_mask=image_atts,  # padding mask
                use_cache=True,     # key, value state will be returned to speed up decoding
                return_dict=True,
            )
            # 2. text tokenize
            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(im_features.device)

            decoder_input_ids = text_tokens.input_ids.clone()
            decoder_input_ids[:0] = self.tokenizer.bos_token_id
            labels = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)

            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(im_features.device)
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
            lm_output = self.bonder(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output.past_key_values,  # 传入预先计算好的key, value
                return_dict=True,
                labels=labels,
            )
            loss_ic = lm_output.loss
        else:
            query_output = self.bonder.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=im_features,  # cross_attention input, serve as decoder
                encoder_attention_mask=image_atts,  # padding mask
                return_dict=True,
            )
            loss_ic = 0     # dummy
        prompts = query_output.last_hidden_state  # 应该输出哪一个？last_hidden_state，hidden_states，past_key_values
                                                  # blip2_t5取了last_hidden_state

        prompts = self.clip_proj(prompts)

        return prompts, loss_ic
        # todo: 看blip2_t5中文本编码器输入是否需要适配其prefix、suffix


class CustomCLIP_v3(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_v3(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_v3(clip_model)
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


class Baseline_ic(BaseModel):
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

        self.logger.info("Building custom CLIP")
        self.model = CustomCLIP_v3(cfg, classnames, clip_model)

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
