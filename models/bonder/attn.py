import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from timm.models.layers import DropPath, Mlp
import clip


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Transformer Encoder Block which adopt self-attention
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    """
    Cross-attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_text = nn.LayerNorm(dim)
        self.norm_img = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, query, kv):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        # 1. project q, insert a learnable layer
        query = self.q(query)
        k, v = kv, kv

        attn = (query @ k.transpose(1,2)) * self.scale   # [B, n_query, n_tkn]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        query = attn @ v    # [B, n_query, dim]
        query = self.proj(query)
        query = self.proj_drop(query)

        return query


class CrossAttention_projkv(nn.Module):
    """
    Cross-attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_text = nn.LayerNorm(dim)
        self.norm_img = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.dim = dim

    def forward(self, query, kv):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        # 1. project q, insert a learnable layer
        query = self.q(query)
        kv = self.kv(kv)
        k = kv[:, :, :self.dim]
        v = kv[:, :, self.dim:]

        attn = (query @ k.transpose(1,2)) * self.scale   # [B, n_query, n_tkn]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        query = attn @ v    # [B, n_query, dim]
        query = self.proj(query)
        query = self.proj_drop(query)

        return query


class CrossAttention_projk(nn.Module):
    """
    Cross-attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_text = nn.LayerNorm(dim)
        self.norm_img = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.dim = dim

    def forward(self, query, kv):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        # 1. project q, insert a learnable layer
        query = self.q(query)
        k = self.k(kv)
        # kv = self.kv(kv)
        # k = kv[:, :, :self.dim]
        # v = kv[:, :, self.dim:]

        attn = (query @ k.transpose(1,2)) * self.scale   # [B, n_query, n_tkn]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        query = attn @ kv    # [B, n_query, dim]
        query = self.proj(query)
        query = self.proj_drop(query)

        return query

class CrossAttention_projkv_bert(nn.Module):
    """
    Cross-attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kv_dim=512):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_text = nn.LayerNorm(dim)
        self.norm_img = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(kv_dim, dim*2, bias=qkv_bias)
        self.dim = dim
        self.kv_dim = kv_dim

    def forward(self, query, kv):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        # 1. project q, insert a learnable layer
        query = self.q(query)
        kv = self.kv(kv)
        k = kv[:, :, :self.dim]
        v = kv[:, :, self.dim:]

        attn = (query @ k.transpose(1,2)) * self.scale   # [B, n_query, n_tkn]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        query = attn @ v    # [B, n_query, dim]
        query = self.proj(query)
        query = self.proj_drop(query)

        return query


# todo: self-attention -> cross-attention
# todo: cross-attention -> self-attention
# todo: flamingo
class CrossAttnBlock(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(CrossAttnBlock, self).__init__()
        self.self_attn_layer = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.mlp = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        self.norm1 = norm_layer(dim)
        self.norm2_q = norm_layer(dim)
        self.norm2_kv = norm_layer(dim)

        self.norm_mlp = norm_layer(dim)

    def forward(self, query, img_fea):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        # LN + self-attn + residual
        query = query + self.self_attn_layer(self.norm1(query))     # flamingo: + tanh(alpha)

        # LN + cross-attn + residual
        query = query + self.cross_attn(self.norm2_q(query), self.norm2_kv(img_fea))

        # LN + MLP + residual
        # text_tokens = text_tokens + self.mlp_text(self.norm_final(text_tokens))
        # img_tokens = img_tokens + self.mlp_img(self.norm_final(img_tokens))
        query = query + self.mlp(self.norm_mlp(query))

        return query    # [B, n_query, dim]


class CrossAttention_sam(nn.Module):
    """
    Cross-attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv_text = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_img = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_text = nn.LayerNorm(dim)
        self.norm_img = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(256, dim, bias=qkv_bias)     # the output dim of SAM is 256
        self.v = nn.Linear(256, dim, bias=qkv_bias)

    def forward(self, query, kv):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        # 1. project q, insert a learnable layer
        query = self.q(query)
        k = self.k(kv)
        v = self.v(kv)

        attn = (query @ k.transpose(1,2)) * self.scale   # [B, n_query, n_tkn]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        query = attn @ v    # [B, n_query, dim]
        query = self.proj(query)
        query = self.proj_drop(query)

        return query


# todo: self-attention -> cross-attention
# todo: cross-attention -> self-attention
# todo: flamingo
class CrossAttnBlock_sam(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(CrossAttnBlock_sam, self).__init__()
        self.self_attn_layer = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention_sam(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.mlp = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        self.norm1 = norm_layer(dim)
        self.norm2_q = norm_layer(dim)
        self.norm2_kv = norm_layer(256)

        self.norm_mlp = norm_layer(dim)

    def forward(self, query, img_fea):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        # LN + self-attn + residual
        query = query + self.self_attn_layer(self.norm1(query))     # flamingo: + tanh(alpha)

        # LN + cross-attn + residual
        query = query + self.cross_attn(self.norm2_q(query), self.norm2_kv(img_fea))

        # LN + MLP + residual
        # text_tokens = text_tokens + self.mlp_text(self.norm_final(text_tokens))
        # img_tokens = img_tokens + self.mlp_img(self.norm_final(img_tokens))
        query = query + self.mlp(self.norm_mlp(query))

        return query    # [B, n_query, dim]


class CrossAttnBlock_projkv_pe_rn(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, input_size=(32, 512), num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kv_dim=2048):
        super(CrossAttnBlock_projkv_pe_rn, self).__init__()
        self.self_attn_layer = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention_projkv_bert(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, kv_dim=kv_dim)

        self.mlp = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        self.norm1 = norm_layer(dim)
        self.norm2_q = norm_layer(dim)
        self.norm2_kv = norm_layer(kv_dim)

        self.norm_mlp = norm_layer(dim)

        self.pos_embed = nn.Parameter(torch.zeros(input_size))

    def forward(self, query, img_fea):
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.expand(query.size(0), -1, -1)
            query = query + pos_embed

        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        # LN + self-attn + residual
        query = query + self.self_attn_layer(self.norm1(query))     # flamingo: + tanh(alpha)

        # LN + cross-attn + residual
        query = query + self.cross_attn(self.norm2_q(query), self.norm2_kv(img_fea))

        # LN + MLP + residual
        # text_tokens = text_tokens + self.mlp_text(self.norm_final(text_tokens))
        # img_tokens = img_tokens + self.mlp_img(self.norm_final(img_tokens))
        query = query + self.mlp(self.norm_mlp(query))

        return query    # [B, n_query, dim]


class CrossAttnBlock_projkv(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(CrossAttnBlock_projkv, self).__init__()
        self.self_attn_layer = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention_projkv(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.mlp = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        self.norm1 = norm_layer(dim)
        self.norm2_q = norm_layer(dim)
        self.norm2_kv = norm_layer(dim)

        self.norm_mlp = norm_layer(dim)

    def forward(self, query, img_fea):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        # LN + self-attn + residual
        query = query + self.self_attn_layer(self.norm1(query))     # flamingo: + tanh(alpha)

        # LN + cross-attn + residual
        query = query + self.cross_attn(self.norm2_q(query), self.norm2_kv(img_fea))

        # LN + MLP + residual
        # text_tokens = text_tokens + self.mlp_text(self.norm_final(text_tokens))
        # img_tokens = img_tokens + self.mlp_img(self.norm_final(img_tokens))
        query = query + self.mlp(self.norm_mlp(query))

        return query    # [B, n_query, dim]


class CrossAttnBlock_projk(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(CrossAttnBlock_projk, self).__init__()
        self.self_attn_layer = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention_projkv(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.mlp = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        self.norm1 = norm_layer(dim)
        self.norm2_q = norm_layer(dim)
        self.norm2_kv = norm_layer(dim)

        self.norm_mlp = norm_layer(dim)

    def forward(self, query, img_fea):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        # LN + self-attn + residual
        query = query + self.self_attn_layer(self.norm1(query))     # flamingo: + tanh(alpha)

        # LN + cross-attn + residual
        query = query + self.cross_attn(self.norm2_q(query), self.norm2_kv(img_fea))

        # LN + MLP + residual
        # text_tokens = text_tokens + self.mlp_text(self.norm_final(text_tokens))
        # img_tokens = img_tokens + self.mlp_img(self.norm_final(img_tokens))
        query = query + self.mlp(self.norm_mlp(query))

        return query    # [B, n_query, dim]


# ==================== nx cattn =======================
class CrossAttnBlock_nx(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=1):
        super(CrossAttnBlock_nx, self).__init__()
        # self.blk = nn.Sequential(
        #     *[CrossAttnBlock_v1(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
        #                         drop_path, act_layer, norm_layer) for _ in range(depth)]
        # )
        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = CrossAttnBlock(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                drop_path, act_layer, norm_layer)
            self.blocks.append(blk)

    def forward(self, query, img_fea):
        # if self.pos_embed is not None:
        #     x = x + self.pos_embed

        for blk in self.blocks:
            query = blk(query, img_fea)

        return query    # [B, n_query, dim]


class CrossAttnBlock_nx_projkv(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=1):
        super(CrossAttnBlock_nx_projkv, self).__init__()
        # self.blk = nn.Sequential(
        #     *[CrossAttnBlock_v1(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
        #                         drop_path, act_layer, norm_layer) for _ in range(depth)]
        # )
        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = CrossAttnBlock_projkv(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                drop_path, act_layer, norm_layer)
            self.blocks.append(blk)

    def forward(self, query, img_fea):
        # if self.pos_embed is not None:
        #     x = x + self.pos_embed

        for blk in self.blocks:
            query = blk(query, img_fea)

        return query    # [B, n_query, dim]


class CrossAttnBlock_nx_pe(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, input_size=(32, 512), num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=1):
        super(CrossAttnBlock_nx_pe, self).__init__()
        # self.blk = nn.Sequential(
        #     *[CrossAttnBlock_v1(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
        #                         drop_path, act_layer, norm_layer) for _ in range(depth)]
        # )
        self.pos_embed = nn.Parameter(torch.zeros(input_size))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = CrossAttnBlock(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                drop_path, act_layer, norm_layer)
            self.blocks.append(blk)

    def forward(self, query, img_fea):
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.expand(query.size(0), -1, -1)
            query = query + pos_embed

        for blk in self.blocks:
            query = blk(query, img_fea)

        return query    # [B, n_query, dim]


class CrossAttnBlock_nx_projk_pe(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, input_size=(32, 512), num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=1):
        super(CrossAttnBlock_nx_projk_pe, self).__init__()

        self.pos_embed = nn.Parameter(torch.zeros(input_size))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = CrossAttnBlock_projk(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                drop_path, act_layer, norm_layer)
            self.blocks.append(blk)

    def forward(self, query, img_fea):
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.expand(query.size(0), -1, -1)
            query = query + pos_embed

        for blk in self.blocks:
            query = blk(query, img_fea)

        return query    # [B, n_query, dim]


class CrossAttnBlock_nx_pe_auxi(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, input_size=(32, 512), num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=1):
        super(CrossAttnBlock_nx_pe_auxi, self).__init__()
        # self.blk = nn.Sequential(
        #     *[CrossAttnBlock_v1(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
        #                         drop_path, act_layer, norm_layer) for _ in range(depth)]
        # )
        self.pos_embed = nn.Parameter(torch.zeros(input_size))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = CrossAttnBlock(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                drop_path, act_layer, norm_layer)
            self.blocks.append(blk)

    def forward(self, query, img_fea, instance_target=None, criterion=None, vocab_size=None, vocab_head=None, is_training=False):
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.expand(query.size(0), -1, -1)
            query = query + pos_embed

        loss_instance = torch.tensor(0.).to(query.device)
        for blk in self.blocks:
            query = blk(query, img_fea)
            if is_training:
                query_vocab = vocab_head(query)  # [B, num_query, vocab_size]
                loss_instance += criterion(
                    query_vocab.view(-1, vocab_size),
                    instance_target.view(-1)
                )

        return query, loss_instance    # [B, n_query, dim]


class CrossAttnBlock_nx_projk_pe_auxi(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, input_size=(32, 512), num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=1):
        super(CrossAttnBlock_nx_projk_pe_auxi, self).__init__()
        # self.blk = nn.Sequential(
        #     *[CrossAttnBlock_v1(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
        #                         drop_path, act_layer, norm_layer) for _ in range(depth)]
        # )
        self.pos_embed = nn.Parameter(torch.zeros(input_size))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = CrossAttnBlock_projk(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                drop_path, act_layer, norm_layer)
            self.blocks.append(blk)

    def forward(self, query, img_fea, instance_target=None, criterion=None, vocab_size=None, vocab_head=None, is_training=False):
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.expand(query.size(0), -1, -1)
            query = query + pos_embed

        loss_instance = torch.tensor(0.).to(query.device)
        loss_final = torch.tensor(0.).to(query.device)
        for blk in self.blocks:
            query = blk(query, img_fea)
            if is_training:
                query_vocab = vocab_head(query)  # [B, num_query, vocab_size]
                loss_final = criterion(
                    query_vocab.view(-1, vocab_size),
                    instance_target.view(-1)
                )
                loss_instance += loss_final

        return query, loss_instance, loss_final    # [B, n_query, dim]



class CrossAttnBlock_projkv_pe_bert(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, input_size=(32, 512), num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kv_dim=512):
        super(CrossAttnBlock_projkv_pe_bert, self).__init__()
        self.self_attn_layer = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention_projkv_bert(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, kv_dim=kv_dim)

        self.mlp = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        self.norm1 = norm_layer(dim)
        self.norm2_q = norm_layer(dim)
        self.norm2_kv = norm_layer(kv_dim)

        self.norm_mlp = norm_layer(dim)

        self.pos_embed = nn.Parameter(torch.zeros(input_size))

    def forward(self, query, img_fea):
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.expand(query.size(0), -1, -1)
            query = query + pos_embed

        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        # LN + self-attn + residual
        query = query + self.self_attn_layer(self.norm1(query))     # flamingo: + tanh(alpha)

        # LN + cross-attn + residual
        query = query + self.cross_attn(self.norm2_q(query), self.norm2_kv(img_fea))

        # LN + MLP + residual
        # text_tokens = text_tokens + self.mlp_text(self.norm_final(text_tokens))
        # img_tokens = img_tokens + self.mlp_img(self.norm_final(img_tokens))
        query = query + self.mlp(self.norm_mlp(query))

        return query    # [B, n_query, dim]


class CrossAttnBlock_single_query(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(CrossAttnBlock_single_query, self).__init__()
        self.self_attn_layer = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.mlp = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        self.norm1 = norm_layer(dim)
        self.norm2_q = norm_layer(dim)
        self.norm2_kv = norm_layer(dim)

        self.norm_mlp = norm_layer(dim)

    def forward(self, query, img_fea):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        query = query.expand(img_fea.size(0), -1, -1)
        # LN + self-attn + residual
        query = query + self.self_attn_layer(self.norm1(query))     # flamingo: + tanh(alpha)

        # LN + cross-attn + residual
        query = query + self.cross_attn(self.norm2_q(query), self.norm2_kv(img_fea))

        # LN + MLP + residual
        # text_tokens = text_tokens + self.mlp_text(self.norm_final(text_tokens))
        # img_tokens = img_tokens + self.mlp_img(self.norm_final(img_tokens))
        query = query + self.mlp(self.norm_mlp(query))

        return query    # [B, n_query, dim]


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    clip_model, preprocess = clip.load("ViT-B/32")

    # 2. preprocess image
    image = torch.rand((4,3,224,224)).cuda()
    # image_input = preprocess(image).cuda()
    # 3. tokenize text
    # text_descriptions = [f"This is a photo of a {label}" for label in labels]  # maybe manual text prompt
    text_descriptions = [f"This is a photo of a pedestrian"]  # maybe manual text prompt    # todo: 9 -> padding -> 77  [sot_token, ..., eot_token]
    text_tokens = clip.tokenize(text_descriptions).cuda()   # [1,77]
    # print(text_tokens)
    b = text_tokens.argmax(dim=-1) + 1


    # clip_model, preprocess = clip.load("ViT-B/32")
    # for param in clip_model.parameters():
    #     param.requires_grad=False

    # todo:
    # 1. manual prompt
    # 2. batch problem
    # 3. start train
    prompt_templates = [
        'The pedestrian in the photo is {}.'
    ]

    a = 1
