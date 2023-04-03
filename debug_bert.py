import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from transformers import BertTokenizer
from transformers import AutoModel


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    return tokenizer

if __name__ == '__main__':
    text = 'I want an apple.'
    tokenizer = init_tokenizer()
    text_tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    a = 1

    decoder_input_ids = text_tokens.input_ids.clone()
    decoder_input_ids[:, 0] = tokenizer.bos_token_id
    labels = decoder_input_ids.masked_fill(
        decoder_input_ids == tokenizer.pad_token_id, -100
    )
    l = labels.view(-1)
    b = 1
    # text_tokens.input_ids
    # text_tokens.attention_mask

    # model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", add_pooling_layer=False)
    # a = 1
    shifted_prediction_scores = torch.randn((3,77,512))
    labels = torch.rand((3,77)).type(torch.long)
    loss_fct = nn.CrossEntropyLoss(reduction='sum', label_smoothing=0.1)
    lm_loss = loss_fct(
        shifted_prediction_scores.view(-1, 512),
        labels.view(-1),
    )
    b = 1