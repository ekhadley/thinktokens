import os, time, random, datasets, wandb, functools
wandb.require("core")
from tqdm import trange, tqdm
from dataclasses import dataclass
from typing import *
import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

purple = '\033[95m';blue = '\033[94m';cyan = '\033[96m';lime = '\033[92m';yellow = '\033[93m';red = "\033[38;5;196m";pink = "\033[38;5;206m";orange = "\033[38;5;202m";green = "\033[38;5;34m";gray = "\033[38;5;8m";bold = '\033[1m';underline = '\033[4m';endc = '\033[0m'


_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
_tokenizer.add_special_tokens({'pad_token': '<PAD>'})
def dataset_tokenize_func(examples, max_length=256):
    return _tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

def to_str_tokens(tokenizer, input):
    toks = tokenizer(input)['input_ids']
    return [tokenizer.decode(tok) for tok in toks]
