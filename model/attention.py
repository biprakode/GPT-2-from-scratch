import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

from model.ModelConfig import ModelConfig
from model.TrainingConfig import TrainingConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig, train_config: TrainingConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.W_Q = nn.Linear(config.n_embd, config.n_embd)
        self.W_K = nn.Linear(config.n_embd, config.n_embd)
        self.W_V = nn.Linear(config.n_embd, config.n_embd)

        self.final_linear = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(train_config.attn_pdrop)
        self.resid_dropout = nn.Dropout(train_config.resid_pdrop)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        mask = mask.view(1, 1, config.block_size, config.block_size)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        self.register_buffer('mask', mask)

    def _split_heads(self , x:Tensor) -> Tensor:
        batch_size , seq_length , n_embd = x.shape
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, n_head, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.n_embd)

    def _attn(self , q:Tensor , k:Tensor , v:Tensor , mask:Optional[Tensor] = None) -> Tensor:
        attn_score = torch.matmul(q , k.transpose(-2 , -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_score += mask # no attention to masked tokens

        attn_probs = F.softmax(attn_score, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        return torch.matmul(attn_probs , v) #

    def forward(self, x:Tensor) -> Tensor:
        b, num_tokens, d_in = x.shape

        q = self._split_heads(self.W_Q(x))
        k = self._split_heads(self.W_K(x))
        v = self._split_heads(self.W_V(x))

        curr_mask = self.mask[:, :, :num_tokens, :num_tokens]
        attn_probs = self._attn(q , k , v , curr_mask)
        out = self._merge_heads(attn_probs)
        final_out = self.final_linear(out)
        return self.resid_dropout(final_out)

