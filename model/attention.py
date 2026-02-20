import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import FloatTensor, LongTensor, Tensor

class MultiHeadAttention(nn.Module):
    def __init__(self , d_in, d_out, context_length, dropout, num_heads:int):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.d_in = d_in
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_Q = nn.Linear(d_in , d_out)
        self.W_K = nn.Linear(d_in , d_out)
        self.W_V = nn.Linear(d_in , d_out)

        self.final_linear = nn.Linear(d_out , d_out)
        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(context_length , context_length))
        mask = mask.view(1, 1, context_length, context_length)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        self.register_buffer('mask', mask)

    def _split_heads(self , x:Tensor) -> Tensor:
        batch_size , seq_length , n_embd = x.shape
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self , x:Tensor) -> Tensor:
        batch, n_head, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_out)

    def _attn(self , q:Tensor , k:Tensor , v:Tensor , mask:Optional[Tensor] = None) -> Tensor:
        attn_score = torch.matmul(q , k.transpose(-2 , -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_score += mask # no attention to prev tokens

        attn_probs = F.softmax(attn_score , dim = -1)
        attn_probs = self.dropout(attn_probs)
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
        return self.dropout(final_out)

