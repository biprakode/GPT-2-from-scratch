import torch
from torch import nn
from model.ModelConfig import ModelConfig
from model.PositionalEmbedding import PositionalEmbedding
from model.TrainingConfig import TrainingConfig
from tokenizer.bpe import BPETokenizer
from model.attention import MultiHeadAttention
from model.layernorm import LayerNorm
from model.feedforward import FeedForward
from model.block import TransformerBlock
from torch import Tensor

class GPT2(nn.Module):
    def __init__(self , modelconfig:ModelConfig , trainingconfig:TrainingConfig):
        super().__init__()
        self.wte = nn.Embedding(modelconfig.vocab_size , modelconfig.n_embd)
        self.wpe = PositionalEmbedding(modelconfig)
        self.drop = nn.Dropout(modelconfig.embd_pdrop) # applied to- token_emb + pos_emb
        self.h = nn.ModuleList([TransformerBlock(modelconfig , trainingconfig) for _ in range(modelconfig.n_layer)])
        self.ln_f = LayerNorm(modelconfig)
        self.lm_head = nn.Linear(modelconfig.n_embd , modelconfig.vocab_size , bias=False)

        self.wte.weight = self.lm_head.weight # weight tying (token -> vector -> token)

        self.config = modelconfig
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module , 'c_proj') or hasattr(module , 'final_linear'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)


    def forward(self , input_ids , position_ids , past_key_values=None , use_cache=False):
        batch_size , seq_len = input_ids.size()
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0)  # (1, seq_len)
            position_ids = position_ids.to(input_ids.device)

        token_emb = self.wte(input_ids)
        pos_emb = self.wpe(position_ids)

        hidden_state = token_emb + pos_emb
        hidden_state = self.drop(hidden_state)
        for block in self.h:
            hidden_state = block(hidden_state)

        hidden = self.ln_f(hidden_state)
        logits = self.lm_head(hidden) # batch , seq_len , vocab_size
        return logits

    def generate(self):
        pass


