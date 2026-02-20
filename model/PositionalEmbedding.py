from torch import nn
from model.ModelConfig import ModelConfig
from torch import Tensor

class PositionalEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_ctx = config.block_size
        self.n_embd = config.n_embd
        self.wpe = nn.Embedding(self.n_ctx, self.n_embd)

    def forward(self, pos_ids : Tensor) -> Tensor:
        return self.wpe(pos_ids)