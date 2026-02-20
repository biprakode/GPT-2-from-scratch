import torch.nn as nn

from model.ModelConfig import ModelConfig
from model.TrainingConfig import TrainingConfig


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig, train_config: TrainingConfig):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(train_config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x