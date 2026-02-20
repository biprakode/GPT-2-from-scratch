from torch import nn
import torch
from model.ModelConfig import ModelConfig

class LayerNorm(nn.Module):
    def __init__(self , config:ModelConfig):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(config.n_embd))
        self.beta = nn.Parameter(torch.zeros(config.n_embd))
        self.eps = config.eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True , unbiased=False)
        norm_x = (x - mean) / (torch.sqrt(var + self.eps))
        return self.gamma * norm_x + self.beta
