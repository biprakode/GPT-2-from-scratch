import torch.nn.functional as F
from torch import Tensor

def compute_loss(logits : Tensor, labels : Tensor , label_smoothing : float = 0.1) -> Tensor:
    logits = logits.view(logits.size(0) * logits.size(1), -1)
    labels = labels.view(-1)

    loss = F.cross_entropy(logits, labels , label_smoothing=label_smoothing)
    return loss