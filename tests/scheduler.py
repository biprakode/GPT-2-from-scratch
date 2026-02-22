import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from model.ModelConfig import ModelConfig
from model.TrainingConfig import TrainingConfig
from model.gpt2 import GPT2
from training.optimizer import configure_optimizer
from training.scheduler import CosineAnnealingScheduler


class SmallConfig(ModelConfig):
    n_embd = 64
    n_layer = 2
    n_head = 4
    block_size = 32
    vocab_size = 100
    eps = 1e-12
    embd_pdrop = 0.0


class NoDropTrainConfig(TrainingConfig):
    resid_pdrop = 0.0
    attn_pdrop = 0.0

model = GPT2(SmallConfig, NoDropTrainConfig)
optimizer = configure_optimizer(model)

# Create scheduler
scheduler = CosineAnnealingScheduler(
    optimizer=optimizer,
    warmup_steps=1000,
    total_steps=10000,
    max_lr=0.0006,
    min_lr=0.00006
)

# Track learning rates
lrs = []
for step in range(10000):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

# Plot
plt.plot(lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()