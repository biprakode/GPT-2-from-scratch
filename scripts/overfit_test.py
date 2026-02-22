"""
Overfit test: train a small GPT-2 on a tiny repeated sequence.
If the model is wired correctly, loss should drop to near zero.
Runs on CPU in ~1-2 minutes.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.ModelConfig import ModelConfig
from model.TrainingConfig import TrainingConfig
from model.gpt2 import GPT2
from training.dataset import TextDataset
from training.loss import compute_loss
from training.optimizer import configure_optimizer
from training.scheduler import CosineAnnealingScheduler

# --- tiny configs ---
class SmallConfig(ModelConfig):
    n_embd = 128
    n_layer = 2
    n_head = 4
    block_size = 64
    vocab_size = 256  # byte-level, keep embedding small
    eps = 1e-12
    embd_pdrop = 0.0

class NoDropConfig(TrainingConfig):
    resid_pdrop = 0.0
    attn_pdrop = 0.0

# --- create tiny dataset ---
def make_tiny_bin(path, block_size, num_repeats=200):
    """Write a repeated token pattern so the model can memorize it."""
    pattern = np.arange(1, block_size + 1, dtype=np.uint16)  # [1,2,...,block_size]
    data = np.tile(pattern, num_repeats)
    data.tofile(path)
    return path

def main():
    tmpdir = tempfile.mkdtemp()
    train_path = os.path.join(tmpdir, 'train.bin')
    val_path = os.path.join(tmpdir, 'val.bin')

    block_size = SmallConfig.block_size
    make_tiny_bin(train_path, block_size, num_repeats=200)
    make_tiny_bin(val_path, block_size, num_repeats=20)

    train_dataset = TextDataset(train_path, block_size=block_size)
    val_dataset = TextDataset(val_path, block_size=block_size)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = GPT2(SmallConfig(), NoDropConfig())
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")

    NUM_EPOCHS = 20
    optimizer = configure_optimizer(model, lr=1e-3, weight_decay=0.0)
    num_steps = len(train_loader) * NUM_EPOCHS
    scheduler = CosineAnnealingScheduler(optimizer, warmup_steps=10, total_steps=num_steps, max_lr=1e-3, min_lr=1e-5)

    print(f"\nTraining for {NUM_EPOCHS} epochs ({num_steps} steps) on CPU...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}\n")

    model.train()
    step = 0
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch['input_id']
            labels = batch['label']

            logits = model(input_ids, position_ids=None)
            loss = compute_loss(logits, labels, label_smoothing=0.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1

            if step % 10 == 0:
                print(f"  step {step:4d}  loss={loss.item():.4f}")

        avg = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  avg_loss={avg:.4f}")

    # final validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch['input_id'], position_ids=None)
            val_loss += compute_loss(logits, batch['label'], label_smoothing=0.0).item()
    val_loss /= len(val_loader)
    print(f"\nFinal val loss: {val_loss:.4f}")

    if val_loss < 0.5:
        print("PASS - model can overfit (loss < 0.5)")
    else:
        print("FAIL - loss did not drop enough, something is broken")
        sys.exit(1)

    # cleanup
    os.remove(train_path)
    os.remove(val_path)
    os.rmdir(tmpdir)

if __name__ == '__main__':
    main()