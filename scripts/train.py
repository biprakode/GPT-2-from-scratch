"""
GPT-2 training entry point.
Usage:
  python scripts/train.py --data_dir data/tokenized --vocab_dir data/vocab --checkpoint_dir checkpoints
  python scripts/train.py --data_dir /kaggle/working/tokenized --checkpoint_dir /kaggle/working/checkpoints --device cuda --amp
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader

from model.ModelConfig import ModelConfig
from model.TrainingConfig import TrainingConfig
from model.gpt2 import GPT2
from training.dataset import TextDataset
from training.loss import compute_loss
from training.optimizer import configure_optimizer
from training.scheduler import CosineAnnealingScheduler
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train GPT-2')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with train.bin and val.bin')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision (fp16)')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"AMP: {args.amp}")
    print(f"Block size: {args.block_size}")

    # datasets
    train_path = os.path.join(args.data_dir, 'train.bin')
    val_path = os.path.join(args.data_dir, 'val.bin')
    train_dataset = TextDataset(train_path, block_size=args.block_size)
    val_dataset = TextDataset(val_path, block_size=args.block_size)
    print(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(args.device == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(args.device == 'cuda'))

    # model
    model = GPT2(ModelConfig(), TrainingConfig())
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")

    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    model = model.to(args.device)

    # optimizer + scheduler
    optimizer = configure_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingScheduler(optimizer, warmup_steps=args.warmup_steps,
                                          total_steps=total_steps, max_lr=args.lr, min_lr=args.min_lr)

    # trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        scheduler=scheduler,
        train_config=TrainingConfig(),
        optimizer=optimizer,
        loss=compute_loss,
        device=args.device,
        use_amp=args.amp,
    )

    if args.resume:
        start_epoch, best_val_loss = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}, best val loss {best_val_loss:.4f}")

    trainer.train(num_epochs=args.epochs, checkpoint_dir=args.checkpoint_dir)


if __name__ == '__main__':
    main()