import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from model.TrainingConfig import TrainingConfig
from training.optimizer import configure_optimizer
from training.scheduler import CosineAnnealingScheduler
from training.loss import compute_loss

class Trainer:
    def __init__(self , model , train_loader:DataLoader , validation_loader:DataLoader, scheduler:CosineAnnealingScheduler , train_config:TrainingConfig , optimizer = None, loss = None, device='cpu', use_amp=False):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.loss_fn = loss
        self.scheduler = scheduler
        self.train_config = train_config
        self.device = device
        self.use_amp = use_amp and device != 'cpu'
        self.scaler = GradScaler() if self.use_amp else None

    def train(self , num_epochs , checkpoint_dir ):
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")

            train_loss = self.train_epoch(epoch)
            print(f"\nTrain Loss: {train_loss:.4f}")

            # validate
            val_metrics = self.validate()
            val_loss = val_metrics['val_loss']
            perplexity = val_metrics['perplexity']
            print(f"Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            self.save_checkpoint(epoch, val_loss, checkpoint_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(checkpoint_dir, 'best_model.pt')
                self.save_checkpoint(epoch, val_loss, best_path)
                print(f"New best model! Val loss: {val_loss:.4f}")

        print("\nTraining complete!")
        return best_val_loss

    def train_epoch(self , epoch):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_id'].to(self.device)
            labels = batch['label'].to(self.device)

            if self.use_amp:
                with autocast(device_type='cuda'):
                    logits = self.model(input_ids, position_ids=None)
                    loss_value = self.loss_fn(logits, labels)
                self.scaler.scale(loss_value).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids, position_ids=None)
                loss_value = self.loss_fn(logits, labels)
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()

            total_loss += loss_value.item()

            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Step {batch_idx}: "
                      f"Loss={loss_value.item():.4f}, LR={current_lr:.6f}")

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss


    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.validation_loader:
                input_ids = batch['input_id'].to(self.device)
                labels = batch['label'].to(self.device)
                logits = self.model(input_ids, position_ids=None)
                loss_value = self.loss_fn(logits , labels)
                total_loss += loss_value.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss))

        return {
            'val_loss': avg_loss,
            'perplexity': perplexity
        }

    def save_checkpoint(self , epoch , val_loss , filepath : str):
        checkpoint = {
            'epoch' : epoch,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'scheduler_state_dict' : self.scheduler.state_dict(),
            'val_loss' : val_loss,
            'train_config' : self.train_config,
        }
        torch.save(checkpoint, filepath)
        return checkpoint

    def load_checkpoint(self , filepath : str):
        checkpoint = torch.load(filepath, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']

        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from epoch {start_epoch}")

        return start_epoch, best_val_loss