from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
import torch


class CosineAnnealingScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self):
        if(self.current_step > self.total_steps):
            return self.min_lr
        elif (self.warmup_steps == 0) :
            return self.max_lr

        if self.current_step < self.warmup_steps:
            ratio = self.current_step / self.warmup_steps
            return self.max_lr * ratio
        else:
            steps_into_cosine = self.current_step - self.warmup_steps
            total_cosine_steps = self.total_steps - self.warmup_steps
            progress = torch.tensor(steps_into_cosine/total_cosine_steps)
            return (self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + torch.cos(torch.pi * progress))).item()

    def step(self):
        new_lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.current_step += 1


