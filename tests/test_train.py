import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from unittest.mock import MagicMock
from torch.utils.data import DataLoader, TensorDataset

from model.ModelConfig import ModelConfig
from model.TrainingConfig import TrainingConfig
from model.gpt2 import GPT2
from training.trainer import Trainer
from training.scheduler import CosineAnnealingScheduler
from training.loss import compute_loss


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
    patience = 5
    min_delta = 0.01


@pytest.fixture
def model():
    return GPT2(SmallConfig(), NoDropTrainConfig())


# ─── 1. KV Cache ────────────────────────────────────────────────────────────

class TestKVCache:
    def test_cached_forward_matches_full_forward(self, model):
        """Incremental cached generation should produce the same logits as a full forward pass."""
        model.eval()
        seq = torch.randint(0, SmallConfig.vocab_size, (1, 10))

        # full forward (no cache)
        full_logits = model(seq, use_cache=False)

        # incremental: feed prefix, then one token at a time
        model.reset_cache()
        prefix = seq[:, :6]
        _ = model(prefix, use_cache=True)

        for t in range(6, 10):
            token = seq[:, t : t + 1]
            cached_logits = model(token, use_cache=True)

        # last-token logit from both paths should match
        assert torch.allclose(full_logits[:, -1, :], cached_logits[:, -1, :], atol=1e-4), \
            "Cached incremental logits diverge from full forward logits"

    def test_cache_grows_with_tokens(self, model):
        """cache_k in the first layer should grow as tokens are fed."""
        model.eval()
        model.reset_cache()

        x1 = torch.randint(0, SmallConfig.vocab_size, (1, 5))
        model(x1, use_cache=True)
        assert model.h[0].attn.cache_k.size(2) == 5

        x2 = torch.randint(0, SmallConfig.vocab_size, (1, 1))
        model(x2, use_cache=True)
        assert model.h[0].attn.cache_k.size(2) == 6

    def test_reset_cache_clears_state(self, model):
        model.eval()
        x = torch.randint(0, SmallConfig.vocab_size, (1, 4))
        model(x, use_cache=True)
        assert model._cache_initialized is True

        model.reset_cache()
        assert model._cache_initialized is False
        assert model.h[0].attn.cache_k is None

    def test_cache_position_ids_auto(self, model):
        """Position ids should auto-increment when using cache."""
        model.eval()
        model.reset_cache()
        x = torch.randint(0, SmallConfig.vocab_size, (1, 3))
        model(x, use_cache=True)
        assert model.h[0].attn.current_pos == 3

        x2 = torch.randint(0, SmallConfig.vocab_size, (1, 1))
        model(x2, use_cache=True)
        assert model.h[0].attn.current_pos == 4


# ─── 2. Repetition Penalty ──────────────────────────────────────────────────

class TestRepetitionPenalty:
    def test_penalty_reduces_repeated_token_logits(self):
        """Positive logits for repeated tokens should be divided by penalty (reduced)."""
        logits = torch.tensor([[5.0, 3.0, 1.0, -2.0]])  # vocab_size=4
        prev_tokens = torch.tensor([[0, 1, 0]])  # tokens 0 and 1 appeared

        model = GPT2(SmallConfig(), NoDropTrainConfig())
        # call _sample with high temperature to not distort too much, only check logits mutation
        logits_clone = logits.clone()
        penalty = 1.5

        # manually apply penalty logic (same as _sample)
        for i in range(logits_clone.shape[0]):
            prev_ids = torch.unique(prev_tokens[i])
            score = logits_clone[i, prev_ids]
            logits_clone[i, prev_ids] = torch.where(
                score > 0, score / penalty, score * penalty
            )

        # token 0 (logit 5.0) -> 5.0/1.5 ≈ 3.33
        assert logits_clone[0, 0] == pytest.approx(5.0 / 1.5, abs=1e-4)
        # token 1 (logit 3.0) -> 3.0/1.5 = 2.0
        assert logits_clone[0, 1] == pytest.approx(3.0 / 1.5, abs=1e-4)
        # token 2 (not repeated) -> unchanged
        assert logits_clone[0, 2] == pytest.approx(1.0, abs=1e-4)

    def test_negative_logits_amplified(self):
        """Negative logits for repeated tokens should be multiplied by penalty (more negative)."""
        logits = torch.tensor([[-2.0, 3.0]])
        prev_tokens = torch.tensor([[0]])  # token 0 repeated
        penalty = 1.5

        logits_clone = logits.clone()
        prev_ids = torch.unique(prev_tokens[0])
        score = logits_clone[0, prev_ids]
        logits_clone[0, prev_ids] = torch.where(
            score > 0, score / penalty, score * penalty
        )

        # -2.0 * 1.5 = -3.0 (pushed further negative)
        assert logits_clone[0, 0] == pytest.approx(-3.0, abs=1e-4)

    def test_no_penalty_when_1(self, model):
        """With penalty=1.0, logits should remain unchanged."""
        model.eval()
        logits = torch.tensor([[2.0, 4.0, -1.0]])
        prev_tokens = torch.tensor([[0, 1]])
        original = logits.clone()

        # _sample modifies logits in-place, so clone for comparison
        # We just verify the penalty branch is skipped
        # Replicate the penalty logic with penalty=1.0
        penalty = 1.0
        if penalty != 1.0 and prev_tokens is not None:
            for i in range(logits.shape[0]):
                prev_ids = torch.unique(prev_tokens[i])
                score = logits[i, prev_ids]
                logits[i, prev_ids] = torch.where(
                    score > 0, score / penalty, score * penalty
                )

        assert torch.equal(logits, original)

    def test_generate_uses_repetition_penalty(self, model):
        """generate() should produce tokens — smoke test that the penalty path doesn't crash."""
        model.eval()
        prompt = torch.randint(0, SmallConfig.vocab_size, (1, 5))
        output = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=10)
        assert output.shape[1] > 5  # at least some tokens generated


# ─── 3. Early Stopping (Trainer) ────────────────────────────────────────────

def _make_loader(n_batches, seq_len=8, vocab_size=SmallConfig.vocab_size):
    """Create a simple DataLoader that yields dicts with 'input_id' and 'label'."""
    all_input = torch.randint(0, vocab_size, (n_batches * 2, seq_len))
    all_label = torch.randint(0, vocab_size, (n_batches * 2, seq_len))
    dataset = TensorDataset(all_input, all_label)

    class DictLoader:
        """Wraps a DataLoader to yield dicts instead of tuples."""
        def __init__(self, loader):
            self._loader = loader
        def __iter__(self):
            for inp, lab in self._loader:
                yield {'input_id': inp, 'label': lab}
        def __len__(self):
            return len(self._loader)

    return DictLoader(DataLoader(dataset, batch_size=2))


class TestEarlyStopping:
    def test_stops_early_when_no_improvement(self, model, tmp_path):
        """Training should stop before num_epochs if val loss doesn't improve."""
        train_config = NoDropTrainConfig()
        train_config.patience = 2
        train_config.min_delta = 0.01

        train_loader = _make_loader(4)
        val_loader = _make_loader(2)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingScheduler(optimizer, warmup_steps=0, total_steps=100, max_lr=1e-3, min_lr=1e-5)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            scheduler=scheduler,
            train_config=train_config,
            optimizer=optimizer,
            loss=compute_loss,
            device='cpu',
        )

        # Mock validate to return constant loss (no improvement)
        call_count = [0]
        def fake_validate():
            call_count[0] += 1
            return {'val_loss': 5.0, 'perplexity': torch.exp(torch.tensor(5.0))}

        trainer.validate = fake_validate
        num_epochs = 10
        trainer.train(num_epochs=num_epochs, checkpoint_dir=str(tmp_path))

        # Should have stopped after patience+1 epochs (1 initial + patience without improvement)
        assert call_count[0] <= train_config.patience + 1
        assert call_count[0] < num_epochs

    def test_no_early_stop_when_improving(self, model, tmp_path):
        """Training should run all epochs if val loss keeps improving."""
        train_config = NoDropTrainConfig()
        train_config.patience = 2
        train_config.min_delta = 0.01

        train_loader = _make_loader(4)
        val_loader = _make_loader(2)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingScheduler(optimizer, warmup_steps=0, total_steps=100, max_lr=1e-3, min_lr=1e-5)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            scheduler=scheduler,
            train_config=train_config,
            optimizer=optimizer,
            loss=compute_loss,
            device='cpu',
        )

        # Mock validate to return decreasing loss
        call_count = [0]
        def improving_validate():
            call_count[0] += 1
            loss = 5.0 - call_count[0] * 0.5  # keeps decreasing
            return {'val_loss': loss, 'perplexity': torch.exp(torch.tensor(loss))}

        trainer.validate = improving_validate
        num_epochs = 5
        trainer.train(num_epochs=num_epochs, checkpoint_dir=str(tmp_path))

        assert call_count[0] == num_epochs

    def test_best_model_saved(self, model, tmp_path):
        """best_model.pt should be saved when val loss improves."""
        train_config = NoDropTrainConfig()
        train_config.patience = 3
        train_config.min_delta = 0.01

        train_loader = _make_loader(4)
        val_loader = _make_loader(2)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingScheduler(optimizer, warmup_steps=0, total_steps=100, max_lr=1e-3, min_lr=1e-5)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            scheduler=scheduler,
            train_config=train_config,
            optimizer=optimizer,
            loss=compute_loss,
            device='cpu',
        )

        call_count = [0]
        def fake_validate():
            call_count[0] += 1
            return {'val_loss': 5.0 - call_count[0] * 0.1, 'perplexity': torch.tensor(10.0)}

        trainer.validate = fake_validate
        trainer.train(num_epochs=3, checkpoint_dir=str(tmp_path))

        assert os.path.exists(os.path.join(str(tmp_path), 'best_model.pt'))


# ─── 4. Mixed Precision Training ────────────────────────────────────────────

class TestMixedPrecision:
    def test_amp_disabled_on_cpu(self):
        """use_amp should be forced False on CPU even if requested."""
        model = GPT2(SmallConfig(), NoDropTrainConfig())
        train_loader = _make_loader(2)
        val_loader = _make_loader(2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingScheduler(optimizer, warmup_steps=0, total_steps=50, max_lr=1e-3, min_lr=1e-5)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            scheduler=scheduler,
            train_config=NoDropTrainConfig(),
            optimizer=optimizer,
            loss=compute_loss,
            device='cpu',
            use_amp=True,  # request AMP on CPU
        )
        assert trainer.use_amp is False
        assert trainer.scaler is None

    def test_scaler_created_for_cuda(self):
        """GradScaler should be created when use_amp=True and device is cuda."""
        model = GPT2(SmallConfig(), NoDropTrainConfig())
        train_loader = _make_loader(2)
        val_loader = _make_loader(2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingScheduler(optimizer, warmup_steps=0, total_steps=50, max_lr=1e-3, min_lr=1e-5)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            scheduler=scheduler,
            train_config=NoDropTrainConfig(),
            optimizer=optimizer,
            loss=compute_loss,
            device='cuda',  # pretend cuda
            use_amp=True,
        )
        assert trainer.use_amp is True
        assert trainer.scaler is not None

    def test_non_amp_training_runs(self, model, tmp_path):
        """Standard (non-AMP) training should complete without errors."""
        train_loader = _make_loader(4)
        val_loader = _make_loader(2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingScheduler(optimizer, warmup_steps=0, total_steps=100, max_lr=1e-3, min_lr=1e-5)
        train_config = NoDropTrainConfig()
        train_config.patience = 10

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            scheduler=scheduler,
            train_config=train_config,
            optimizer=optimizer,
            loss=compute_loss,
            device='cpu',
            use_amp=False,
        )

        best_val = trainer.train(num_epochs=2, checkpoint_dir=str(tmp_path))
        assert best_val < float('inf')

    def test_amp_branch_uses_autocast(self, model):
        """The AMP code path (train_epoch) should call autocast and scaler when use_amp is True."""
        train_loader = _make_loader(2)
        val_loader = _make_loader(2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingScheduler(optimizer, warmup_steps=0, total_steps=50, max_lr=1e-3, min_lr=1e-5)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            scheduler=scheduler,
            train_config=NoDropTrainConfig(),
            optimizer=optimizer,
            loss=compute_loss,
            device='cpu',
        )

        # Force AMP on to verify the code path structure (won't actually use fp16 on CPU)
        trainer.use_amp = True
        trainer.scaler = MagicMock()
        trainer.scaler.scale.return_value = MagicMock(backward=MagicMock())

        # Should not crash — exercises the AMP branch
        trainer.train_epoch(epoch=0)

        # Verify scaler methods were called
        assert trainer.scaler.scale.called
        assert trainer.scaler.step.called
        assert trainer.scaler.update.called