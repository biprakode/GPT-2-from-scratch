import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from model.ModelConfig import ModelConfig
from model.TrainingConfig import TrainingConfig
from model.gpt2 import GPT2


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


@pytest.fixture
def model():
    return GPT2(SmallConfig(), NoDropTrainConfig())


@pytest.fixture
def sample_input():
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, SmallConfig.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    return input_ids, position_ids


class TestForward:
    def test_output_shape(self, model, sample_input):
        input_ids, position_ids = sample_input
        logits = model(input_ids, position_ids)
        batch_size, seq_len = input_ids.shape
        assert logits.shape == (batch_size, seq_len, SmallConfig.vocab_size)

    def test_output_dtype(self, model, sample_input):
        input_ids, position_ids = sample_input
        logits = model(input_ids, position_ids)
        assert logits.dtype == torch.float32

    def test_output_finite(self, model, sample_input):
        input_ids, position_ids = sample_input
        logits = model(input_ids, position_ids)
        assert torch.isfinite(logits).all()

    def test_position_ids_none(self, model):
        input_ids = torch.randint(0, SmallConfig.vocab_size, (1, 5))
        logits = model(input_ids, position_ids=None)
        assert logits.shape == (1, 5, SmallConfig.vocab_size)

    def test_different_seq_lengths(self, model):
        for seq_len in [1, 4, 16, SmallConfig.block_size]:
            input_ids = torch.randint(0, SmallConfig.vocab_size, (1, seq_len))
            logits = model(input_ids, position_ids=None)
            assert logits.shape == (1, seq_len, SmallConfig.vocab_size)

    def test_batch_independence(self, model):
        """Each sample in a batch should produce the same output as when run alone."""
        model.eval()
        x1 = torch.randint(0, SmallConfig.vocab_size, (1, 6))
        x2 = torch.randint(0, SmallConfig.vocab_size, (1, 6))
        batched = torch.cat([x1, x2], dim=0)

        logits_single_1 = model(x1, position_ids=None)
        logits_single_2 = model(x2, position_ids=None)
        logits_batched = model(batched, position_ids=None)

        assert torch.allclose(logits_batched[0], logits_single_1[0], atol=1e-5)
        assert torch.allclose(logits_batched[1], logits_single_2[0], atol=1e-5)

    def test_deterministic_in_eval(self, model):
        model.eval()
        input_ids = torch.randint(0, SmallConfig.vocab_size, (1, 5))
        out1 = model(input_ids, position_ids=None)
        out2 = model(input_ids, position_ids=None)
        assert torch.equal(out1, out2)

    def test_weight_tying(self, model):
        """wte and lm_head should share the same weight tensor."""
        assert model.wte.weight is model.lm_head.weight


class TestBackward:
    def test_loss_backward(self, model, sample_input):
        input_ids, position_ids = sample_input
        logits = model(input_ids, position_ids)
        targets = torch.randint(0, SmallConfig.vocab_size, input_ids.shape)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, SmallConfig.vocab_size), targets.view(-1)
        )
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_gradients_nonzero(self, model, sample_input):
        input_ids, position_ids = sample_input
        logits = model(input_ids, position_ids)
        targets = torch.randint(0, SmallConfig.vocab_size, input_ids.shape)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, SmallConfig.vocab_size), targets.view(-1)
        )
        loss.backward()

        has_nonzero = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_nonzero = True
                break
        assert has_nonzero, "All gradients are zero"

    def test_optimizer_step_changes_params(self, model, sample_input):
        input_ids, position_ids = sample_input
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # capture params before
        params_before = {
            name: p.clone().detach()
            for name, p in model.named_parameters() if p.requires_grad
        }

        logits = model(input_ids, position_ids)
        targets = torch.randint(0, SmallConfig.vocab_size, input_ids.shape)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, SmallConfig.vocab_size), targets.view(-1)
        )
        loss.backward()
        optimizer.step()

        changed = False
        for name, p in model.named_parameters():
            if p.requires_grad and not torch.equal(p.data, params_before[name]):
                changed = True
                break
        assert changed, "No parameters changed after optimizer step"

    def test_loss_decreases_over_steps(self, model, sample_input):
        input_ids, position_ids = sample_input
        targets = torch.randint(0, SmallConfig.vocab_size, input_ids.shape)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(input_ids, position_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, SmallConfig.vocab_size), targets.view(-1)
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_grad_accumulation(self, model):
        """Two micro-batches accumulated should equal one big batch gradient."""
        model.zero_grad()
        x1 = torch.randint(0, SmallConfig.vocab_size, (1, 4))
        t1 = torch.randint(0, SmallConfig.vocab_size, (1, 4))
        x2 = torch.randint(0, SmallConfig.vocab_size, (1, 4))
        t2 = torch.randint(0, SmallConfig.vocab_size, (1, 4))

        # accumulated gradients
        model.eval()  # disable dropout
        loss1 = torch.nn.functional.cross_entropy(
            model(x1, None).view(-1, SmallConfig.vocab_size), t1.view(-1)
        )
        loss2 = torch.nn.functional.cross_entropy(
            model(x2, None).view(-1, SmallConfig.vocab_size), t2.view(-1)
        )
        combined_loss = (loss1 + loss2) / 2
        combined_loss.backward()
        accum_grads = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }

        # single batch
        model.zero_grad()
        x_cat = torch.cat([x1, x2], dim=0)
        t_cat = torch.cat([t1, t2], dim=0)
        loss_full = torch.nn.functional.cross_entropy(
            model(x_cat, None).view(-1, SmallConfig.vocab_size), t_cat.view(-1)
        )
        loss_full.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, accum_grads[name], atol=1e-5), (
                    f"Gradient mismatch for {name}"
                )