"""Dropout must actually behave differently in train() vs eval() mode -- this is
the one training/inference distinction (see README) that isn't tested anywhere
else, since every other test in this suite deliberately uses dropout_prob=0.0 for
determinism."""

import torch

from nika_ai.model import NikaMoEConfig, NikaMoETransformer


def _dropout_config() -> NikaMoEConfig:
    return NikaMoEConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_experts=2,
        expert_intermediate_size=48,
        max_seq_len=64,
        dropout_prob=0.5,  # high on purpose, so the effect isn't lost in noise
    )


def test_train_mode_is_stochastic():
    model = NikaMoETransformer(_dropout_config())
    model.train()
    x = torch.randint(0, 64, (2, 16))

    torch.manual_seed(0)
    logits_a, _, _ = model(x)
    torch.manual_seed(1)
    logits_b, _, _ = model(x)

    assert not torch.allclose(logits_a, logits_b)


def test_eval_mode_is_deterministic():
    model = NikaMoETransformer(_dropout_config())
    model.eval()
    x = torch.randint(0, 64, (2, 16))

    torch.manual_seed(0)
    logits_a, _, _ = model(x)
    torch.manual_seed(1)
    logits_b, _, _ = model(x)

    assert torch.allclose(logits_a, logits_b)
