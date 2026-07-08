import pytest

from nika_ai.model import NikaMoEConfig


@pytest.fixture
def tiny_config() -> NikaMoEConfig:
    return NikaMoEConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        n_experts=2,
        expert_intermediate_size=48,
        max_seq_len=64,
    )
