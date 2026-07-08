"""The most important correctness check in this codebase: training's dense,
full-sequence attention and inference's one-token-at-a-time KV-cached decoding
must produce identical logits (and identical MoE routing) for the same tokens.
A RoPE position-offset or cache-indexing bug would silently diverge here without
ever crashing -- this is the only test that catches that class of bug."""

import torch

from nika_ai.inference import KVCache
from nika_ai.model import NikaMoETransformer


def _run_cached(model, seq):
    cache = KVCache(model.config, batch_size=1, device="cpu", dtype=torch.float32)
    logits_steps, routing_steps = [], []

    with torch.no_grad():
        logits, _, routing = model(seq[:, :1], kv_cache=cache, start_pos=0, is_causal=True, collect_routing=True)
        cache.advance(1)
        logits_steps.append(logits)
        routing_steps.append(routing)

        for t in range(1, seq.shape[1]):
            logits, _, routing = model(
                seq[:, t : t + 1], kv_cache=cache, start_pos=cache.cur_len, is_causal=False, collect_routing=True
            )
            cache.advance(1)
            logits_steps.append(logits)
            routing_steps.append(routing)

    return torch.cat(logits_steps, dim=1), routing_steps


def test_kv_cache_logits_match_dense_forward(tiny_config):
    model = NikaMoETransformer(tiny_config)
    model.eval()

    torch.manual_seed(0)
    seq = torch.randint(0, tiny_config.vocab_size, (1, 10))

    with torch.no_grad():
        dense_logits, _, _ = model(seq, kv_cache=None, start_pos=0, is_causal=True)

    cached_logits, _ = _run_cached(model, seq)

    assert torch.allclose(dense_logits, cached_logits, atol=1e-4, rtol=1e-3)


def test_moe_routing_identical_between_dense_and_cached_paths(tiny_config):
    model = NikaMoETransformer(tiny_config)
    model.eval()

    torch.manual_seed(1)
    seq = torch.randint(0, tiny_config.vocab_size, (1, 6))

    with torch.no_grad():
        _, _, dense_routing = model(seq, kv_cache=None, start_pos=0, is_causal=True, collect_routing=True)

    _, cached_routing_per_step = _run_cached(model, seq)

    for layer_idx in range(tiny_config.n_layers):
        cached_layer_routing = torch.cat([step[layer_idx] for step in cached_routing_per_step], dim=1)
        assert torch.equal(dense_routing[layer_idx], cached_layer_routing)
