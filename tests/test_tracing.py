"""nika_ai.tracing: confirms the MoE routing's data-dependent control flow
(`if mask.any():` / boolean-mask indexing) really does break both torch.fx and
torch.export, and that forward_traceable really does fix it -- and that the fix
computes the same numbers as the original, just densely instead of sparsely."""

import torch

from nika_ai.model import MoELayer
from nika_ai.tracing import (
    default_tiny_config,
    trace_attention_with_fx,
    try_export_moe,
    try_export_moe_traceable,
    try_fx_trace_moe,
    try_fx_trace_moe_traceable,
)


def test_attention_traces_cleanly_with_fx():
    traced = trace_attention_with_fx(default_tiny_config())
    assert traced is not None


def test_eager_moe_fails_fx_trace():
    ok, message = try_fx_trace_moe(default_tiny_config())
    assert not ok
    assert "control flow" in message.lower()


def test_eager_moe_fails_export():
    ok, message = try_export_moe(default_tiny_config())
    assert not ok
    assert "data-dependent" in message.lower() or "guardondatadependent" in message.lower()


def test_traceable_moe_succeeds_fx_trace():
    ok, message, traced = try_fx_trace_moe_traceable(default_tiny_config())
    assert ok, message
    assert traced is not None


def test_traceable_moe_succeeds_export():
    ok, message, op_names = try_export_moe_traceable(default_tiny_config())
    assert ok, message
    assert op_names is not None
    assert len(op_names) > 0
    # The fix must not silently drop the routing math -- softmax and the
    # per-expert linear projections should still show up in the exported graph.
    assert any("softmax" in name for name in op_names)


def test_traceable_variant_matches_eager_numerically():
    """The whole point: forward_traceable is a different implementation of the
    exact same computation, not an approximation of it."""
    config = default_tiny_config()
    layer = MoELayer(config)
    layer.eval()
    x = torch.randn(2, 5, config.hidden_size)

    out_eager, _, _ = layer(x)
    out_traceable = layer.forward_traceable(x)

    assert torch.allclose(out_eager, out_traceable, atol=1e-6)
