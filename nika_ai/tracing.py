"""
nika_ai.tracing — torch.fx and torch.export, and where MoE routing breaks them.

This is the "PyTorch at the graph level" exercise: eager PyTorch code and its
graph representation are two different things, and a compiler ingestion pipeline
(the kind Etched's ASIC would consume) needs the latter. Two tools, increasing in
strictness:

  torch.fx.symbolic_trace -- traces eager code into an explicit graph of nodes
  (calls, args, ops). Good for building intuition: print the graph and match each
  node back to a line in forward().

  torch.export.export -- the modern, stricter export path. It requires the *whole*
  forward pass to be a single static graph: no data-dependent Python control flow
  (an `if` whose condition depends on a tensor's runtime value) and no
  data-dependent tensor shapes.

Our MoELayer.forward() has both problems on purpose (see nika_ai.model): it uses
`if mask.any():` and boolean-mask indexing (`x_flat[mask]`) to only run each
expert on the tokens actually routed to it -- real compute savings, and exactly
the kind of thing real MoE routing does, and exactly the kind of thing static
graph export cannot represent. MoELayer.forward_traceable() is the fix: same
math, computed densely and combined with a one-hot mask instead of a conditional
dispatch, so the graph is static regardless of what the router decides at runtime.

Run as a script to see all of this happen:
    python -m nika_ai.tracing
"""

from __future__ import annotations

import torch
import torch.fx

from nika_ai.model import Attention, MoELayer, NikaMoEConfig, precompute_rope_cache


def default_tiny_config() -> NikaMoEConfig:
    return NikaMoEConfig(
        vocab_size=256, hidden_size=32, n_layers=1, n_heads=4, n_experts=2, expert_intermediate_size=48, max_seq_len=16
    )


def trace_attention_with_fx(config: NikaMoEConfig) -> torch.fx.GraphModule:
    """Attention has no data-dependent control flow -- is_causal is a plain Python
    bool, not a tensor -- so this traces cleanly and is the reference case for
    what "the aha moment" (eager code vs. graph) looks like when nothing goes
    wrong.

    concrete_args fixes the non-tensor args (kv_cache, layer_idx, is_causal) to
    literal values instead of letting fx substitute Proxies for them: fx
    substitutes a Proxy for *every* forward() argument by default, including
    ones with defaults, and F.scaled_dot_product_attention's is_causal argument
    requires a concrete bool, not a Proxy -- this is the fix, not a workaround.
    """
    attn = Attention(config)
    return torch.fx.symbolic_trace(attn, concrete_args={"kv_cache": None, "layer_idx": None, "is_causal": True})


def try_fx_trace_moe(config: NikaMoEConfig) -> tuple[bool, str]:
    """Attempt to fx-trace the real (sparse-dispatch) MoELayer.forward. Expected
    to fail: `if mask.any():` branches on a Proxy (symbolic placeholder), and fx
    cannot evaluate a Proxy's truthiness at trace time."""
    layer = MoELayer(config)
    try:
        torch.fx.symbolic_trace(layer)
        return True, "traced successfully (unexpected)"
    except Exception as e:  # noqa: BLE001 -- deliberately broad, we want to show *any* trace failure
        return False, f"{type(e).__name__}: {e}"


def try_export_moe(config: NikaMoEConfig, batch: int = 2, seq: int = 4) -> tuple[bool, str]:
    """Attempt torch.export.export on the real MoELayer. Expected to fail on the
    same data-dependent control flow, via export's own (typically more specific)
    error about guarding on a data-dependent value."""
    layer = MoELayer(config)
    example = (torch.randn(batch, seq, config.hidden_size),)
    try:
        torch.export.export(layer, example)
        return True, "exported successfully (unexpected)"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def try_fx_trace_moe_traceable(config: NikaMoEConfig) -> tuple[bool, str, torch.fx.GraphModule | None]:
    """fx-trace MoELayer.forward_traceable -- the dense/one-hot-combine fix. No
    Python control flow depends on a tensor value, so this should trace cleanly."""
    layer = MoELayer(config)
    try:
        # symbolic_trace traces `forward` by default; to trace forward_traceable
        # instead, wrap it in a tiny module that exposes it as `forward`.
        class _Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner.forward_traceable(x)

        traced = torch.fx.symbolic_trace(_Wrapper(layer))
        return True, "traced successfully", traced
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}", None


def try_export_moe_traceable(
    config: NikaMoEConfig, batch: int = 2, seq: int = 4
) -> tuple[bool, str, set[str] | None]:
    """torch.export.export on the traceable MoE variant. Should succeed; returns
    the set of operator names used in the exported graph (our own miniature
    "operator coverage check" for the MoE layer)."""

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner.forward_traceable(x)

    wrapped = _Wrapper(MoELayer(config))
    example = (torch.randn(batch, seq, config.hidden_size),)
    try:
        exported = torch.export.export(wrapped, example)
        op_names = {
            str(node.target) for node in exported.graph.nodes if node.op in ("call_function", "call_method")
        }
        return True, "exported successfully", op_names
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}", None


if __name__ == "__main__":
    config = default_tiny_config()

    print("=== torch.fx.symbolic_trace(Attention) ===")
    traced_attn = trace_attention_with_fx(config)
    print(traced_attn.graph)
    print()

    print("=== torch.fx.symbolic_trace(MoELayer.forward) -- expected to fail ===")
    ok, msg = try_fx_trace_moe(config)
    print(f"succeeded={ok}: {msg}")
    print()

    print("=== torch.export.export(MoELayer.forward) -- expected to fail ===")
    ok, msg = try_export_moe(config)
    print(f"succeeded={ok}: {msg}")
    print()

    print("=== torch.fx.symbolic_trace(MoELayer.forward_traceable) -- expected to succeed ===")
    ok, msg, traced = try_fx_trace_moe_traceable(config)
    print(f"succeeded={ok}: {msg}")
    if traced is not None:
        print(traced.graph)
    print()

    print("=== torch.export.export(MoELayer.forward_traceable) -- expected to succeed ===")
    ok, msg, op_names = try_export_moe_traceable(config)
    print(f"succeeded={ok}: {msg}")
    if op_names is not None:
        print("operators used in the exported MoE graph:")
        for name in sorted(op_names):
            print(f"  {name}")
