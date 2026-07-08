"""
nika_ai.model — architecture only (no training loop, no generation loop).

    tokens -> embed -> [ (RMSNorm -> Attention -> +residual) -> (RMSNorm -> MoE(2 experts) -> +residual) ] x N -> RMSNorm -> unembed -> logits

Pre-norm decoder with RMSNorm + RoPE, standard (non-grouped) multi-head causal
self-attention, and a 2-expert top-1-routed Mixture of Experts MLP in place of a
dense MLP: each token is sent to exactly one of two SwiGLU experts, intended to
specialize on different subject matter when trained on mixed-domain data.
train.py and inference.py both import this file so the architecture never drifts
between them.

Toy spec (see README for the full parameter accounting):
    vocab_size (trained)  ~1-4K   d_model = 128     n_layers = 4
    n_heads    = 4 (head_dim=32)  n_experts = 2      top_k = 1 (hardcoded, not configurable)
    expert_intermediate_size = 2688  (-> ~1M params per SwiGLU expert)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NikaMoEConfig:
    vocab_size: int
    hidden_size: int                 # d_model
    n_layers: int
    n_heads: int                    # standard multi-head attention: every head has its own K/V
    n_experts: int                   # number of MoE experts (2, one per subject domain)
    expert_intermediate_size: int     # SwiGLU hidden dim, per expert (d_ff)
    max_seq_len: int = 128
    dropout_prob: float = 0.1        # attention-probability and residual dropout; disabled via model.eval()
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    router_aux_loss_coef: float = 0.01   # weight of the load-balancing loss added to the LM loss

    def __post_init__(self):
        assert self.hidden_size % self.n_heads == 0
        assert self.n_experts >= 1


class RMSNorm(nn.Module):
    """Rescales activations by their root-mean-square; skips mean-centering.

    Cheaper than LayerNorm (one reduction instead of two) and empirically just as
    effective for transformers, which is why it's the default in modern open LLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_fp32 * rms).to(x.dtype) * self.weight


def precompute_rope_cache(
    head_dim: int, max_seq_len: int, theta: float = 10000.0, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE, once, up to max_seq_len."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)          # (max_seq_len, head_dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)          # (max_seq_len, head_dim)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[-x2, x1] where x1, x2 are the two halves of the last dim.

    The real-valued equivalent of multiplying by e^{i*theta} in the complex-number
    formulation of RoPE, without needing a complex dtype.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k by the angle for each position.

    cos/sin must already be sliced to the positions this call covers, shape
    (seq_len, head_dim). Training always slices from position 0; inference slices
    starting at the KV cache's current offset.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class Attention(nn.Module):
    """Standard (non-grouped) causal multi-head self-attention with RoPE.

    Every query head has its own K/V head (no GQA) -- at this toy scale (n_heads=4,
    head_dim=32) there's no KV-cache-memory pressure to trade away quality for, so
    plain multi-head attention is the right default, not a simplification of
    something better.
    """

    def __init__(self, config: NikaMoEConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_size // config.n_heads
        self.dropout_prob = config.dropout_prob

        self.q_proj = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache=None,
        layer_idx: int | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # SDPA's own dropout_p only ever applies when we explicitly pass it, so we
        # gate it on self.training ourselves -- SDPA has no notion of train/eval.
        dropout_p = self.dropout_prob if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLUExpert(nn.Module):
    """A gated MLP: SiLU(gate(x)) * up(x), then projected down. One MoE expert."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """Linear gate producing one logit per expert, per token."""

    def __init__(self, hidden_size: int, n_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


class MoELayer(nn.Module):
    """Top-1 routed MoE MLP: each token goes to exactly one expert (Switch-Transformer style).

    Two SwiGLU experts, meant to specialize on different subject matter when
    trained on mixed-domain data (see nika_ai.data). A load-balancing auxiliary
    loss discourages the router from collapsing onto a single expert; it does not
    prevent domain-correlated specialization — if the two domains are roughly
    equally represented in training, specializing by domain *is* the balanced
    solution, so the two objectives don't fight each other.
    """

    def __init__(self, config: NikaMoEConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.router = Router(config.hidden_size, config.n_experts)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(config.hidden_size, config.expert_intermediate_size) for _ in range(config.n_experts)]
        )

    def forward(
        self, x: torch.Tensor, collect_routing: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        b, t, h = x.shape
        x_flat = x.reshape(-1, h)  # (N, H), N = B*T

        router_logits = self.router(x_flat)          # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)
        expert_weight, expert_idx = router_probs.max(dim=-1)  # (N,), (N,)

        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask = expert_idx == e
            if mask.any():
                # Sparse dispatch: only the tokens routed to expert e are run
                # through it, not the whole batch — this is the actual MoE
                # compute-saving mechanism, not just a masked dense computation.
                out[mask] = self.experts[e](x_flat[mask]) * expert_weight[mask].unsqueeze(-1)

        aux_loss = self._load_balancing_loss(router_probs, expert_idx)
        routing = expert_idx.view(b, t) if collect_routing else None
        return out.view(b, t, h), aux_loss, routing

    def forward_traceable(self, x: torch.Tensor) -> torch.Tensor:
        """Export/fx-traceable version of the same routing math -- see nika_ai.tracing.

        forward() above uses `if mask.any():` and boolean-mask indexing (`x_flat[mask]`)
        to run each expert only on the tokens routed to it -- real compute savings,
        but both the Python `if` on a tensor-derived condition and the
        data-dependent-shape mask indexing are exactly what torch.export (and even
        plain torch.fx.symbolic_trace) cannot represent in a static graph: the graph
        has to be the same shape regardless of what the router decides at runtime.
        This version computes every expert on every token (strictly more FLOPs) and
        combines with a one-hot mask instead of a boolean-indexed dispatch, so every
        op and every tensor shape is fixed at trace time. Not used by real
        training/inference (forward() is more efficient) -- this exists purely to
        demonstrate the fix for the class of bug real MoE graph export hits.
        """
        b, t, h = x.shape
        x_flat = x.reshape(-1, h)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        expert_weight, expert_idx = router_probs.max(dim=-1)

        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)  # (N, n_experts, H)
        one_hot = F.one_hot(expert_idx, num_classes=self.n_experts).to(expert_outputs.dtype)  # (N, n_experts)
        combined = (expert_outputs * one_hot.unsqueeze(-1)).sum(dim=1)  # (N, H)

        return (combined * expert_weight.unsqueeze(-1)).view(b, t, h)

    def _load_balancing_loss(self, router_probs: torch.Tensor, expert_idx: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer load-balancing loss: n_experts * sum_i(f_i * P_i).

        f_i = fraction of tokens hard-routed (argmax) to expert i.
        P_i = average soft router probability mass assigned to expert i.
        Minimized (at 1.0) when routing is uniform across experts; penalizes the
        router for collapsing all tokens onto a single expert.
        """
        f = F.one_hot(expert_idx, num_classes=self.n_experts).float().mean(dim=0)
        p = router_probs.mean(dim=0)
        return self.n_experts * (f * p).sum()


class TransformerBlock(nn.Module):
    """One pre-norm block: RMSNorm -> Attention -> +residual, RMSNorm -> MoE -> +residual.

    Residual dropout (nn.Dropout, so it automatically respects model.train()/eval())
    is applied to each sublayer's output before it's added back to the residual
    stream -- the standard GPT-style dropout placement.
    """

    def __init__(self, config: NikaMoEConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = MoELayer(config)
        self.resid_dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache=None,
        layer_idx: int | None = None,
        is_causal: bool = True,
        collect_routing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = x + self.resid_dropout(self.self_attn(self.input_layernorm(x), cos, sin, kv_cache, layer_idx, is_causal))
        moe_out, aux_loss, routing = self.moe(self.post_attention_layernorm(x), collect_routing=collect_routing)
        x = x + self.resid_dropout(moe_out)
        return x, aux_loss, routing


class NikaMoETransformer(nn.Module):
    """Full model: embed -> N x TransformerBlock (MoE) -> final RMSNorm -> unembed.

    forward() always returns (logits, aux_loss, routing) so callers have one
    consistent contract: training uses aux_loss, inference and tests typically
    discard it, and either can opt into routing analysis with collect_routing=True.
    """

    def __init__(self, config: NikaMoEConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        head_dim = config.hidden_size // config.n_heads
        cos, sin = precompute_rope_cache(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache=None,
        start_pos: int = 0,
        is_causal: bool = True,
        collect_routing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        b, t = input_ids.shape
        x = self.tok_embeddings(input_ids)

        cos = self.rope_cos[start_pos : start_pos + t]
        sin = self.rope_sin[start_pos : start_pos + t]

        aux_loss_total = x.new_zeros(())
        routing_per_layer = [] if collect_routing else None

        for layer_idx, layer in enumerate(self.layers):
            x, aux_loss, routing = layer(
                x, cos, sin, kv_cache=kv_cache, layer_idx=layer_idx, is_causal=is_causal, collect_routing=collect_routing
            )
            aux_loss_total = aux_loss_total + aux_loss
            if collect_routing:
                routing_per_layer.append(routing)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, aux_loss_total, routing_per_layer

    def get_num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params
