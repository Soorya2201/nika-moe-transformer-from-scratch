"""
nika_ai.inference — autoregressive generation + MoE routing analysis.

Generation: one new token at a time, each new token fed back in as the next input
(the model's own sampled output, not ground truth). Uses a KV cache so each step
only computes K/V for the one new token instead of recomputing the whole
sequence's K/V from scratch. No gradients anywhere in this file.

Routing analysis: runs the model with collect_routing=True and reports which
expert handled which tokens -- the tool used to check whether the two experts
actually specialized by subject-matter domain (see nika_ai.data.DOMAINS).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from nika_ai.model import NikaMoEConfig, NikaMoETransformer
from nika_ai.tokenizer import FastBPETokenizer


class KVCache:
    """Preallocated per-layer key/value buffers for incremental decoding.

    Preallocating to max_seq_len avoids repeated torch.cat-and-reallocate on every
    decode step.
    """

    def __init__(self, config: NikaMoEConfig, batch_size: int, device, dtype=torch.float32):
        head_dim = config.hidden_size // config.n_heads
        shape = (batch_size, config.n_heads, config.max_seq_len, head_dim)
        self.k_cache = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layers)]
        self.v_cache = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layers)]
        self.cur_len = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        new_len = k.shape[2]
        start, end = self.cur_len, self.cur_len + new_len
        self.k_cache[layer_idx][:, :, start:end, :] = k
        self.v_cache[layer_idx][:, :, start:end, :] = v
        return self.k_cache[layer_idx][:, :, :end, :], self.v_cache[layer_idx][:, :, :end, :]

    def advance(self, n: int) -> None:
        self.cur_len += n


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Sample one token id per batch row from the last-position logits.

    temperature==0 means greedy/deterministic decoding (argmax); top_k restricts
    to the k highest-probability tokens; top_p (nucleus) keeps the smallest token
    set whose cumulative probability covers top_p.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate(
    model: NikaMoETransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Prefill the prompt densely, then decode one token at a time using the KV cache.

    Runs entirely under inference_mode: no autograd graph, no .backward() anywhere
    in this file -- the structural opposite of training's dense, gradient-tracked,
    teacher-forced forward pass.
    """
    model.eval()
    device = input_ids.device
    batch_size = input_ids.shape[0]

    cache = KVCache(model.config, batch_size, device, dtype=next(model.parameters()).dtype)

    logits, _aux_loss, _routing = model(input_ids, kv_cache=cache, start_pos=0, is_causal=True)
    cache.advance(input_ids.shape[1])

    generated = [input_ids]
    next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
    generated.append(next_token)

    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        logits, _aux_loss, _routing = model(next_token, kv_cache=cache, start_pos=cache.cur_len, is_causal=False)
        cache.advance(1)

        next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
        generated.append(next_token)

    return torch.cat(generated, dim=1)


@torch.no_grad()
def analyze_routing(model: NikaMoETransformer, input_ids: torch.Tensor, device) -> list[torch.Tensor]:
    """Return, per layer, which expert each token in input_ids was routed to.

    Shape: a list of length n_layers, each entry a (batch, seq_len) tensor of
    expert indices. Uses the same dense forward pass as training (no KV cache
    needed -- routing doesn't depend on generation).
    """
    model.eval()
    _logits, _aux_loss, routing = model(input_ids.to(device), kv_cache=None, start_pos=0, is_causal=True, collect_routing=True)
    return routing


def expert_usage_fraction(routing: list[torch.Tensor], n_experts: int) -> list[list[float]]:
    """Per layer, the fraction of tokens routed to each expert -- the basic
    specialization signal: compare this across domains to see whether the router
    learned to send different subject matter to different experts."""
    usage = []
    for layer_routing in routing:
        flat = layer_routing.reshape(-1)
        usage.append([(flat == e).float().mean().item() for e in range(n_experts)])
    return usage


def load_model_for_inference(checkpoint_path: str, device: str) -> tuple[NikaMoETransformer, FastBPETokenizer]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_config = NikaMoEConfig(**ckpt["model_config"])
    model = NikaMoETransformer(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # train.py saves with FastBPETokenizer (see nika_ai.tokenizer) -- must load
    # with the same class, its on-disk format differs from the from-scratch
    # BPETokenizer's.
    tokenizer_path = os.path.join(os.path.dirname(checkpoint_path), "tokenizer.json")
    tokenizer = FastBPETokenizer()
    tokenizer.load(tokenizer_path)

    return model, tokenizer


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m nika_ai.inference <checkpoint_path> [prompt text]")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_model_for_inference(sys.argv[1], device)

    prompt_text = sys.argv[2] if len(sys.argv) > 2 else "def "
    prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], device=device)

    output_ids = generate(model, prompt_ids, max_new_tokens=40, temperature=0.8, top_k=40)
    print(tokenizer.decode(output_ids[0].tolist()))
