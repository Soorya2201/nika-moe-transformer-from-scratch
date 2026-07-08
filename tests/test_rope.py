"""RoPE correctness: shape, the rotate-half identity, norm preservation, and the
one property that makes RoPE work at all -- q.k depends only on relative position."""

import torch

from nika_ai.model import apply_rotary_pos_emb, precompute_rope_cache, rotate_half


def test_precompute_rope_cache_shapes():
    cos, sin = precompute_rope_cache(head_dim=16, max_seq_len=32)
    assert cos.shape == (32, 16)
    assert sin.shape == (32, 16)


def test_rotate_half_twice_negates():
    # rotate_half([x1, x2]) = [-x2, x1]; rotate_half([-x2, x1]) = [-x1, -x2] = -[x1, x2]
    x = torch.randn(2, 3, 5, 8)
    assert torch.allclose(rotate_half(rotate_half(x)), -x)


def test_apply_rotary_pos_emb_preserves_norm():
    head_dim = 8
    cos, sin = precompute_rope_cache(head_dim, max_seq_len=16)
    q = torch.randn(1, 1, 16, head_dim)
    k = torch.randn(1, 1, 16, head_dim)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    # Rotation is norm-preserving per position -- RoPE only ever rotates, never scales.
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)


def _rotate_single(vec: torch.Tensor, cos_slice: torch.Tensor, sin_slice: torch.Tensor) -> torch.Tensor:
    rotated, _ = apply_rotary_pos_emb(vec, vec, cos_slice, sin_slice)
    return rotated


def test_dot_product_depends_only_on_relative_position():
    """The entire point of RoPE: q(i) . k(j) depends only on i - j, not on i, j
    individually. This is what lets a model trained on short sequences still
    make sense of relative offsets it never saw in absolute-position terms."""
    head_dim = 8
    cos, sin = precompute_rope_cache(head_dim, max_seq_len=200)

    torch.manual_seed(0)
    q_vec = torch.randn(1, 1, 1, head_dim)
    k_vec = torch.randn(1, 1, 1, head_dim)

    def dot_at(pos_q: int, pos_k: int) -> float:
        q_rot = _rotate_single(q_vec, cos[pos_q : pos_q + 1], sin[pos_q : pos_q + 1])
        k_rot = _rotate_single(k_vec, cos[pos_k : pos_k + 1], sin[pos_k : pos_k + 1])
        return (q_rot * k_rot).sum().item()

    same_offset_near = dot_at(5, 3)      # relative offset 2
    same_offset_far = dot_at(105, 103)   # relative offset 2, far-away absolute positions
    different_offset = dot_at(5, 4)      # relative offset 1

    assert abs(same_offset_near - same_offset_far) < 1e-4
    assert abs(same_offset_near - different_offset) > 1e-4
