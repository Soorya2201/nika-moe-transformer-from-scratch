"""MoE correctness: dispatch actually matches each token's assigned expert (not
some average over both), the load-balancing loss matches its closed-form formula,
and a batch that happens to skip one expert doesn't break gradients."""

import torch
import torch.nn.functional as F

from nika_ai.model import MoELayer


def test_moe_output_and_routing_shapes(tiny_config):
    layer = MoELayer(tiny_config)
    x = torch.randn(2, 5, tiny_config.hidden_size)

    out, aux_loss, routing = layer(x, collect_routing=True)

    assert out.shape == x.shape
    assert routing.shape == (2, 5)
    assert aux_loss.dim() == 0


def test_dispatch_matches_manual_per_token_expert_choice(tiny_config):
    """Each token's output must equal running only its assigned expert, scaled by
    that expert's router probability -- the actual sparse MoE mechanism, not a
    dense average over both experts."""
    layer = MoELayer(tiny_config)
    layer.eval()
    x = torch.randn(3, 4, tiny_config.hidden_size)

    out, _, routing = layer(x, collect_routing=True)

    x_flat = x.reshape(-1, tiny_config.hidden_size)
    out_flat = out.reshape(-1, tiny_config.hidden_size)
    routing_flat = routing.reshape(-1)
    router_probs = F.softmax(layer.router(x_flat), dim=-1)

    for i in range(x_flat.shape[0]):
        e = routing_flat[i].item()
        expected = layer.experts[e](x_flat[i : i + 1]) * router_probs[i, e]
        assert torch.allclose(out_flat[i], expected.squeeze(0), atol=1e-6)


def test_load_balancing_loss_matches_hand_computation(tiny_config):
    layer = MoELayer(tiny_config)

    # Perfectly balanced: one token to each expert, 0.5/0.5 confidence.
    router_probs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    expert_idx = torch.tensor([0, 1])
    loss = layer._load_balancing_loss(router_probs, expert_idx)
    # f=[0.5,0.5], P=[0.5,0.5] -> n_experts * sum(f*P) = 2*(0.25+0.25) = 1.0 (the minimum)
    assert torch.allclose(loss, torch.tensor(1.0))

    # Fully collapsed onto expert 0.
    router_probs_collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    expert_idx_collapsed = torch.tensor([0, 0])
    loss_collapsed = layer._load_balancing_loss(router_probs_collapsed, expert_idx_collapsed)
    # f=[1,0], P=[1,0] -> 2*(1*1 + 0*0) = 2.0 (the maximum penalty)
    assert torch.allclose(loss_collapsed, torch.tensor(2.0))
    assert loss_collapsed > loss


def test_batch_using_only_one_expert_does_not_break_gradients(tiny_config):
    """When every token in a batch happens to route to one expert, the other
    expert's forward is skipped entirely (mask.any() is False) as a compute
    saving -- confirm this doesn't break backprop through the router or the
    expert that *was* used."""
    layer = MoELayer(tiny_config)
    x = torch.randn(1, 1, tiny_config.hidden_size, requires_grad=True)

    out, aux_loss, _ = layer(x)
    (out.sum() + aux_loss).backward()

    assert layer.router.gate.weight.grad is not None
    assert not torch.isnan(layer.router.gate.weight.grad).any()
