"""
nika_ai.train — training.

Dense, parallel forward pass over full sequences: teacher forcing (the true next
token is always the target), loss on every position at once, gradients, optimizer
step. No KV cache -- training always has the whole sequence available up front.

The two-expert MoE adds one extra loss term: the router's load-balancing auxiliary
loss (see nika_ai.model.MoELayer), added to the cross-entropy loss with a small
coefficient so it nudges routing without dominating the language-modeling signal.

Single-process:
    python -m nika_ai.train

Distributed (hand-rolled ring all-reduce, see nika_ai.distributed), 2 local processes:
    torchrun --nproc_per_node=2 -m nika_ai.train --distributed
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nika_ai.data import DOMAINS, build_datasets, load_domain_text
from nika_ai.distributed import broadcast_initial_weights, cleanup_distributed, setup_distributed, sync_gradients
from nika_ai.model import NikaMoEConfig, NikaMoETransformer
from nika_ai.tokenizer import BPETokenizer


@dataclass
class TrainConfig:
    batch_size: int = 4
    seq_len: int = 64
    grad_accum_steps: int = 1
    max_steps: int = 300
    warmup_steps: int = 30
    lr_max: float = 3e-3
    lr_min: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95          # not 0.999: more responsive to noisy small-batch gradients
    eps: float = 1e-8
    grad_clip: float = 1.0
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    eval_interval: int = 50
    log_interval: int = 10
    distributed: bool = False


def configure_optimizer(
    model: NikaMoETransformer, weight_decay: float, lr: float, betas: tuple[float, float], eps: float
) -> torch.optim.AdamW:
    """AdamW with decoupled weight decay applied only to weight matrices (dim >= 2).

    RMSNorm weights and router gate biases (there are none -- no biases anywhere
    in this model) are excluded from decay.
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)


def get_lr(step: int, warmup_steps: int, max_steps: int, lr_max: float, lr_min: float) -> float:
    """Linear warmup, then cosine decay to lr_min."""
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    if step >= max_steps:
        return lr_min
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + coeff * (lr_max - lr_min)


def _infinite_batches(loader: DataLoader, sampler: DistributedSampler | None):
    """Cycle through the (tiny) dataset indefinitely, re-shuffling each epoch."""
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


@torch.no_grad()
def evaluate_domain_losses(model: NikaMoETransformer, domain_splits, device: str) -> dict[str, float]:
    """Held-out cross-entropy loss on each domain's val split (no aux loss, no grads)."""
    model.eval()
    losses = {}
    for domain in DOMAINS:
        val_ds = domain_splits[domain].val
        if len(val_ds) == 0:
            losses[domain] = float("nan")
            continue
        xs = torch.stack([val_ds[i][0] for i in range(len(val_ds))]).to(device)
        ys = torch.stack([val_ds[i][1] for i in range(len(val_ds))]).to(device)
        logits, _, _ = model(xs, kv_cache=None, start_pos=0, is_causal=True)
        losses[domain] = F.cross_entropy(logits.view(-1, logits.size(-1)), ys.view(-1)).item()
    model.train()
    return losses


def save_checkpoint(
    checkpoint_dir: str,
    name: str,
    model: NikaMoETransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: NikaMoEConfig,
    train_config: TrainConfig,
) -> None:
    """Save weights + optimizer state + both configs; model_config lets inference.py
    reconstruct the exact architecture from the checkpoint alone."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
        },
        os.path.join(checkpoint_dir, name),
    )


def load_checkpoint(path: str, device: str) -> tuple[NikaMoETransformer, dict]:
    ckpt = torch.load(path, map_location=device)
    model_config = NikaMoEConfig(**ckpt["model_config"])
    model = NikaMoETransformer(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def train(model_config: NikaMoEConfig, cfg: TrainConfig, history: list | None = None):
    """Train, returning (model, tokenizer, domain_splits).

    If `history` is passed a list, an entry is appended at every eval_interval
    step with the current train loss and per-domain val losses -- this is what
    lets notebooks/analysis.ipynb plot loss curves without reimplementing the
    training loop itself.
    """
    rank, world_size = 0, 1
    if cfg.distributed:
        rank, world_size = setup_distributed()

    is_main = rank == 0

    # Tokenizer: train once (fast at this corpus size) and persist alongside the
    # checkpoint so inference.py can load the exact same vocabulary/merges.
    tokenizer = BPETokenizer()
    combined_text = "".join(load_domain_text(d) for d in DOMAINS)
    tokenizer.train(combined_text, vocab_size=model_config.vocab_size)
    if is_main:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        tokenizer.save(os.path.join(cfg.checkpoint_dir, "tokenizer.json"))

    combined_train, domain_splits = build_datasets(tokenizer, cfg.seq_len)

    sampler = None
    if cfg.distributed:
        sampler = DistributedSampler(combined_train, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(combined_train, batch_size=cfg.batch_size, shuffle=(sampler is None), sampler=sampler, drop_last=True)
    batches = _infinite_batches(loader, sampler)

    model = NikaMoETransformer(model_config).to(cfg.device)
    if cfg.distributed:
        broadcast_initial_weights(model, src=0)

    optimizer = configure_optimizer(model, cfg.weight_decay, cfg.lr_max, (cfg.beta1, cfg.beta2), cfg.eps)

    t0 = time.time()
    for step in range(cfg.max_steps):
        lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr_max, cfg.lr_min)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y, _domain_label = next(batches)
            x, y = x.to(cfg.device), y.to(cfg.device)

            logits, aux_loss, _ = model(x, kv_cache=None, start_pos=0, is_causal=True)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = ce_loss + model_config.router_aux_loss_coef * aux_loss

            (loss / cfg.grad_accum_steps).backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if cfg.distributed:
            sync_gradients(model, rank, world_size)
        optimizer.step()

        if is_main and step % cfg.log_interval == 0:
            dt = time.time() - t0
            print(f"step {step:5d} | loss {total_loss / cfg.grad_accum_steps:.4f} | lr {lr:.2e} | {dt:.2f}s", flush=True)
            t0 = time.time()

        if is_main and step > 0 and step % cfg.eval_interval == 0:
            domain_losses = evaluate_domain_losses(model, domain_splits, cfg.device)
            print(f"         val loss by domain: {domain_losses}", flush=True)
            if history is not None:
                history.append({"step": step, "train_loss": total_loss / cfg.grad_accum_steps, **domain_losses})

    if is_main:
        save_checkpoint(cfg.checkpoint_dir, "ckpt_final.pt", model, optimizer, cfg.max_steps, model_config, cfg)
        print(f"saved checkpoint to {os.path.join(cfg.checkpoint_dir, 'ckpt_final.pt')}", flush=True)

    if cfg.distributed:
        cleanup_distributed()

    return model, tokenizer, domain_splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true", help="sync gradients across ranks via ring all-reduce")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    model_config = NikaMoEConfig(
        vocab_size=768,
        hidden_size=128,
        n_layers=3,
        n_heads=8,
        n_kv_heads=4,
        n_experts=2,
        expert_intermediate_size=256,
        max_seq_len=256,
    )
    train_config = TrainConfig(
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        distributed=args.distributed,
    )
    train(model_config, train_config)
