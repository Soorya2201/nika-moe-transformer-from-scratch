"""
nika_ai.distributed — hand-rolled ring all-reduce for gradient synchronization.

Reimplements the bandwidth-optimal ring all-reduce algorithm (the same algorithm
NCCL/Horovod use internally) directly on top of torch.distributed's point-to-point
send/recv primitives, instead of calling the built-in dist.all_reduce. At this
model's ~1M-parameter scale there is no practical need for multi-process training
-- this module exists so the mechanics of data-parallel gradient sync are visible
and testable in this codebase, not hidden inside a library call.

Run with torchrun, e.g.:
    torchrun --nproc_per_node=2 -m nika_ai.train --distributed
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def setup_distributed(backend: str = "gloo") -> tuple[int, int]:
    """Initialize the process group from torchrun's environment variables.

    Returns (rank, world_size). gloo is the CPU-friendly backend -- no CUDA/NCCL
    required, since this model is meant to be trained on a laptop.
    """
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def broadcast_initial_weights(model: torch.nn.Module, src: int = 0) -> None:
    """Copy rank src's weights to every other rank before training starts.

    Data-parallel replicas only stay in sync if they start identical and apply
    identical (all-reduced) gradient updates every step -- without this, each
    rank's independently-random init would make the replicas diverge immediately.
    """
    for param in model.parameters():
        dist.broadcast(param.data, src=src)


def _ring_all_reduce_sum(tensor: torch.Tensor, rank: int, world_size: int) -> None:
    """In-place ring all-reduce (sum) of `tensor` across all ranks.

    Two phases of world_size - 1 steps each, every step doing exactly one send and
    one recv per rank -- this is what makes the ring bandwidth-optimal: the total
    data moved per rank does not grow with world_size.

      1. Scatter-reduce: a running partial sum for chunk i circulates around the
         ring once per step; after world_size - 1 steps, each rank holds one fully
         -summed chunk (a different chunk index per rank).
      2. All-gather: each rank's fully-summed chunk circulates around the ring so
         that after world_size - 1 more steps, every rank holds every chunk, fully
         reduced.
    """
    if world_size == 1:
        return

    # Work on an owned contiguous copy: chunks are views into `flat`, and in-place
    # updates to a chunk therefore mutate `flat` directly, regardless of whether
    # the original `tensor` happened to be contiguous.
    flat = tensor.contiguous().view(-1).clone()
    chunks = list(torch.tensor_split(flat, world_size))

    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1) % world_size

    # --- Phase 1: scatter-reduce ---
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        send_buf = chunks[send_idx].contiguous()
        recv_buf = torch.empty_like(chunks[recv_idx])

        send_req = dist.isend(send_buf, dst=send_rank)
        recv_req = dist.irecv(recv_buf, src=recv_rank)
        send_req.wait()
        recv_req.wait()

        chunks[recv_idx] += recv_buf

    # --- Phase 2: all-gather ---
    for step in range(world_size - 1):
        send_idx = (rank - step + 1) % world_size
        recv_idx = (rank - step) % world_size

        send_buf = chunks[send_idx].contiguous()
        recv_buf = torch.empty_like(chunks[recv_idx])

        send_req = dist.isend(send_buf, dst=send_rank)
        recv_req = dist.irecv(recv_buf, src=recv_rank)
        send_req.wait()
        recv_req.wait()

        chunks[recv_idx].copy_(recv_buf)

    tensor.copy_(flat.view_as(tensor))


def sync_gradients(model: torch.nn.Module, rank: int, world_size: int) -> None:
    """Average every parameter's gradient across all ranks via ring all-reduce.

    Call after loss.backward() and before optimizer.step(). Each rank sees
    different data and computes different gradients; this call turns those into
    the same averaged gradient on every rank, so the (therefore identical)
    optimizer step keeps every replica in sync.
    """
    for param in model.parameters():
        if param.grad is None:
            continue
        _ring_all_reduce_sum(param.grad, rank, world_size)
        param.grad /= world_size
