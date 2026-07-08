# NIKA_AI_1B — nika_ai

A from-scratch decoder-only transformer, in pure PyTorch:

```
tokens -> embed -> [ (RMSNorm -> Attention -> +residual) -> (RMSNorm -> MoE(2 experts) -> +residual) ] x N -> RMSNorm -> unembed -> logits
```

- **RMSNorm** instead of LayerNorm
- **RoPE** (rotary position embeddings) instead of learned absolute positions
- **GQA** (grouped-query attention): fewer KV heads than query heads
- **MoE MLP**: top-1 routed, 2 SwiGLU experts (Switch-Transformer style), meant to specialize by subject matter
- Target scale: **~1M parameters total** — small enough to train and experiment with on a laptop CPU in seconds, not a competitive language model

## Layout

```
nika_ai/
├── nika_ai/
│   ├── model.py          # NikaMoEConfig, Attention (RoPE+GQA), MoELayer, RMSNorm, NikaMoETransformer
│   ├── tokenizer.py       # from-scratch byte-level BPE (train/encode/decode)
│   ├── data.py            # PackedTextDataset + the two subject-matter corpora
│   ├── train.py           # training loop, entry point for torchrun
│   ├── inference.py        # KV-cached generation, routing analysis
│   └── distributed.py       # hand-rolled ring all-reduce
├── data/
│   ├── prose.txt           # domain "prose": narrative English text
│   └── code.txt            # domain "code": Python source
├── tests/
│   ├── test_rope.py
│   ├── test_moe_routing.py
│   └── test_train_vs_inference_parity.py
├── notebooks/
│   └── analysis.ipynb      # imports from nika_ai, visualizes results only
└── README.md
```

`model.py` is architecture only — `train.py` and `inference.py` both import it, so the two never define the model differently by accident.

## Setup

```bash
pip install -e .
pip install pytest matplotlib   # for tests / the notebook
```

## Quickstart

**Train** (single process, ~1M-param config, trains its own BPE tokenizer on `data/prose.txt` + `data/code.txt`):

```bash
python -m nika_ai.train --max-steps 300 --checkpoint-dir checkpoints
```

**Train, distributed** (hand-rolled ring all-reduce over N local processes, gloo/CPU):

```bash
torchrun --nproc_per_node=2 -m nika_ai.train --distributed --max-steps 300 --checkpoint-dir checkpoints
```

**Generate**:

```bash
python -m nika_ai.inference checkpoints/ckpt_final.pt "def "
```

**Run tests**:

```bash
pytest tests/ -v
```

**Analysis notebook** (loss curves + routing specialization, MoE vs. a dense `n_experts=1` baseline):

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Config and parameter budget

Default config (`nika_ai/train.py`):

```
vocab_size            = 768
hidden_size           = 128
n_layers              = 3
n_heads               = 8
n_kv_heads            = 4   # GQA ratio 2:1
n_experts             = 2
expert_intermediate_size = 256
```

Per layer: attention ≈ 49K params, MoE (router + 2 experts) ≈ 197K params → ≈246K/layer × 3 layers ≈ 739K, + tied embedding (768×128) ≈ 98K ≈ **≈0.84M total**. Verify with:

```python
from nika_ai.model import NikaMoEConfig, NikaMoETransformer
cfg = NikaMoEConfig(vocab_size=768, hidden_size=128, n_layers=3, n_heads=8, n_kv_heads=4, n_experts=2, expert_intermediate_size=256, max_seq_len=256)
NikaMoETransformer(cfg).get_num_params()
```

## The MoE mechanism

Each token's hidden state is routed by a linear gate (`Router`) to exactly one of two `SwiGLUExpert` MLPs (top-1, Switch-Transformer style). A load-balancing auxiliary loss (`MoELayer._load_balancing_loss`, weighted by `router_aux_loss_coef` in the total loss) discourages the router from collapsing every token onto a single expert — it does not fight domain specialization, since if `prose` and `code` are roughly equally represented in training, specializing by domain *is* the balanced solution.

`nika_ai.inference.analyze_routing` / `expert_usage_fraction` report which expert handled which domain's tokens after training — that's the evaluation `notebooks/analysis.ipynb` runs.

## Distributed training

`nika_ai/distributed.py` reimplements ring all-reduce (the same algorithm NCCL/Horovod use) directly on `torch.distributed`'s point-to-point `isend`/`irecv`, rather than calling the built-in `dist.all_reduce`. At this model's scale there's no practical need for multi-process training — this exists so the mechanics of gradient synchronization are visible and testable in this codebase, runnable locally via `torchrun --nproc_per_node=N`.

## Scope notes

- The tokenizer, corpora, and model are all intentionally tiny — this is a from-scratch learning/demonstration project, not a competitive language model.
- `data/prose.txt` and `data/code.txt` are original short texts written for this project (not scraped), so the two domains are small but unambiguous and copyright-free.
