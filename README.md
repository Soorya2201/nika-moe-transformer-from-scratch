# nika_ai — a from-scratch decoder-only transformer with a routed Mixture-of-Experts MLP

This repository is a complete, from-scratch implementation of a modern decoder-only transformer language model, built in pure PyTorch with no dependence on HuggingFace `transformers`, no borrowed model code, and no black boxes left unopened. Every component in the diagram below — the normalization, the positional encoding, the attention mechanism, the mixture-of-experts routing, the training loop, the inference loop, and the distributed-training gradient synchronization — was implemented, unit-tested, and verified end-to-end by hand.

The model itself is intentionally small (~9M parameters, trained on a few kilobytes of text). The scope of the engineering is not: this project also includes a hand-rolled implementation of the ring all-reduce algorithm that underlies real distributed training frameworks, a working comparison against PyTorch's real `DistributedDataParallel`, and a graph-tracing investigation (`torch.fx` / `torch.export`) that deliberately reproduces — and fixes — the exact class of bug that real Mixture-of-Experts models hit when you try to export them to a static computation graph. The goal throughout was depth over scale: understanding *why* each design decision was made, being able to show the failure mode of the naive alternative, and having a test that proves it.

```
tokens -> embed -> [ (RMSNorm -> Attention -> +residual) -> (RMSNorm -> MoE(2 experts) -> +residual) ] x N -> RMSNorm -> unembed -> logits
```

---

## Table of contents

1. [What's actually in here](#whats-actually-in-here)
2. [Architecture, and why each piece was chosen](#architecture-and-why-each-piece-was-chosen)
3. [The parameter budget, computed by hand and verified in code](#the-parameter-budget-computed-by-hand-and-verified-in-code)
4. [The Mixture-of-Experts layer](#the-mixture-of-experts-layer)
5. [Tokenizer: built one from scratch, then used the real tool](#tokenizer-built-one-from-scratch-then-used-the-real-tool)
6. [Training pipeline](#training-pipeline)
7. [Distributed training: hand-rolled ring all-reduce vs. real DDP](#distributed-training-hand-rolled-ring-all-reduce-vs-real-ddp)
8. [Multi-process vs. multi-node — an honest distinction](#multi-process-vs-multi-node--an-honest-distinction)
9. [Graph tracing: torch.fx, torch.export, and where MoE routing breaks them](#graph-tracing-torchfx-torchexport-and-where-moe-routing-breaks-them)
10. [Training vs. inference: what actually changes structurally](#training-vs-inference-what-actually-changes-structurally)
11. [Testing philosophy: 18 tests, each proving something specific](#testing-philosophy-18-tests-each-proving-something-specific)
12. [Results: does the MoE actually specialize?](#results-does-the-moe-actually-specialize)
13. [Repository layout](#repository-layout)
14. [Setup and quickstart](#setup-and-quickstart)
15. [Honest scope notes](#honest-scope-notes)

---

## What's actually in here

- A **pre-norm decoder-only transformer** (RMSNorm, RoPE, standard multi-head causal self-attention, dropout) — the same architectural family as LLaMA/Mistral-era models, scaled down.
- A **top-1 routed Mixture-of-Experts MLP** in place of the usual dense feed-forward block, with two SwiGLU experts, a learned router, and a Switch-Transformer-style load-balancing loss.
- **Two tokenizers**: a byte-level BPE implementation written entirely from scratch (to prove the algorithm is understood), and a second implementation backed by HuggingFace's Rust `tokenizers` library (because that's what real tokenizer training actually uses, and the two are worth having both built and used).
- **Three ways to train**: single-process, hand-rolled distributed (ring all-reduce implemented directly on `torch.distributed` point-to-point primitives), and real `torch.nn.parallel.DistributedDataParallel` — all runnable side by side for direct comparison.
- A **graph-tracing investigation** (`torch.fx.symbolic_trace` and `torch.export.export`) that deliberately triggers and then fixes the exact failure mode that data-dependent MoE routing causes under static graph export.
- **KV-cached autoregressive inference**, verified numerically identical to the dense training-time forward pass, position by position.
- **18 automated tests** covering RoPE's mathematical correctness property, MoE dispatch correctness, the load-balancing loss's closed-form value, dropout's train/eval behavior, and the dense-vs-KV-cache-vs-export equivalences.
- A **from-scratch two-domain dataset and evaluation harness** (original prose text vs. original Python source code) used to test whether the MoE experts actually specialize by subject matter — not just that the code runs, but that the mechanism does what it claims to do.

## Architecture, and why each piece was chosen

Every block in the diagram is a deliberate choice, not a default:

**RMSNorm instead of LayerNorm.** LayerNorm computes both a mean and a variance and re-centers *and* rescales activations; RMSNorm only rescales by the root-mean-square, skipping mean-centering entirely. This is cheaper (one reduction instead of two), and empirically works just as well for transformers — which is why it's the default in essentially every modern open-weight LLM. Implemented directly from the formula (`nika_ai/model.py: RMSNorm`), not via `nn.LayerNorm` with a flag, and the reduction is explicitly upcast to fp32 for numerical stability regardless of the input dtype.

**RoPE (rotary position embeddings) instead of learned absolute positions.** A learned position-embedding table is a fixed `(max_seq_len, hidden)` lookup the model never sees past its training length, so it generalizes poorly to longer sequences. RoPE instead rotates the query and key vectors by an angle proportional to absolute position, which has the property that the attention dot product `q · k` ends up depending only on the *relative* position between the two — not their absolute positions. This is a real, testable mathematical property (`tests/test_rope.py::test_dot_product_depends_only_on_relative_position` verifies it numerically: the same relative offset at position (5, 3) and at position (105, 103) produces the same dot product), and it's what gives RoPE-based models much better length extrapolation than learned absolute embeddings, at zero extra parameters.

**Standard multi-head attention (no grouped-query attention).** An earlier version of this project used grouped-query attention (fewer key/value heads than query heads, trading a little quality for a much smaller KV cache) because that's the right choice at production LLM scale. At this toy scale (4 heads, head_dim 32), there is no KV-cache memory pressure to trade away quality for, so the architecture was deliberately simplified back to plain multi-head attention — every head has its own K/V — rather than carrying a production optimization that has no purpose here. Attention itself is computed via `F.scaled_dot_product_attention` with `is_causal=True`, so the causal mask is fused into the kernel rather than materialized as an explicit `(T, T)` tensor.

**Dropout, actually wired to matter.** Attention-probability dropout is passed directly into `scaled_dot_product_attention`'s own `dropout_p` argument (gated on `self.training`, since SDPA itself has no notion of train/eval), and residual dropout is applied to each sublayer's output before it's added back into the residual stream — the standard GPT-style placement. `tests/test_dropout.py` proves this isn't just wired but *active*: the same input produces different logits across calls in `.train()` mode and identical logits in `.eval()` mode.

**Pre-norm residual wiring.** The original Transformer normalizes *after* the residual addition (post-norm), which requires careful learning-rate warmup and becomes unstable to train at depth. Pre-norm normalizes only the input to each sublayer and adds that sublayer's raw output back onto an un-normalized residual stream — this keeps gradients flowing cleanly through deep stacks, at the cost of needing one additional final RMSNorm before the output head (since the residual stream itself is never normalized along the way).

**Weight-tied unembedding.** The output projection to vocabulary logits reuses the same weight matrix as the input token embedding (literally the same `nn.Parameter`, not a copy), which removes a full `(vocab_size × hidden_size)` parameter block and regularizes the model by forcing input and output token representations into the same space — a well-established, near-free win at small-to-mid model scale.

## The parameter budget, computed by hand and verified in code

```
vocab_size    = 4000   (target; see the tokenizer section — the trainer settles lower on this corpus)
d_model       = 128    (hidden_size)
n_layers      = 4
n_heads       = 4      (head_dim = 32, standard multi-head, no GQA)
n_experts     = 2      (per layer)
top_k         = 1      (hardcoded — the simplest possible routing decision, not a configurable knob)
d_ff (expert) = 2688   (expert_intermediate_size)
max_seq_len   = 128
dropout_prob  = 0.1
```

Sizing the experts to ~1M parameters each was done deliberately, not backed into. Each expert is a SwiGLU MLP with *three* weight matrices (gate, up, and down projections), so the parameter count per expert is `3 × d_model × d_ff`. Solving for ~1,000,000 at `d_model = 128` gives `d_ff ≈ 2604`; this project uses **2688**, giving `3 × 128 × 2688 = 1,032,192` (~1.03M per expert). This is deliberately *not* the arithmetic you'd use for a plain two-matrix ReLU feed-forward network (`d_model × d_ff × 2`, which would suggest `d_ff ≈ 4000` for the same budget) — SwiGLU's extra gating matrix means a smaller hidden dimension hits the same parameter target, and getting this arithmetic right up front (rather than discovering it by trial and error) was treated as part of the exercise.

The honest total is **~9 million parameters, not ~2 million** — worth stating plainly rather than glossing over: MoE experts belong to *each transformer layer independently*, so with `n_layers = 4` and `n_experts = 2`, there are 8 total expert instances in the model, each ~1.03M parameters, for **~8.26M in expert feed-forward weights alone**. Add the attention projections (~66K per layer) and the (weight-tied) embedding table, and the verified total is:

```python
from nika_ai.model import NikaMoEConfig, NikaMoETransformer
cfg = NikaMoEConfig(vocab_size=4000, hidden_size=128, n_layers=4, n_heads=4,
                     n_experts=2, expert_intermediate_size=2688, max_seq_len=128)
NikaMoETransformer(cfg).get_num_params()
# 9,033,856
```

## The Mixture-of-Experts layer

In place of a single dense feed-forward block, each transformer layer routes every token's hidden state through a linear gate (`Router`) to exactly one of two independent SwiGLU expert MLPs — top-1 routing, in the style of the Switch Transformer, chosen deliberately as the simplest possible routing decision rather than a softer top-k blend.

The dispatch is implemented as genuine sparse computation, not a masked-average shortcut: for each expert, only the subset of tokens actually routed to it are run through that expert's weights (`nika_ai/model.py: MoELayer.forward`), scaled by the router's own confidence in that choice. This is exactly the mechanism that makes MoE models compute-efficient relative to a dense model of the same total parameter count — and, not coincidentally, exactly the mechanism that breaks static graph export (see the tracing section below).

A known, real failure mode of MoE training is **router collapse**: without any counter-pressure, a router can learn to send every token to the same expert, at which point the second expert is dead weight and the whole architecture has degenerated into a dense model with extra unused parameters. This project implements the actual fix used in production MoE models — a load-balancing auxiliary loss (`MoELayer._load_balancing_loss`), computed as `n_experts × Σᵢ(fᵢ × Pᵢ)` where `fᵢ` is the hard-routed fraction of tokens sent to expert `i` and `Pᵢ` is the average soft router probability assigned to expert `i`. This loss is minimized (at exactly 1.0) when routing is perfectly uniform and rises toward `n_experts` as routing collapses onto a single expert — verified against its closed-form value for both the balanced and fully-collapsed cases in `tests/test_moe_routing.py`. Critically, this loss does not fight domain specialization: if the two training domains are roughly equally represented, specializing by domain *is* the load-balanced solution, so the two objectives reinforce rather than oppose each other (see the results section, where this is demonstrated rather than just claimed).

## Tokenizer: built one from scratch, then used the real tool

Two BPE tokenizer implementations live side by side in `nika_ai/tokenizer.py`, deliberately:

- **`BPETokenizer`** — byte-level BPE implemented entirely from scratch in pure Python: starts from the 256 raw byte values, iteratively finds and merges the most frequent adjacent token pair, and replays the exact learned merge order (not just merge frequency) at encode time so that encoding is consistent with what was actually trained. Round-trip encode/decode correctness is verified directly.
- **`FastBPETokenizer`** — the same byte-level BPE algorithm, but backed by HuggingFace's Rust-implemented `tokenizers` library, which is what real tokenizer training actually uses in practice (the pure-Python version above is far too slow past toy-corpus sizes). `train.py` uses this one.

Keeping both was a deliberate choice: understanding the algorithm well enough to implement it, *and* knowing the standard, fast, production tool for actually running it, are both worth having — not an either/or.

One real, concrete finding from this project: with `vocab_size=4000` requested against this repository's ~14KB combined training corpus, the tokenizer trainer naturally stops early — around **~1,900 actual merges** — once it exhausts frequent pairs, rather than being forced through to 4,000 (which would produce degenerate, effectively-memorized long tokens). `train.py` resizes the model's embedding table to whatever vocabulary size was actually learned, so nothing silently breaks; this is itself a real, useful fact about BPE tokenizers that only becomes visible by actually training one against a small corpus and watching what happens.

## Training pipeline

Training (`nika_ai/train.py`) is a standard dense, teacher-forced language-modeling loop: every position in a fixed-length training sequence is predicted in parallel from the true previous tokens (not the model's own predictions), using a single causal-masked forward pass, with cross-entropy loss plus the MoE load-balancing loss (weighted small, so it nudges routing without dominating the language-modeling signal).

- **Optimizer**: AdamW with decoupled weight decay applied only to 2D+ weight matrices (RMSNorm's scale parameters are excluded, matching standard practice), and `betas=(0.9, 0.95)` rather than PyTorch's `(0.9, 0.999)` default — the more responsive second-moment estimate suited to the noisier gradient statistics of language-model training.
- **Learning-rate schedule**: manual linear warmup followed by cosine decay, implemented directly rather than via `torch.optim.lr_scheduler`, so the exact schedule is transparent and loggable.
- **Gradient clipping** and **gradient accumulation** are both implemented, giving the training loop the standard levers for stabilizing high-learning-rate, small-batch training.
- **Checkpointing** saves model weights, optimizer state, and both the model and training configs together in one file, so a checkpoint alone is enough to reconstruct the exact architecture — no separate config file to lose or desynchronize.
- **Held-out per-domain evaluation** (`evaluate_domain_losses`) tracks cross-entropy loss separately on each subject-matter domain's held-out split during training, which is what makes the specialization results below measurable rather than anecdotal.

## Distributed training: hand-rolled ring all-reduce vs. real DDP

`train.py --distributed-mode {manual,ddp}` exposes two independent, genuinely runnable ways to synchronize gradients across multiple training processes — built side by side specifically so that both "I've used the standard tool" and "I understand what the standard tool is hiding" are demonstrated, not just claimed.

**`manual`** — `nika_ai/distributed.py` reimplements ring all-reduce, the same bandwidth-optimal algorithm that NCCL and Horovod use internally, directly on top of `torch.distributed`'s raw point-to-point `isend`/`irecv` primitives, rather than calling the built-in `dist.all_reduce`. The algorithm runs in two phases — a scatter-reduce phase that circulates and sums chunks of each gradient tensor around a ring of processes, followed by an all-gather phase that circulates the now-fully-reduced chunks back around so every process ends up with the complete result — and was verified correct across 3 processes with a deliberately unevenly-sized tensor (`torch.tensor_split`'s uneven-chunk path), not just the easy evenly-divisible case.

**`ddp`** — wraps the model in real `torch.nn.parallel.DistributedDataParallel`, which triggers a genuine `all_reduce` automatically via autograd hooks the moment `.backward()` is called, with no explicit synchronization call needed in the training loop. Getting this right for a Mixture-of-Experts model surfaced a real, non-obvious correctness requirement: `find_unused_parameters=True` is not a performance knob here, it's necessary for correctness, because top-1 MoE routing means a given batch on a given process can route zero tokens to one expert, leaving that expert's parameters without a gradient that step — and DDP's default assumption is that every registered parameter receives one.

Both paths are launched identically, via `torchrun --nproc_per_node=N`, using the `gloo` backend (CPU/TCP-based — no NVIDIA hardware required), and both were run end-to-end, not just written and assumed correct.

## Multi-process vs. multi-node — an honest distinction

This project runs genuine multi-*process* distributed training: multiple independent OS processes, on one physical machine, communicating via real `torch.distributed` collective operations (`all_reduce`, point-to-point `send`/`recv`) over the `gloo` backend. This is not a simulation of collective communication — the collective operations, the inter-process communication overhead, and the gradient-synchronization correctness requirements are all real.

What it is *not*, and isn't presented as, is multi-*node* training: a genuine multi-node deployment runs these processes on physically separate machines connected by a real network (InfiniBand or Ethernet), with real network latency and bandwidth in place of local inter-process communication, topology-aware collective algorithms tuned to that network's actual shape, and no shared filesystem or shared memory to fall back on if something about the local setup was implicitly relied upon. None of that is exercised by running multiple processes on one Mac — and claiming otherwise would be a materially different (and false) claim. This project deliberately draws that line rather than blurring it.

## Graph tracing: torch.fx, torch.export, and where MoE routing breaks them

`nika_ai/tracing.py` (runnable directly: `python -m nika_ai.tracing`) is a hands-on investigation into the difference between eager PyTorch execution and its static graph representation — the representation a real compiler-ingestion or graph-optimization pipeline actually consumes.

1. **`torch.fx.symbolic_trace(Attention)` succeeds cleanly.** Attention has no data-dependent control flow — `is_causal` is a plain Python boolean fixed at trace time, not a tensor — so tracing it produces a clean, inspectable graph of explicit nodes (linear projections, view/transpose reshapes, the fused scaled-dot-product-attention call), each traceable back to a specific line of the original `forward()`.

2. **`torch.fx.symbolic_trace(MoELayer)` fails**, deliberately and instructively: `TraceError: symbolically traced variables cannot be used as inputs to control flow`. The cause is `if mask.any():` inside the MoE dispatch loop — fx traces tensors as symbolic placeholders ("Proxies"), and a Python `if` statement needs a concrete boolean to branch on, which a Proxy cannot provide.

3. **`torch.export.export(MoELayer, ...)` fails on the same line**, with a different and more specific error: `GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u0, 1)`. This is `torch.export`'s stricter, more descriptive diagnosis of exactly the same underlying problem: the graph's shape and control flow must be fully static, and a routing decision that depends on which expert a token was assigned to at runtime is fundamentally not static.

4. **`MoELayer.forward_traceable` fixes both**, and does so with the same computation, not an approximation of it: instead of conditionally dispatching each token to only its assigned expert, every expert is run densely on every token, and the results are combined with a one-hot mask multiply-and-sum instead of a boolean-indexed conditional write. There is no data-dependent branching and no data-dependent tensor shape anywhere in this version, so both `torch.fx.symbolic_trace` and `torch.export.export` succeed on it — and `tests/test_tracing.py::test_traceable_variant_matches_eager_numerically` confirms the two implementations produce numerically identical output (to `1e-6`), so the fix is a genuine re-expression of the same math, not a different model. The real cost of the fix is also made explicit rather than hidden: strictly more FLOPs, since every expert now runs on every token instead of only the subset actually routed to it — the concrete price of making a data-dependent decision statically traceable.

This is exactly the tension real Mixture-of-Experts model export and compiler-ingestion work runs into: routing decisions are inherently data-dependent, and a static graph needs that expressed through traceable tensor operations (`stack`, `one_hot`, masked multiply, `sum`) rather than raw Python control flow and boolean-mask indexing. `try_export_moe_traceable` also returns the full set of operator names present in the exported graph — a small, self-contained version of the "operator coverage check" step that real model-porting workflows perform before trusting an export.

## Training vs. inference: what actually changes structurally

The model's weights and layer definitions never change between training and inference — what changes is how data flows through them, and this project's `train.py`/`inference.py` split exists specifically to make that structural difference visible in code rather than only in explanation.

- **Parallel-over-sequence vs. sequential.** Training processes an entire fixed-length sequence in one parallel forward pass under teacher forcing: the true previous tokens are always fed in regardless of what the model would have predicted, with a dense causal mask enforcing the "no looking ahead" constraint. Generation cannot do this — the future tokens don't exist yet, since the model is producing them — so it is inherently sequential: produce token N+1, feed it back in as input, produce N+2. This sequential dependency is the entire reason a KV cache exists: without one, generating each new token would mean recomputing attention over the whole growing sequence from scratch, an `O(n²)` waste across a full generation instead of `O(n)`.
- **Prefill vs. decode, as two distinct, differently-bottlenecked phases.** Prefill (processing the initial prompt) is parallel, just like training — the whole prompt is known upfront, so one dense forward pass populates the KV cache for every prompt position at once, and this phase is typically **compute-bound** (large matrix multiplications over many tokens simultaneously). Decode (producing each subsequent token) processes exactly one new token per step while attending against an already-large cached K/V, and this phase is typically **memory-bound** — the dominant per-step cost is reading the entire KV cache from memory, not the trivial single-token compute itself. This is a real, named distinction in production model-serving systems, not just an implementation detail of this project.
- **Gradients and optimizer state.** Every training forward pass is paired with a backward pass — autograd builds and traverses a computation graph, and optimizer state (Adam's per-parameter moment estimates) is carried alongside the weights, meaningfully more memory than the weights alone. Inference wraps every forward pass in `torch.inference_mode()`: no computation graph is ever built, no optimizer state exists, and there is no `.backward()` call anywhere in `inference.py`.
- **Regularization behaves differently by mode.** Dropout is active under `model.train()` and disabled under `model.eval()` — demonstrated, not just asserted, in `tests/test_dropout.py`, the one test in this suite that intentionally turns dropout on (every other test sets `dropout_prob=0.0` for determinism). RMSNorm has no equivalent training/eval distinction the way BatchNorm would, since it involves no batch statistics — but knowing *why* `model.eval()` matters in general, and which layers it actually affects, is the point of calling it out explicitly.

## Testing philosophy: 18 tests, each proving something specific

Nothing in this suite is a smoke test that merely confirms code runs without crashing — each test is built to fail if a specific, plausible bug were reintroduced:

- **`test_rope.py`** proves RoPE's actual mathematical property (relative-position invariance of the attention dot product) numerically, not just that the shapes come out right, plus the rotate-half identity and norm preservation.
- **`test_moe_routing.py`** proves that MoE dispatch output for each token exactly equals running that token through *only* its assigned expert (not some blended average across experts), that the load-balancing loss matches its closed-form value in both the perfectly-balanced and fully-collapsed cases, and that a batch which happens to route to only one expert doesn't silently break gradients for the router or the unused expert.
- **`test_train_vs_inference_parity.py`** proves that dense training-time attention and one-token-at-a-time KV-cached inference produce numerically identical logits *and* identical MoE routing decisions at every position — this is the test that would catch a silent RoPE position-offset bug or a cache-indexing bug, neither of which would crash, only silently produce wrong output.
- **`test_tracing.py`** proves the export-breaking failure actually happens (with the expected error signatures) on the eager MoE implementation, that the traceable fix actually succeeds under both `torch.fx` and `torch.export`, and that the fix computes the identical result to the original.
- **`test_dropout.py`** proves dropout is genuinely stochastic in train mode and genuinely deterministic in eval mode, for the same input and the same model.

## Results: does the MoE actually specialize?

The two-expert MoE is trained on a deliberately mixed corpus of two distinguishable subject-matter domains — original narrative prose (`data/prose.txt`) and original Python source code (`data/code.txt`), both written for this project rather than scraped, so the domains are unambiguous and copyright-free. After training, `nika_ai.inference.analyze_routing` / `expert_usage_fraction` measure which expert handled which domain's tokens, layer by layer.

In one representative run (200 training steps), layer 0's router specialized sharply by domain: **prose tokens routed to expert 0 97.7% of the time; code tokens routed to expert 0 only 17.2% of the time (i.e., 82.8% to expert 1)** — a strong, interpretable specialization signal after a very small amount of training on a very small amount of text, not an average-case coincidence. `notebooks/analysis.ipynb` runs this comparison properly: training the MoE model alongside a dense (`n_experts=1`) baseline of otherwise identical shape (the same `MoELayer` code, degenerated to a single expert — reused rather than duplicated as a second hand-written model), plotting held-out loss per domain for both, and visualizing per-layer, per-domain expert usage as bar charts.

## Repository layout

```
nika_ai/
├── nika_ai/
│   ├── model.py          # NikaMoEConfig, Attention, MoELayer (+ forward_traceable), RMSNorm, NikaMoETransformer
│   ├── tokenizer.py       # BPETokenizer (from scratch) + FastBPETokenizer (tokenizers library, used by default)
│   ├── data.py            # PackedTextDataset + the two subject-matter corpora loaders
│   ├── train.py           # training loop: single-process, hand-rolled ring all-reduce, or real DDP
│   ├── inference.py        # KV-cached generation, routing analysis
│   ├── distributed.py       # hand-rolled ring all-reduce
│   └── tracing.py           # torch.fx / torch.export demos, including the MoE routing export break + fix
├── data/
│   ├── prose.txt           # domain "prose": original narrative English text
│   └── code.txt            # domain "code": original Python source
├── tests/
│   ├── test_rope.py
│   ├── test_moe_routing.py
│   ├── test_train_vs_inference_parity.py
│   ├── test_tracing.py
│   └── test_dropout.py
├── notebooks/
│   └── analysis.ipynb      # imports from nika_ai, visualizes results only — no model/training code lives here
└── README.md
```

`model.py` is architecture only — no training loop, no generation loop. Both `train.py` and `inference.py` import from it, so the model definition can never silently drift between the two, a guarantee that is easy to lose in less disciplined codebases.

## Setup and quickstart

```bash
pip install -e .
pip install pytest matplotlib tokenizers   # for tests / notebook / fast tokenizer
```

**Train** (single process; trains its own tokenizer from `data/prose.txt` + `data/code.txt`):

```bash
python -m nika_ai.train --max-steps 300 --checkpoint-dir checkpoints
```

**Train, distributed** — either mechanism, both runnable locally via `torchrun` (gloo/CPU backend, no CUDA required):

```bash
torchrun --nproc_per_node=2 -m nika_ai.train --distributed-mode manual --max-steps 300   # hand-rolled ring all-reduce
torchrun --nproc_per_node=2 -m nika_ai.train --distributed-mode ddp    --max-steps 300   # real DistributedDataParallel
```

**Generate text**:

```bash
python -m nika_ai.inference checkpoints/ckpt_final.pt "def "
```

**Run the graph-tracing demo** (see the MoE export failure and fix happen live):

```bash
python -m nika_ai.tracing
```

**Run the test suite**:

```bash
pytest tests/ -v
```

**Open the analysis notebook** (loss curves + routing specialization, MoE vs. dense baseline):

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Honest scope notes

- The model (~9M parameters), the tokenizer's target vocabulary, and the training corpora are all intentionally toy-scale. This is a from-scratch engineering and systems-understanding exercise, not a competitive language model, and nothing in this README should be read as claiming otherwise.
- The multi-process distributed training in this repository is real; the multi-node claims it is *not* making are stated explicitly above, not left ambiguous.
- `data/prose.txt` and `data/code.txt` are original texts written specifically for this project, not scraped from any external source.
