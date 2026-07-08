"""
nika_ai.data — packed dataset + the two subject-matter corpora used to train and
evaluate the MoE experts' specialization.

Domains:
  "prose" — narrative English text (data/prose.txt)
  "code"  — Python source (data/code.txt)

The two domains are meant to be easy to tell apart, so that after training we can
ask "which expert handles which domain" and get an interpretable answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_PACKAGE_DIR), "data")

DOMAINS = ("prose", "code")


def load_domain_text(domain: str) -> str:
    assert domain in DOMAINS, f"unknown domain {domain!r}, expected one of {DOMAINS}"
    path = os.path.join(_DATA_DIR, f"{domain}.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


class PackedTextDataset(Dataset):
    """Tiles a flat token-id array into contiguous, non-overlapping (input, target) windows.

    Deterministic tiling (rather than random-offset sampling) is the right choice
    for a small fixed corpus: every token gets seen exactly once per epoch, which
    matters when the whole dataset is only a few thousand tokens.
    """

    def __init__(self, token_ids, seq_len: int, domain_label: int | None = None):
        self.token_ids = np.asarray(token_ids, dtype=np.int64)
        self.seq_len = seq_len
        self.domain_label = domain_label
        self.n_sequences = max(0, (len(self.token_ids) - 1) // seq_len)

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        x = torch.from_numpy(self.token_ids[start : start + self.seq_len])
        y = torch.from_numpy(self.token_ids[start + 1 : start + 1 + self.seq_len])
        if self.domain_label is None:
            return x, y
        return x, y, self.domain_label


@dataclass
class DomainSplit:
    train: PackedTextDataset
    val: PackedTextDataset


def build_datasets(tokenizer, seq_len: int, val_fraction: float = 0.1) -> tuple[Dataset, dict[str, DomainSplit]]:
    """Encode both domain corpora, split each into train/val, return (combined_train, per-domain splits).

    combined_train mixes both domains for actual training. domain_splits gives
    each domain's held-out val set for the per-domain loss comparison, and either
    split can be fed to inference.analyze_routing() for the routing/specialization
    check.
    """
    domain_splits: dict[str, DomainSplit] = {}
    train_sets = []

    for label, domain in enumerate(DOMAINS):
        ids = tokenizer.encode(load_domain_text(domain))
        split_idx = int(len(ids) * (1 - val_fraction))
        train_ids, val_ids = ids[:split_idx], ids[split_idx:]

        train_ds = PackedTextDataset(train_ids, seq_len, domain_label=label)
        val_ds = PackedTextDataset(val_ids, seq_len, domain_label=label)
        domain_splits[domain] = DomainSplit(train=train_ds, val=val_ds)
        train_sets.append(train_ds)

    combined_train = ConcatDataset(train_sets)
    return combined_train, domain_splits
