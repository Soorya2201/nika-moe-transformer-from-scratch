"""
nika_ai.tokenizer — minimal from-scratch byte-level BPE.

Starts from the 256 raw byte values as the base vocabulary and iteratively merges
the most frequent adjacent pair into a new token, exactly like GPT-2's tokenizer
(Sennrich et al.'s BPE applied to bytes instead of characters, avoiding any
"unknown token" problem since every possible byte sequence is representable).
No dependency on tiktoken/sentencepiece — this is the "from scratch" project's
tokenizer too.
"""

from __future__ import annotations

import json


class BPETokenizer:
    def __init__(self):
        # merges records the ORDER pairs were learned in (dict preserves insertion
        # order in Python 3.7+), which encode() relies on to replicate training's
        # merge priority instead of a greedy single pass.
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int) -> None:
        assert vocab_size >= 256, "vocab_size must be at least 256 (the base byte alphabet)"
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))
        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            counts = self._pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + i
            ids = self._merge(ids, pair, new_id)
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = self._pair_counts(ids)
            # Apply the pair that was learned EARLIEST in training (lowest merged
            # id), not just the most frequent pair in this text — this replicates
            # training's merge order, which is what makes encode/decode consistent
            # with the vocabulary actually learned.
            candidates = [p for p in counts if p in self.merges]
            if not candidates:
                break
            pair = min(candidates, key=lambda p: self.merges[p])
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        # Save merges as an ordered list of [a, b, new_id] triples: order is load-bearing.
        payload = {"merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()]}
        with open(path, "w") as f:
            json.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            payload = json.load(f)

        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for a, b, new_id in payload["merges"]:
            merges[(a, b)] = new_id
            vocab[new_id] = vocab[a] + vocab[b]

        self.merges = merges
        self.vocab = vocab

    @staticmethod
    def _pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        merged = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged
