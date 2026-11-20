"""Calibration samples: n random windows of `seqlen` tokens, fixed by a seed.

The same tokens are used for every method (and for the GPTQModel reference
run), so differences in results come from the method, not the data. C4 follows
the GPTQ paper: random documents from the first train shard, each long enough
to hold a full window. WikiText-2 (train split) is for the calibration-domain
ablation; it must not be calibrated on its own test split.
"""
from __future__ import annotations

import random

import torch


def get_calib_tokens(
    tokenizer,
    dataset: str = "c4",
    n_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
) -> torch.Tensor:
    """Return LongTensor [n_samples, seqlen]."""
    from datasets import load_dataset

    rng = random.Random(seed)
    if dataset == "c4":
        ds = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        samples = []
        while len(samples) < n_samples:
            doc = ds[rng.randint(0, len(ds) - 1)]["text"]
            ids = tokenizer(doc, return_tensors="pt").input_ids
            if ids.shape[1] <= seqlen:
                continue
            start = rng.randint(0, ids.shape[1] - seqlen - 1)
            samples.append(ids[:, start : start + seqlen])
        return torch.cat(samples, dim=0)

    if dataset == "wikitext2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids
        starts = [rng.randint(0, ids.shape[1] - seqlen - 1) for _ in range(n_samples)]
        return torch.cat([ids[:, s : s + seqlen] for s in starts], dim=0)

    raise ValueError(f"unknown calibration dataset {dataset!r}")
