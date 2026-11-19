"""Perplexity in the protocol used by GPTQ/AWQ papers and most quant repos.

- WikiText-2: the raw *test* split, joined with "\\n\\n", tokenized once.
- C4: the fixed validation shard most papers use, first 256 x seqlen tokens.
- Non-overlapping windows of `seqlen` tokens (default 2048); trailing partial
  window dropped; loss averaged over all predicted tokens.

Published numbers depend on every one of these choices, so they are written into
the results JSON. If your fp16 number doesn't match a paper, check seqlen and
whether that paper used overlapping (strided) windows before suspecting the model.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def load_eval_tokens(dataset: str, tokenizer, seqlen: int = 2048, c4_windows: int = 256) -> torch.Tensor:
    from datasets import load_dataset

    if dataset == "wikitext2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
        ids = tokenizer(text, return_tensors="pt").input_ids
    elif dataset == "c4":
        ds = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        text = " ".join(ds[:1100]["text"])
        ids = tokenizer(text, return_tensors="pt").input_ids[:, : c4_windows * seqlen]
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return ids


@torch.no_grad()
def perplexity(
    model: nn.Module,
    tokenizer,
    dataset: str = "wikitext2",
    seqlen: int = 2048,
    max_windows: int | None = None,
    device: str = "cuda",
) -> dict:
    ids = load_eval_tokens(dataset, tokenizer, seqlen)
    n_windows = ids.shape[1] // seqlen
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    model.eval()
    nll_sum, n_tokens = 0.0, 0
    for i in range(n_windows):
        batch = ids[:, i * seqlen : (i + 1) * seqlen].to(device)
        logits = model(batch, use_cache=False).logits
        # Shift so position t predicts token t+1; float32 for the loss.
        loss = nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            batch[:, 1:].reshape(-1),
            reduction="sum",
        )
        nll_sum += loss.item()
        n_tokens += batch.shape[1] - 1

    return {
        "dataset": dataset,
        "seqlen": seqlen,
        "windows": n_windows,
        "tokens": n_tokens,
        "ppl": math.exp(nll_sum / n_tokens),
    }
