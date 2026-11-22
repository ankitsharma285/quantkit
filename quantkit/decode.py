"""Greedy batch-1 decoding with a static KV cache, eager or replayed from a CUDA graph.

A static cache keeps every tensor at a fixed address, which CUDA-graph capture
requires; eager mode uses the same cache so the two differ only in how kernels
are launched. Argmax stays on the GPU, so the decode loop never synchronizes
with the host.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import StaticCache


class Decoder:
    def __init__(self, model: nn.Module, prompt_len: int, new_tokens: int):
        self.model = model
        self.P, self.T = prompt_len, new_tokens
        self.cache = StaticCache(config=model.config, max_cache_len=prompt_len + new_tokens)
        dev = next(model.parameters()).device
        self.positions = torch.arange(prompt_len, prompt_len + new_tokens, device=dev)
        self.graph = None

    def _forward(self, ids, cache_position):
        return self.model(input_ids=ids, past_key_values=self.cache, cache_position=cache_position,
                          use_cache=True).logits

    @torch.no_grad()
    def prefill(self, prompt: torch.Tensor) -> torch.Tensor:
        """prompt [1, P] -> first generated token [1, 1]."""
        self.cache.reset()
        pos = torch.arange(self.P, device=prompt.device)
        return self._forward(prompt, pos)[:, -1:].argmax(-1)

    @torch.no_grad()
    def step(self, ids: torch.Tensor, t: int) -> torch.Tensor:
        """Eager decode step t: token ids [1, 1] at position P + t -> next token [1, 1]."""
        return self._forward(ids, self.positions[t : t + 1])[:, -1:].argmax(-1)

    @torch.no_grad()
    def capture(self, warmup: int = 3) -> None:
        """Record one decode step as a CUDA graph (call after prefill and kernel warm-up)."""
        self._ids = torch.zeros(1, 1, dtype=torch.long, device=self.positions.device)
        self._pos = self.positions[:1].clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._forward(self._ids, self._pos)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._logits = self._forward(self._ids, self._pos)

    @torch.no_grad()
    def graph_step(self, ids: torch.Tensor, t: int) -> torch.Tensor:
        self._ids.copy_(ids)
        self._pos.copy_(self.positions[t : t + 1])
        self.graph.replay()
        return self._logits[:, -1:].argmax(-1)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, use_graph: bool = False) -> torch.Tensor:
        """Greedy-generate T tokens after the prompt; returns [1, T]."""
        tok = self.prefill(prompt)
        out = [tok]
        stepper = self.graph_step if use_graph else self.step
        for t in range(self.T - 1):
            tok = stepper(tok, t)
            out.append(tok)
        return torch.cat(out, dim=1)
