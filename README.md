# quantkit: from-scratch INT4 weight quantization with a fused Triton W4A16 kernel

RTN, GPTQ and AWQ implemented from the papers, validated against the reference libraries, and served
through a fused Triton dequant-GEMM kernel so the quantized model is actually faster, not just smaller.
Everything runs on a single NVIDIA L4 (24 GB, ~300 GB/s) with Llama-3.2-1B as the main model.

Most quantization reimplementations stop at fake quantization (quantize, dequantize back to fp16, report
perplexity). That shows accuracy but no speedup. This project closes the loop: packed int4 weights,
a kernel that reads them directly, and end-to-end decode throughput and memory against fp16.

**Status:** Phases 1 (harness, RTN, int4 packing) and 2 (GPTQ) complete on the L4. Open items: citing a
published fp16 baseline with a matching protocol, and an unexplained ~3% gap to GPTQModel (see Phase 2 finding).

## Objectives

1. **Correct quantization, verified against references.** Implement RTN, GPTQ and AWQ from scratch and
   match the perplexity of GPTQModel (GPTQ) and llm-compressor (AWQ) on the same model, calibration set
   and settings, within 1% relative.
2. **Real speedup from a custom kernel.** A fused Triton W4A16 kernel that dequantizes inside the GEMM
   loop, correct to fp16 tolerance against dequantize-then-matmul, and faster than fp16 cuBLAS for decode.
3. **Explain where int4 wins and where it loses.** Show, with measured bandwidth and a roofline, that
   W4A16 helps memory-bound decode (small M) and stops helping as M grows and the GEMM becomes
   compute-bound.
4. **Honest ablations.** Bits (3/4), group size, calibration set size and domain, and which layers are
   most sensitive, all with the same evaluation protocol.
5. **Reproducible engineering.** Installable package, unit tests, one results JSON per run with the full
   environment recorded, and a notebook that only clones, installs and calls scripts.

### Non-goals (this version)

Activation quantization (W8A8, SmoothQuant), quantization-aware training, FP8, multi-GPU, and
on-device export. On-device export (ExecuTorch on ARM/Android) is planned as a second backend that reuses
this core; see [Roadmap](#roadmap).

## Methods

All three methods share the same quantizer (`quant/rtn.py`) and storage format; they differ only in
how the integer codes and scales are chosen.

| Method | Idea | Calibration data | Key details implemented |
| --- | --- | --- | --- |
| **RTN** | Round each weight to the nearest grid point, per group | None | Asymmetric/symmetric, per-channel or group-wise, 2–8 bits |
| **GPTQ** (Frantar et al., 2022) | Quantize weight columns one at a time and spread the error onto not-yet-quantized columns using the inverse Hessian of the layer's inputs | 128 × 2048 tokens | H = 2XXᵀ accumulated per layer; 1% diagonal damping; Cholesky of H⁻¹; blocked lazy updates (block 128); optional act-order with static groups |
| **AWQ** (Lin et al., 2023) | Protect salient weights by scaling input channels by activation magnitude before quantizing, then fold the inverse scale into the previous op | 128 × 512 tokens | Grid search over α ∈ [0, 1] (20 points) minimizing block output MSE; weight-clipping search; scales folded into RMSNorm / previous Linear |

**GPTQ implementation notes.**

- *Verified against the textbook algorithm.* `tests/test_gptq.py` runs a naive Optimal Brain Surgeon loop
  (one column at a time, explicit H⁻¹ with the column removed after each step, no Cholesky, no blocking).
  The blocked Cholesky implementation produces bit-identical codes in float64 for group sizes 32, 128,
  256 and per-channel.
- *True sequential.* Within each decoder layer, Linears are quantized in execution order
  (q/k/v → o → gate/up → down), and each sub-group's Hessian is collected after the previous sub-groups
  are already quantized. Each layer's outputs are recomputed with quantized weights before they become
  the next layer's calibration inputs.
- *Scales are fp16 during the search.* Group scales are rounded to fp16 before quantizing, because the
  error feedback depends on the exact stored values.
- *Act-order uses static groups.* Group scales are fixed from the original contiguous columns before
  reordering, so the storage format and kernel need no per-column group index.
- *Damping.* The paper uses 1% of the mean Hessian diagonal; GPTQModel 7.x defaults to 5%, citing a
  massive activation on Llama's BOS token that can dominate the Hessian. Both are run.

**Llama-3.2 GQA detail for AWQ.** `o_proj` scales normally fold into `v_proj`, but with grouped-query
attention `v_proj` outputs 512 channels while `o_proj` takes 2048, so the fold is shape-incompatible.
Like the reference implementation, `o_proj` is left unscaled; the ablation table reports its effect.

### Storage format

The contract between the quantizers and the kernel (`quant/pack.py`):

```
qweight : [K/8, N]  int32   8 nibbles packed along K, little-nibble first
scales  : [K/G, N]  fp16
qzeros  : [K/G, N]  uint8
W[n,k]  = (q[k,n] - qzeros[k/G, n]) * scales[k/G, n]
```

Packing along K with N contiguous means a (BLOCK_K, BLOCK_N) kernel tile loads BLOCK_K/8 rows of
coalesced int32s. `lm_head` and embeddings stay fp16 (quantizing `lm_head` is an ablation).

### Kernel

At decode (M = 1) a Linear is a matrix-vector product: almost no compute, and the time is spent reading
weights. Int4 cuts weight bytes ~3.8× (4 bits + ~0.25 bits of scales/zeros per weight), so the ceiling
on a single layer is close to that. Two code paths:

- **Small M, M ≤ 16 (split-K, `_w4a16_splitk`):** one 16-row block covers all tokens, so the grid is only
  N / BLOCK_N programs (16 for `k/v_proj` at BLOCK_N = 32, against the L4's 58 SMs). A second grid dimension
  splits K across SPLIT_K programs, which atomically add fp32 partial sums into the output.
- **Larger M (tiled GEMM, `_w4a16_gemm`):** standard (M, N) tiling with the full K loop per program.

Both dequantize in registers: tile row k reads packed row k / 8, nibble k mod 8; BLOCK_K divides the group size,
so each K-tile needs one row of scales and zeros. Dequantized weights are bit-identical to the reference path;
outputs differ only in summation order (and, for split-K, atomic-add order, so they are not bitwise
deterministic run to run). Block sizes, SPLIT_K, warps and pipeline stages are autotuned per (M, N, K).

Correctness: `tests/test_kernel.py` runs both kernels against an fp32 reference on the exact dequantized
weights, including N and M not divisible by the block sizes, extreme codes and zero points, and (on a GPU) the
autotuned kernels on all Llama-3.2-1B layer shapes for M from 1 to 255. Without a GPU the tests run the real
kernel code in Triton's CPU interpreter; deliberately broken nibble shifts, zero points and group indexing all
make them fail.

The crossover M between the two paths, and between W4A16 and fp16, is measured rather than assumed
(`scripts/bench_kernel.py`).

**End-to-end ceiling.** Llama-3.2-1B ties a 128,256 × 2048 fp16 embedding/`lm_head` (~0.5 GB), about a
fifth of the fp16 model. Decode reads it every token, so even with perfect int4 kernels the per-token
bytes fall from ~2.5 GB to ~1.0 GB, about a 2.4× ceiling, not 3.8×. Results are reported against this
ceiling.

## Evaluation protocol

| Item | Setting |
| --- | --- |
| Model | `meta-llama/Llama-3.2-1B` (main); Llama-3.2-3B as a scale check |
| Format | INT4, asymmetric, group size 128 unless stated |
| Calibration | Fixed C4 sample, seed 0, same for every method; 128 × 2048 tokens (GPTQ), 128 × 512 (AWQ) |
| Perplexity | WikiText-2 test and C4 validation shard, non-overlapping 2048-token windows |
| Downstream | lm-eval-harness: ARC-e, ARC-c, HellaSwag, PIQA, WinoGrande (0-shot, acc_norm where defined) |
| Kernel benchmarks | `triton.testing.do_bench`, median of repeats; configs interleaved with cooldowns (the L4 is power-capped at 72 W and throttles under sustained load) |
| Speed metrics | Kernel time, achieved GB/s as % of peak, decode tokens/s, time to first token, peak memory |
| Environment | GPU name, driver, clocks, power limit, CUDA/torch/Triton/transformers versions saved in every results JSON |

## Results

_Filled in as phases complete. Every cell is rebuilt from `results/*.json`._

**Accuracy (Llama-3.2-1B, W4 g128 unless stated)**

| Method | Bits | Group | WikiText-2 ↓ | C4 ↓ | Avg 0-shot ↑ | Reference library |
| --- | --- | --- | --- | --- | --- | --- |
| fp16 | 16 | – | 9.751 | 14.012 | | – |
| RTN | 4 | per-channel | 18.084 (+85%) | 27.622 (+97%) | | – |
| RTN | 4 | 128 | 11.728 (+20%) | 17.439 (+24%) | | – |
| GPTQ (paper defaults) | 4 | 128 | 34.171 (+250%) | 74.037 | | GPTQModel, same settings: 50.863 |
| GPTQ, damp 0.05 | 4 | 128 | 81.593 | 130.603 | | |
| GPTQ, first calibration token left out of H | 4 | 128 | 11.155 (+14%) | | | |
| GPTQ, first 4 tokens left out of H | 4 | 128 | 11.099 (+14%) | | | |
| GPTQ, static groups (no reordering) | 4 | 128 | 35.795 | | | |
| GPTQ, static groups, first token left out of H | 4 | 128 | 11.145 | | | |
| **GPTQ + act-order (static groups)** | 4 | 128 | **10.780 (+11%)**; 3 seeds 10.71 ± 0.11 | 15.320 (+9%) | | GPTQModel (bf16): 3 seeds 11.01 ± 0.02 |
| GPTQ + act-order, first token left out of H | 4 | 128 | 11.238 | | | |
| AWQ | 4 | 128 | | | | llm-compressor: |
| RTN (fake quant) | 3 | 128 | 35.041 (+259%) | 57.175 (+308%) | | – |
| GPTQ (fake quant) | 3 | 128 | 32.172 | 43.773 | | |
| AWQ (fake quant) | 3 | 128 | | | | |

Percentages are relative to fp16. The packed W4 g128 checkpoint, saved and reloaded from disk, gives
WikiText-2 perplexity 11.728, identical to the in-memory run. Plain RTN costs Llama-3.2-1B 20% perplexity
at W4 g128, which sets the gap GPTQ and AWQ must close.

**Phase 2 finding: vanilla GPTQ breaks on Llama-3.2-1B, and column order is what fixes it.**

- With the paper's settings, GPTQ W4 g128 reaches WikiText-2 perplexity 34.2, three times worse than plain
  RTN (11.7). GPTQModel with the same settings and calibration windows also breaks (50.9), so this is not
  specific to this implementation.
- Leaving the first position of each calibration window out of the Hessians recovers 11.16. The first
  token of a sequence becomes an attention sink with massive activations in a few channels; one such token
  per window is enough to wreck GPTQ in natural column order.
- *Not* the group scales: fixing scales from the original weights without reordering still breaks (35.8).
- Act-order (columns in decreasing Hessian diagonal, static groups) fixes it without dropping any tokens:
  10.71 ± 0.11 over 3 calibration seeds, −8.7% vs RTN. With act-order, dropping the first token makes
  things *worse* (11.24, about 5 standard deviations of the seed spread), so the sink token carries useful
  information once columns are ordered.
- Raising damping to 5% makes plain GPTQ worse (81.6), not better; not yet explained.
- GPTQ's own objective does not predict the outcome. Summed layer-wise reconstruction loss on the calibration
  set is *lower* for plain GPTQ (196) than for act-order (217), yet perplexity is 34.2 vs 10.8: plain GPTQ fits
  the calibration set, sink tokens included, and fails to generalize. The most lopsided Hessians are attention
  `q_proj` inputs (diagonal max/median up to ~230, vs under ~75 for MLP inputs).
- Parity with GPTQModel (matched act-order settings, same calibration windows):
  - GPTQModel is worse on every seed: 11.01 ± 0.02 vs 10.71 ± 0.11 here, a systematic ~3% gap.
  - Group scales are identical in every module; integer codes agree on 75–91% of weights, with no growing
    drift through the layers.
  - Code agreement is not a quality measure for GPTQ: re-running quantkit with TF32 matmuls flips about as
    many codes (15% by layer 1) yet moves perplexity by 0.1% (10.768 vs 10.780). Every rounding error feeds
    all later columns, so one early flip changes the rest of the row.
  - Not dtype: GPTQModel calibrates and evaluates in bfloat16, quantkit in float16, but switching precision
    barely moves either side. The fp16 model evaluated in bf16: 9.757 vs 9.751; quantkit GPTQ run fully in
    bf16: 10.819 vs 10.780; GPTQModel's checkpoint evaluated in fp16: 11.031 vs 11.028.
  - The remaining gap is unexplained. Candidates not tested: differences in how GPTQModel captures and
    deduplicates layer inputs for the Hessians, and where damping is applied. Correctness of this
    implementation rests on the exact match with the naive OBS reference, identical group scales, no drift
    across layers, and a seed-tested gain over RTN.

**Kernel (per layer shape, M = 1)**

Practical DRAM ceiling of this L4, measured (`scripts/bw_ceiling.py`): **243 GB/s** for a streaming read
(81% of the 300 GB/s spec), 231 GB/s for a copy, 232 GB/s for cuBLAS on an 8192 × 8192 fp16 matrix-vector
product. Percentages below are of the measured 243 GB/s.

| Layer (N × K) | fp16 cuBLAS µs | W4A16, Phase 4 → 5 µs | Speedup, Phase 4 → 5 | Achieved GB/s: W4A16 / fp16 |
| --- | --- | --- | --- | --- |
| q/o_proj 2048 × 2048 | 54.3 | 29.7 → 24.6 | 1.83× → **2.21×** | 89 (37%) / 154 (64%) |
| k/v_proj 512 × 2048 | 18.4 | 16.4 → 15.4 | 1.12× → **1.19×** | 36 (15%) / 114 (47%) |
| gate/up_proj 8192 × 2048 | 192.5 | 73.7 → 57.9 | 2.61× → **3.32×** | 152 (62%) / 174 (72%) |
| down_proj 2048 × 8192 | 187.4 | 76.3 → 60.4 | 2.46× → **3.10×** | 145 (60%) / 179 (74%) |

Speedup over fp16 cuBLAS as M grows (measured dispatch, median of 3 interleaved rounds, L2 flushed):

| Layer | M=1 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- |
| q/o_proj | 2.21× | 2.00× | 1.60× | 1.33× | 1.11× | 1.00× |
| k/v_proj | 1.20× | 1.17× | 0.96× | 0.82× | 0.73× | 0.71× |
| gate/up_proj | 3.32× | 3.26× | 3.32× | 2.39× | 1.87× | 1.37× |
| down_proj | 3.10× | 2.98× | 1.69× | 1.36× | 1.30× | 0.89× |

(Measured before the dispatcher also chose the unpack variant. At M = 32 the per-nibble kernel was faster:
1.75× on q/o_proj, 1.15× on k/v_proj and 1.92× on down_proj.)

**Phase 4 → 5: what each change bought** (same-session ablation in `results/phase5/bench_l4_v2.json`):

- *Loading each packed int32 once per tile* (instead of once per nibble) is the main win on large layers:
  split-K at M = 1 went from 73.7 to 63.5 µs on gate/up_proj and 75.8 to 62.5 µs on down_proj; the tiled
  kernel at M = 64 from 124.9 to 92.2 µs. It is not uniformly better: on k/v_proj the tiled kernel got slower
  (16.4 → 21.5 µs), and at M = 32 per-nibble won on three of four shapes. The dispatcher now measures both.
- *A tiled kernel with 16-row blocks* (single launch, direct fp16 store) is the fastest decode kernel for
  gate/up_proj (58 µs vs 64 µs for split-K).
- *A GEMV kernel for M = 1* (no tensor cores) wins narrowly on down_proj and k/v_proj.
- *Measured dispatch* replaced the fixed M ≤ 16 rule, which picked a kernel up to 50% slower than the best.

On the two large layers the kernel now reaches 81–87% of cuBLAS's achieved bandwidth, so decode is 3.1–3.3×
faster against an ideal of 3.8× (int4 moves 3.8× fewer bytes).

**Why small layers stay slow** (Nsight Compute, k/v_proj, M = 1, split-K): the matmul kernel itself takes
7.0 µs at 30% of DRAM bandwidth, using 128 thread blocks on 58 SMs (17% achieved occupancy, 96 registers per
thread). The zero-fill (1.8 µs) and fp16 cast (2.2 µs) kernels plus the gaps between launches account for the
rest of the 15.4 µs. A 0.55 MB weight read cannot fill the GPU in one call. The standard fix is at the model
level, not the kernel: fuse q/k/v into one matmul (N = 3072) and gate/up into one (N = 16384), so the small
projection never runs alone. That is measured end to end in Phase 6.

The `reference` path (dequantize all of W to fp16, then matmul) is 15–30× *slower* than fp16 (Phase 4
benchmark), so Phase 1–2 perplexity runs measured accuracy only, never speed.

**End to end (decode, batch 1)**: Llama-3.2-1B, GPTQ W4 g128 act-order, 128-token prompt, 256 greedy tokens,
static KV cache (`scripts/bench_decode.py`).

| Config | Weights GB | Peak GB | Prefill ms | Eager tok/s | CUDA-graph tok/s | GPU ms/token | vs fp16 (graph) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | 2.47 | 2.53 | 22.1 | 49.1 | 85.3 | 11.58 | 1.00× |
| fp16, q/k/v + gate/up fused | 2.47 | 2.63 | 27.3 | 38.1 | 86.2 | 11.46 | 1.01× |
| W4A16 | 1.03 | 1.11 | 41.1 | 22.0 | 154.6 | 6.36 | 1.81× |
| **W4A16, fused** | **1.03** | **1.21** | 35.5 | 26.9 | **155.5** | **6.13** | **1.82×** |

Accuracy of the fast path: WikiText-2 perplexity through the fused Triton kernels is 10.781 vs 10.780 on the
reference path (fp16 model: 9.751). CUDA-graph and eager decoding produce identical tokens.

Where the GPU time per token goes:

| Config | int4 matmuls | other matmuls (cuBLAS) | attention | everything else | total |
| --- | --- | --- | --- | --- | --- |
| fp16 | – | 9.92 ms (all linears + lm_head) | 0.34 | 1.32 | 11.58 ms |
| W4A16, fused | 2.28 ms | 2.06 ms (fp16 lm_head) | 0.33 | 1.46 | 6.13 ms |

What this shows:

- *Decode is 1.8× faster with CUDA graphs, and the graph numbers are bandwidth-bound:* graph time per token is
  within 0.1–0.3 ms of pure GPU kernel time for every config.
- *The int4 linears are 3.4× faster than fp16 in the model* (7.86 → 2.28 ms per token), moving weights at about
  223 GB/s; in the model, cuBLAS moves fp16 weights at about 247 GB/s, faster than the standalone L2-flushed
  benchmark suggested.
- *The end-to-end ceiling is lower than the 2.4× estimated earlier,* because that estimate counted weights only.
  In the W4A16 model, the fp16 lm_head (2.06 ms, 34%) and the many small non-matmul kernels (norms, rotary
  embeddings, residual adds, KV-cache writes, the split-K zero-fill and cast: 1.46 ms, 24%) now cost as much as
  all the int4 matmuls. With perfect int4 kernels the limit is about 5.7 ms per token, about 2.0×.
- *Fusing q/k/v and gate/up barely matters end to end* (+0.6% with graphs, 0.23 ms of GPU time), even though it
  removes the slow small-layer calls: those layers were a small share of total time.
- *Without CUDA graphs, W4A16 decode is 2.2× slower than fp16* (22.0 vs 49.1 tok/s). Eager decoding at batch 1
  is limited by the CPU launching kernels: about 9 ms of host overhead per token for fp16, about 31–39 ms for
  W4A16, i.e. roughly 0.3 ms of Python launch cost per Triton matmul call (Triton launcher, autotuner lookup,
  dispatch, output allocation). Eager prefill is slower for the same reason (35–41 vs 22 ms). Serving stacks
  (vLLM, SGLang, TensorRT-LLM) decode from CUDA graphs, so the graph numbers are the relevant ones, but the
  custom kernel is only a win there.

Next steps this points to: quantize lm_head (removes up to 2 ms of the 6.1 ms), cut the small-kernel time with
operator fusion (e.g. torch.compile; helps fp16 and W4A16 alike), and launch the Triton kernels through a cached
compiled handle instead of the Python autotuner path to fix eager mode.

Plus: a speedup-vs-M plot for M ∈ {1, 2, 4, …, 256} with the crossover marked, and a roofline plot.

## Ablations

1. **Group size:** per-channel, 256, 128, 64, 32: accuracy vs storage overhead.
2. **Bits:** 3 vs 4 for each method (3-bit as fake quantization; the kernel is int4 only).
3. **Calibration size:** 16, 32, 64, 128, 256 samples for GPTQ and AWQ.
4. **Calibration domain:** C4 vs WikiText-2 vs code: does calibrating on the eval domain inflate perplexity
   gains? Reported on both eval sets to make this visible.
5. **Layer sensitivity:** quantize one block type at a time (attention vs MLP, early vs late layers).
6. **GPTQ act-order** on/off, and **AWQ** with and without clipping search and `o_proj` handling.
7. **`lm_head` quantized** vs fp16: accuracy cost against the end-to-end ceiling it removes.

## Roadmap

| Phase | Deliverable | Done when |
| --- | --- | --- |
| 1 · Harness + RTN ✅ | Perplexity eval, RTN, int4 packing, `QuantLinear` (reference path), save/load | ✅ RTN orders as fp16 < g128 < per-channel; ✅ reloaded checkpoint reproduces in-memory perplexity exactly; ✅ unit tests pass; ⏳ fp16 perplexity matches a cited source and protocol |
| 2 · GPTQ ✅ | Layer-by-layer calibration capture, Hessian, blocked updates, act-order | ✅ matches a naive OBS reference exactly (unit test); ✅ beats RTN g128 (act-order, 3 seeds); ✅ reloaded checkpoint reproduces in-memory perplexity; ✗ within 1% of GPTQModel: ours 2.8% lower on 3-seed means; noise and dtype ruled out, cause unexplained |
| 3 · AWQ | Scale search, clip search, scale folding (GQA-aware) | Within 1% of llm-compressor; folded model output equals unfolded model output before quantization |
| 4 · Triton kernel ✅ | GEMV/split-K and tiled-GEMM W4A16 paths behind `QuantLinear(backend="triton")` | ✅ matches the reference path on all layer shapes, M = 1…255 (GPU tests); ✅ whole-model perplexity through the kernel reproduces the reference (11.728); ✅ faster than fp16 cuBLAS at M = 1 on every layer shape (1.19–2.60×) |
| 5 · Kernel tuning ✅ | Practical ceiling, Nsight Compute profile, then unpack-once tiles, a GEMM path with BLOCK_M = 16, a GEMV kernel for M = 1 and measured per-shape dispatch, each benchmarked separately | ✅ achieved bandwidth at M = 1 is 81–87% of cuBLAS's on the large shapes (3.1–3.3× speedup); ✗ small-shape floor barely moved (k/v_proj 16.4 → 15.4 µs; fixed per-call cost, see above); ✅ dispatch measures paths and unpack variants per shape; ✅ GPU tests and whole-model perplexity through the kernel (11.728) still pass |
| 6 · End to end 🔄 | Swap all Linears; fuse q/k/v and gate/up; decode tok/s (eager, CUDA graph) and GPU time per token, memory; lm-eval tasks; ablations | ✅ decode 1.82× faster with CUDA graphs (155.5 vs 85.3 tok/s), weights 2.47 → 1.03 GB; ✅ perplexity through the fused kernels 10.781 vs 10.780; ✗ eager decode 2.2× slower (launch overhead); ⏳ lm-eval tasks; ⏳ ablations |
| 7 · Write-up | README results, plots, known limitations | A reader can rebuild every table from `results/` with one command |

**Stretch (only if Phases 1–7 are done):** FP8 W8A8 on the L4 (sm_89 supports it); comparison against the
Marlin kernel via vLLM; Llama-3.2-3B repeat. **Next project:** on-device backend reusing the same core
(W8A8 with static activation scales, ExecuTorch + XNNPACK export, latency on an ARM device).

## Repository layout

| Path | What it holds |
| --- | --- |
| `quantkit/quant/` | `pack.py` (int4 format), `rtn.py` (shared quantizer), `qlinear.py` (drop-in Linear), `gptq.py` (Hessian + blocked GPTQ for one matrix); `awq.py` (Phase 3) |
| `quantkit/calib/` | calibration sampling (C4, WikiText-2 train) and layer-0 input capture |
| `quantkit/gptq_model.py` | layer-by-layer, true-sequential GPTQ over a HF decoder |
| `quantkit/kernels/` | `w4a16.py`: split-K and tiled Triton W4A16 kernels, autotuning, `w4a16_matmul` entry point |
| `quantkit/eval/` | perplexity; lm-eval-harness wrapper (Phase 6) |
| `quantkit/model.py` | apply quantization to a HF model; save/load packed checkpoints |
| `quantkit/fuse.py` | runtime fusion of q/k/v and gate/up into single matmuls (fp16 or int4) |
| `quantkit/decode.py` | greedy batch-1 decoding with a static KV cache, eager or replayed from a CUDA graph |
| `quantkit/env.py` | environment capture written into every result |
| `scripts/` | `quantize_eval.py`; `reference_gptqmodel.py` (parity run); `compare_checkpoints.py` (module-by-module diff vs a GPTQ-format checkpoint); `inspect_gptq.py` (per-module loss, Hessian lopsidedness); benchmark and table scripts as phases land |
| `tests/` | unit tests (CPU) and GPU tests (`-m gpu`) |
| `notebooks/` | Colab runner: clone, install, call scripts; no logic in cells |
| `results/` | committed results JSON and plots |

## Setup

```bash
pip install -e '.[dev]'      # add '.[eval]' for lm-eval-harness
python -m pytest -q          # CPU unit tests
```

On Colab: select an **L4** runtime, add `HF_TOKEN` to Colab Secrets (Llama-3.2 is gated; accept the
license on Hugging Face first), and run `notebooks/colab_runner.ipynb`. Checkpoints and results are kept
on Google Drive so a disconnect does not cost a recalibration. The Hugging Face cache stays on local
disk (`/content/hf_cache`): reading a freshly written multi-GB `model.safetensors` back through the Drive
mount corrupted it ("header too large"), and re-downloading the 1B model takes about 10 seconds.

## Reproduce

```bash
python scripts/quantize_eval.py --method fp16
python scripts/quantize_eval.py --method rtn --group-size 128 --save ckpt/rtn-g128
python scripts/quantize_eval.py --method rtn --group-size -1           # per-channel
python scripts/quantize_eval.py --method rtn --bits 3 --fake           # 3-bit ablation
python scripts/quantize_eval.py --load ckpt/rtn-g128                   # reload must match exactly

python scripts/quantize_eval.py --method gptq --save ckpt/gptq-g128     # C4, 128 x 2048, seed 0
python scripts/quantize_eval.py --method gptq --act-order
python scripts/quantize_eval.py --method gptq --bits 3 --fake
python scripts/quantize_eval.py --method gptq --damp 0.05
python scripts/reference_gptqmodel.py                                   # GPTQModel, same calibration windows
```

`--method awq` lands with Phase 3.

```bash
python -m pytest -q -m gpu tests/test_kernel.py                          # kernels on real shapes (GPU)
python scripts/quantize_eval.py --load ckpt/rtn-g128 --backend triton     # perplexity through the kernel
python scripts/bench_kernel.py                                          # kernel vs fp16 cuBLAS, M = 1..256
python scripts/bw_ceiling.py                                            # practical DRAM bandwidth
ncu --profile-from-start off python scripts/profile_kernel.py --shape gate/up_proj --M 1

python scripts/quantize_eval.py --load ckpt/gptq-ao-g128 --backend triton --fuse    # fused fast path, accuracy
python scripts/bench_decode.py --configs fp16 fp16+fuse w4:ckpt/gptq-ao-g128 w4+fuse:ckpt/gptq-ao-g128
```

## Known limitations

- **One GPU.** All speed numbers are from an L4, whose low memory bandwidth makes W4A16 decode gains
  look large. On higher-bandwidth GPUs (A100, H100) the relative gain is expected to shrink; achieved GB/s
  is reported so the result can be reasoned about across hardware.
- **Small models.** 1B and 3B only. Quantization error and the share of time in `lm_head` both change with
  model size.
- **Shared, throttling hardware.** Colab GPUs sit on shared hosts and clocks cannot be locked; results are
  medians of interleaved repeats, and key configs are rerun to check stability.
- **Profiler clocks.** Nsight Compute works on Colab, but by default it locks the L4 to its base clock
  (~795 MHz, against a ~2 GHz boost). ncu is used for percent-of-peak metrics (DRAM throughput, memory vs
  compute bound); wall-clock times come from `do_bench` at normal clocks.
- **Perplexity is protocol-dependent.** Window length, overlap and dataset split change the numbers;
  comparisons to published values are made only against sources with a matching protocol.

## License

MIT.
