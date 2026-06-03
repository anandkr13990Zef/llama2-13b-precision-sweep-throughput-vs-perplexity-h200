# Low-Precision Inference on LLaMA-2-13B: Throughput and Representational Loss across Data Types (2×H200)

A first-principles study of how numerical precision affects a real LLaMA-2-13B,
measured on a 2×H200 node. Two axes are characterized: **kernel throughput**
(real GEMM speed per data type) and **representational loss** (perplexity
degradation from quantizing the weights). All quantizers are implemented by hand;
all throughput numbers come from real H200 tensor-core kernels in core PyTorch.

The intent is reproduction and understanding, not novelty: the bits-vs-quality
behaviour here is well established in the quantization literature. The specific
contribution is a clean, honest characterization on this exact hardware and
software stack, including which low-precision kernels actually deliver a speedup
and which do not.

---

## Environment

| Item | Value |
|---|---|
| GPUs | 2 × NVIDIA H200, 143771 MiB each, driver 580.126.20 |
| Interconnect | NV18 (18 bonded NVLinks) |
| Compute capability | sm_90 (Hopper) |
| PyTorch | 2.11.0+cu130 |
| CUDA / NCCL | 13.0 / 2.28.9 |
| Kernels present | `float8_e4m3fn`, `_scaled_mm`, `_int_mm`, `_weight_int4pack_mm` (all available) |
| transformers / accelerate / datasets | 5.7.0 / installed / installed |
| Model | `TheBloke/Llama-2-13B-fp16` (fp16 checkpoint, `.bin` shards) |

The throughput microbench needs only one GPU (per-GPU compute property). The loss
sweep loads the model across both cards via `device_map="auto"`; perplexity is
invariant to the sharding scheme, so the second card is used for capacity and
speed, not because parallelism affects the result.

---

## Repository layout

```
.
├── README.md
├── throughput.py        # prefill GEMM throughput per dtype (real kernels)
├── decode.py            # M=1..8 decode latency, bf16 vs int4 weight-only
├── dtype_loss.py        # weight-only precision sweep, perplexity on wikitext-2
├── accuracy.py          # 0-shot ARC-Easy logprob accuracy per dtype  [NOT YET RUN]
└── results/
    ├── setup.txt
    ├── throughput.csv
    ├── decode_int4.csv
    └── dtype_loss.csv
```

---

## Part A — Throughput

### Method

Each measurement is a single GEMM at a representative LLaMA-2-13B linear shape,
timed with CUDA events (10 warm-up, median of 30). Throughput is reported as
`2·M·N·K / time`. Shapes, given as (K, N):

- `5120 × 5120` — attention projections (q/k/v/o)
- `5120 × 13824` — MLP gate/up
- `13824 × 5120` — MLP down

M is the token count (rows of the activation). Kernels: bf16 `torch.matmul`,
fp8 `torch._scaled_mm` (e4m3, both operands quantized, `use_fast_accum=True`),
int8 `torch._int_mm` (i8×i8→i32, requires M>16 and a column-major second operand),
int4 `torch._weight_int4pack_mm` (tinygemm, weight-only int4 with bf16 activations,
nibble-packed `[N, K/2]`, `innerKTiles=8`).

### Prefill GEMM results (TFLOP/s, and × vs bf16)

**5120 × 5120**

| M | bf16 | fp8 | int8 | int4 |
|---|---|---|---|---|
| 16 | 30.7 (1.00) | 32.1 (1.05) | — (M>16 req.) | 18.4 (0.60) |
| 64 | 119.3 (1.00) | 126.4 (1.06) | 141.7 (1.19) | 26.2 (0.22) |
| 256 | 373.3 (1.00) | 453.9 (1.22) | 311.6 (0.83) | 28.5 (0.08) |
| 1024 | 665.2 (1.00) | 986.3 (1.48) | 616.9 (0.93) | 29.4 (0.04) |
| 4096 | 768.4 (1.00) | 1334.2 (1.74) | 869.0 (1.13) | 29.1 (0.04) |

**5120 × 13824**

| M | bf16 | fp8 | int8 | int4 |
|---|---|---|---|---|
| 16 | 46.2 (1.00) | 63.4 (1.37) | — | 24.4 (0.53) |
| 64 | 183.7 (1.00) | 241.3 (1.31) | 233.8 (1.27) | 27.9 (0.15) |
| 256 | 551.5 (1.00) | 839.5 (1.52) | 490.2 (0.89) | 29.1 (0.05) |
| 1024 | 671.6 (1.00) | 1265.1 (1.88) | 716.3 (1.07) | 29.3 (0.04) |
| 4096 | 781.4 (1.00) | 1326.3 (1.70) | 723.8 (0.93) | 29.4 (0.04) |

**13824 × 5120**

| M | bf16 | fp8 | int8 | int4 |
|---|---|---|---|---|
| 16 | 43.4 (1.00) | 58.7 (1.35) | — | 21.5 (0.49) |
| 64 | 169.3 (1.00) | 244.1 (1.44) | 215.7 (1.27) | 27.6 (0.16) |
| 256 | 502.9 (1.00) | 757.5 (1.51) | 488.2 (0.97) | 29.1 (0.06) |
| 1024 | 762.6 (1.00) | 1313.2 (1.72) | 709.0 (0.93) | 29.2 (0.04) |
| 4096 | 800.0 (1.00) | 1338.9 (1.67) | 777.3 (0.97) | 29.6 (0.04) |

### Decode results (M=1..8, bf16 vs int4 weight-only, latency speedup)

int4 tinygemm is a GEMV-style kernel; its throughput in TFLOP/s is flat (~29) and
its time scales with M, so it is meaningful only near M=1. Measured by latency:

| M | 5120×5120 | 5120×13824 | 13824×5120 |
|---|---|---|---|
| 1 | 1.59× | 1.77× | 1.65× |
| 2 | 1.59× | 1.65× | 1.56× |
| 4 | 1.23× | 1.29× | 1.06× |
| 8 | 0.86× | 0.88× | 0.69× |

### Throughput findings

- **fp8 is the only data type with a real batched-GEMM speedup on this stack.** It
  scales to ~1.3 PFLOP/s and peaks at **1.88× bf16** around M=1024. The benefit is
  batch-dependent: negligible at M≤64 (launch/memory-bound), opening up from M≥256.
- **int8 via `torch._int_mm` lands at roughly bf16 parity** (0.83–1.27×) and does
  **not** realize INT8's theoretical 2× on this build. It is fastest at moderate M
  (~1.2× at M=64) and falls back to ~bf16 at large M. It also requires M>16.
- **int4 (`_weight_int4pack_mm`) is decode-only.** At M=1 it gives **1.6–1.8×**
  lower latency than bf16 (4× less weight traffic), with the advantage gone by M≈4–8.
  As a batched GEMM it is far slower than bf16 and should not be used there.
- Practical split: **fp8 for prefill/batched, int4 for single-stream decode, int8
  offers no throughput reason to use it here.**

---

## Part B — Representational loss

### Method

Weight-only fake quantization: compute dtype is held at fp16; each Linear weight is
quantized → dequantized to the target format, and perplexity is measured on
wikitext-2 (test, first 4096 tokens, single forward). This isolates the
representational loss of each format. 281 Linear layers are quantized per dtype.

The reference is fp16 — the checkpoint's storage precision. No information exists
below fp16, so fp32/fp64 cannot beat it (and don't). The meaningful direction is
downward from fp16.

Quantizers: signed n-bit RTN (int8 per-channel; int4/int2 per-group 128); fp8 via
the real e4m3 grid with per-channel scaling; fp6 (e3m2) emulated with per-channel
scaling and software float-grid rounding.

### Data-type support on this stack

| dtype | bits | status |
|---|---|---|
| fp128 | 128 | not available (no GPU/torch support) — skipped |
| fp64 | 64 | native |
| fp32 | 32 | native |
| fp16 | 16 | native — reference |
| bf16 | 16 | native |
| fp8 (e4m3) | 8 | native on Hopper |
| fp6 (e3m2) | 6 | emulated (no Hopper hardware) |
| int8 | 8 | RTN, per-channel |
| int4 | 4 | RTN, per-group 128 |
| int2 | 2 | RTN, per-group 128 |

### Results (wikitext-2, reference fp16 perplexity = 4.504)

| dtype | bits | perplexity | ratio vs fp16 |
|---|---|---|---|
| fp64 | 64 | 4.504 | 1.0000 |
| fp32 | 32 | 4.504 | 1.0000 |
| fp16 | 16 | 4.504 | 1.0000 |
| bf16 | 16 | 4.504 | 1.0000 |
| int8 | 8 | 4.506 | 1.0004 |
| fp8_e4m3 | 8 | 4.523 | 1.0042 |
| fp6_e3m2 | 6 | 4.553 | 1.0108 |
| int4 | 4 | 4.708 | 1.0453 |
| int2 | 2 | 119345.5 | 26497.5 |

### Loss findings

- **The fp16 information bound holds exactly.** fp64/fp32/fp16/bf16 give identical
  perplexity — wider-than-checkpoint precision cannot recover information that the
  fp16 weights never stored.
- **8-bit is near-lossless, and int8 slightly beats fp8** (1.0004× vs 1.0042×).
  For weight-only quantization, int8's uniform mantissa preserves more precision
  than e4m3, which spends bits on an exponent range the weight distribution doesn't
  need.
- **fp6 (e3m2) costs ~1%** once scaled correctly, sitting between 8-bit and int4.
  Without per-channel scaling it collapses (the format's limited dynamic range
  underflows small weights to zero) — that was an implementation bug, since fixed.
- **int4 is the knee: +4.5% perplexity** with per-group-128 RTN — still small, which
  illustrates that perplexity is a forgiving metric for low-bit damage.
- **int2 RTN collapses** (perplexity ~10⁵, effectively random). Naive
  round-to-nearest at 2 bits destroys the model; usable 2-bit requires error-aware
  methods (GPTQ) or codebook methods (QuIP#, AQLM), not better rounding alone.

---

## What this is, and what it is not

- It is a clean, hand-implemented characterization of precision effects on a real
  13B model and on this exact H200 software stack.
- It is **not** a novel result. The bits-vs-perplexity shape and the
  fp8/int8/int4 throughput behaviour are consistent with prior work.
- Throughput and loss were measured **separately**: throughput from isolated GEMM
  kernels, loss from full-model weight-only quantization with fp16 compute. They are
  not an end-to-end measurement of a model running natively in each dtype.

---

## Limitations

- **Weight-only quantization, fp16 compute.** Activations are not quantized, so the
  loss numbers reflect weight representation only. True end-to-end low-precision
  inference (quantized activations + in-dtype GEMM) would differ.
- **Throughput is a microbench** of single GEMM shapes, not a full forward pass; it
  excludes attention, norm, sampling, and kernel-launch overhead at the model level.
- **int8 `_int_mm`** underperforms its theoretical peak on this build; a
  cuBLASLt-tuned path (e.g. via a dedicated library) could do better.
- **fp6 and int2 have no real kernel here**, so they appear on the loss axis only;
  fp128 is unavailable entirely.
- **Perplexity window** is 4096 tokens of wikitext-2 test; the absolute value
  depends on window and method, so treat ratios, not the absolute 4.504, as the
  signal.
- RTN only. No error-aware or codebook quantization was applied (see below).

---

## Status / not yet run

The instance ended before two planned components executed:

- **Downstream accuracy** (`accuracy.py`): 0-shot ARC-Easy via length-normalized
  logprob, per dtype. Intended to test whether int4's small perplexity bump (+4.5%)
  understates task-level damage. Not run.
- **RTN vs GPTQ at 4/3/2-bit**: quantize with a calibration-based method (`gptqmodel`)
  and compare to RTN, especially at 2-bit where RTN collapses. Not run. (`gptqmodel`
  was not installed at probe time.)

These are the natural next steps to turn the loss axis into a method comparison and
to produce a combined loss-vs-throughput Pareto view.

---

## Reproduction

```bash
# environment
pip install -q transformers accelerate datasets "huggingface_hub[cli]"
hf auth login                       # if the checkpoint requires it
hf download TheBloke/Llama-2-13B-fp16 --local-dir ./llama2-13b-fp16

mkdir -p results

# Part A: throughput (single GPU)
python throughput.py                # -> results/throughput.csv
python decode.py                    # -> results/decode_int4.csv

# Part B: representational loss (2 GPUs via device_map="auto")
python dtype_loss.py --model ./llama2-13b-fp16 --out results/dtype_loss.csv
```

Stack: 2×H200, PyTorch 2.11.0+cu130, CUDA 13.0, NCCL 2.28.9, transformers 5.7.0.
