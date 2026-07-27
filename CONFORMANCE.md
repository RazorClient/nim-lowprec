# nim-lowprec — conformance & golden-vector sources

**The motto, corrected.** Our discipline is *"differential vs `ml_dtypes`."* That is
exactly right for **element codecs** and it is what we already do (exhaustive +
differential, bit-exact). But `ml_dtypes` has **no concept of a block scale** — it
cannot validate the MX / NVFP4 *schemes*. So the motto must become:

> **`ml_dtypes` for elements · `gguf-py` + `microxcaling` for block schemes ·
> TransformerEngine `NVFP4QuantizerRef` for NVFP4.**

Don't let green element tests create false confidence that MXFP4/MXFP6/MXFP8 are
"done" — only their *elements* are proven today; the *schemes* are not yet
diff-tested end-to-end.

> Synthesized from a mid-2026 survey of the OCP MX v1.0 spec, NVIDIA NVFP4,
> DeepSeek-V3 FP8, llama.cpp/ggml, gpt-oss (MXFP4), and the EXL3/QTIP line.

---

## Where we are

Bit-exact **element** codecs for the whole OCP/MX zoo — bf16, fp16, fp8×4
(e4m3fn/e5m2 + fnuz), MXFP4 (e2m1), MXFP6 (e2m3/e3m2), E8M0, int8/4/1 — all
differentially verified vs `ml_dtypes`. Plus block-scheme *representation*
(`calibrateMX` / `calibrateSymmetric` / `calibrateAsymmetric`) and ggml **Q4_0 / Q8_0**
verified by **round-trip only** (self-consistency, no external oracle). The gaps below
are about *schemes and layouts*, not element math.

## Golden / test-vector sources (ranked by "bit-exact & pip-installable")

### Tier A — Python oracles you diff against exactly (mirror `gen_reference.py`)

| Source | Covers | Path / install | How to generate |
|---|---|---|---|
| **`ml_dtypes`** *(current)* | bf16, fp16, fp8×4, fp6×2, fp4, e8m0 — **elements only** | `pip install ml_dtypes` · jax-ml/ml_dtypes | `np.arange(...).view(dtype).astype(f32)` and back. **Cannot** validate block schemes. |
| **`gguf` (gguf-py)** ⭐ biggest cheap win | **all** ggml formats incl. k-quants: Q4_0, Q8_0, **Q4_K, Q5_K, Q6_K**, Q2_K, Q3_K, Q8_K | `pip install gguf` · module `gguf.quants` (`quantize`/`dequantize`); layout in `gguf/constants.py::GGML_QUANT_SIZES` | `gguf.quants.quantize(arr, GGMLQuantizationType.Q4_K)` → **bit-exact bytes**. Replaces our round-trip self-test with a real external oracle. |
| **`microsoft/microxcaling`** | MXFP4/6/8, MXINT8, E8M0 block scaling — the **scheme**, end to end | repo · `mx/mx_ops.py::_quantize_mx(scale_bits, elem_format, block_size, round)`; tests `mx/tests/test_quantize_mx.py` | `_quantize_mx(x, scale_bits=8, elem_format='fp4_e2m1', block_size=32)` — validates `calibrateMX` end-to-end (nothing does today). |
| **`bitsandbytes`** | NF4 (+ FP4, double-quant), block-64 codebook | repo · `functions.py::quantize_4bit(t, blocksize=64, quant_type='nf4')` | 16-entry NF4 code map is a constant to hardcode + diff. |

### Tier B — reference implementations (port + spot-check; some need HW)

| Source | Covers | Path | Note |
|---|---|---|---|
| **`NVIDIA/TransformerEngine`** ⭐ | **NVFP4** (E2M1 + block-16 + FP8-E4M3 scale + FP32 tensor scale), MXFP8 | `transformer_engine/pytorch/custom_recipes/quantization_ref_nvfp4.py::NVFP4QuantizerRef`; tests `tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py` | The `...Ref` is **pure-PyTorch, CPU-runnable** — generate NVFP4 golden with **no Blackwell HW**. |
| **`ggml-org/llama.cpp`** (C) | canonical ggml layout source of truth | `ggml/src/ggml-quants.c`, `ggml-common.h` (block structs); `tests/test-quantize-fns.cpp` (RMSE thresholds) | Upstream verifies by *threshold*, not golden files → use gguf-py for exactness; use C for block **layout** (nibble order, super-block scale packing). |
| **`turboderp-org/exllamav3`** | EXL3 / QTIP trellis + K-bit unpack | `exllamav3_ext/quant/codebook.cuh` (**MCG `0xCBAC1FEDu`**), `exl3_dq.cuh` | Procedural codebook is fully specified by the MCG → generate golden in Python. Hadamard block=128, 1–8 bpw. |
| **GPTQ / AWQ packing** | int4 weight-only on-disk layouts | ModelCloud/GPTQModel; casper-hansen/AutoAWQ | Layout facts: **8×int4 → one int32**; GPTQ `qweight/qzeros/scales`; AWQ interleaved order. Golden = round-trip a known int4 tensor. |
| **`IST-DASLab/qutlass` + `NVIDIA/cutlass`** | NVFP4/MXFP4 GEMM kernels | cutlass `72_blackwell_narrow_precision_gemm/` | Kernel/perf reference for the FP4 GEMV fast-path (needs Blackwell), not a golden source. |

### Tier C — container samples

**`huggingface/safetensors`** — 8-byte LE header-len + JSON header + raw buffer;
dtype tags `F8_E4M3`/`F8_E5M2`/`BF16`/… (we already map these in `interop.nim`).
Synthesize a sample file for parser tests; real MXFP4 (gpt-oss) / NVFP4 (DeepSeek)
checkpoints on HF Hub for integration.

---

## Gaps vs current conformance

1. **k-quant superblocks entirely absent.** We have Q4_0/Q8_0 (block-32). Real GGUF
   models ship as **Q4_K/Q5_K/Q6_K** (256-elt super-blocks, 6-bit sub-scales). Biggest
   ingestion gap — *Q4_0 is not what's in the wild.*
2. **ggml has no external golden** (round-trip only). `gguf.quants` closes this for free.
3. **MX block scheme never diff-tested end-to-end** vs `microxcaling._quantize_mx`.
4. **NVFP4 absent** — we have every primitive (E2M1 element + E4M3 scale codec), none
   of the scheme (block-16 + two-level scale).
5. **MXINT8 not named** — expressible (int8 + E8M0 + `calibrateMX`) but no path/conformance,
   despite being the 8-bit accuracy/efficiency winner per the 2025 INT-vs-FP study.
6. **GPTQ/AWQ int32 packing** absent (our `packInt4` is ggml nibble order).
7. **NF4 absent** (was in the spec).

## Format landscape (what actually matters, 2026)

- **NVFP4** — FP4 accuracy leader (block-16, FP8-E4M3 scale, FP32 tensor scale;
  ~0.08 vs MXFP4's 0.72 block MSE in NVIDIA's example). Blackwell-native.
- **MXFP4** — the *interoperable* FP4 (OCP standard, on both Blackwell **and** AMD
  MI355X; **shipped in production by gpt-oss**). Accuracy trails NVFP4 unless paired
  with Hadamard rotation + GPTQ. **Support both** — MXFP4 for portability/ingestion,
  NVFP4 for accuracy.
- **FP8 e4m3** — DeepSeek-V3 recipe is the reference: E4M3 everywhere + fine-grained
  scaling (act per 1×128 tile, weight per 128×128 block) + fp32 accumulation promoted
  off the tensor core every 128 elements.
- **MXINT8 > MXFP8** in both accuracy and HW efficiency at block-32 (2025 study) — the
  8-bit format to name if adding one.
- **EXL3/QTIP** — SOTA quality ≤3 bpw but **GPU/tensor-core-bound**; on CPU, GGUF
  i-quants/k-quants win. Keep the K-bit seam for *tattletale alignment*, don't
  sequence it ahead of the GGUF path.
- **Rotation / Hadamard (FWHT)** is now load-bearing for *every* winning ≤4-bit scheme
  (QuaRot → SpinQuant → EXL3). See the boundary note below.

## Roadmap (prioritized)

**P0 — highest value, lowest effort, squarely in-scope**
1. **Wire `gguf.quants` into CI as a second oracle** (`gen_ggml_reference.py`): Q4_0/Q8_0
   become *bit-exact vs ggml*, not round-trip. ~half a day.
2. **Add k-quant Q4_K / Q6_K** (then Q5_K, Q2_K/Q3_K, Q8_K) to `ggml.nim`, diff-tested
   vs `gguf.quants`. This is what makes "ingest a real GGUF model" true.
3. **MX end-to-end conformance** vs `microxcaling._quantize_mx` (MXFP4/6/8 block
   round-trips in `test_mx_conformance.nim`). Proves `calibrateMX`, not just elements.

**P1 — the strategic new format**
4. **NVFP4** as a new `QScheme` (`qNVFP4Block`): block-16, per-block F8E4M3 scale,
   per-tensor FP32 scale, reusing `F4E2M1`. Golden from TransformerEngine
   `NVFP4QuantizerRef` (CPU-runnable). Highest strategic ROI — the frontier format we
   already have every primitive for.
5. **Name + conformance-test MXINT8** (int8 element + E8M0 block-32).

**P2 — ecosystem breadth (when a consumer needs it)**
6. **NF4** codec vs `bitsandbytes.quantize_4bit`.
7. **GPTQ/AWQ int32 packing** helpers (`gptq_pack.nim`), golden = round-trip vs GPTQModel/AutoAWQ.
8. **EXL3 K-bit unpacker + procedural MCG codebook** (`0xCBAC1FED`) — the tattletale
   on-ramp; do before a full trellis decoder since the codebook is procedural.

## Boundary reconsideration — FWHT

Our spec puts the Hadamard transform OUT of the substrate. But a bare, verified
**Fast Walsh–Hadamard Transform** is a numeric primitive like `dot`/`convert`
(zero-dep, pure bits→bits) and it is the enabling op for the entire 2026 ≤4-bit quant
frontier — and it *fuses into the dequant-GEMV path*. At minimum, don't let the L1/L2
line block it: it belongs wherever the fused dequant-GEMV lives.

## Sources

OCP MX v1.0 (opencompute.org) · NVFP4 (developer.nvidia.com/blog + arXiv 2509.25149) ·
DeepSeek-V3 (arXiv 2412.19437) · INT-vs-FP (arXiv 2510.25602) · gpt-oss MXFP4 (arXiv
2508.16700) · QTIP (arXiv 2406.11235) · repos: jax-ml/ml_dtypes, microsoft/microxcaling,
NVIDIA/TransformerEngine, ggml-org/llama.cpp, turboderp-org/exllamav3, bitsandbytes,
IST-DASLab/qutlass, huggingface/safetensors, casper-hansen/AutoAWQ.
