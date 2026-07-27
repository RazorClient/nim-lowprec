# nim-lowprec

[![CI](https://github.com/RazorClient/nim-lowprec/actions/workflows/ci.yml/badge.svg)](https://github.com/RazorClient/nim-lowprec/actions/workflows/ci.yml)
[![SIMD (x86)](https://github.com/RazorClient/nim-lowprec/actions/workflows/simd-x86.yml/badge.svg)](https://github.com/RazorClient/nim-lowprec/actions/workflows/simd-x86.yml)

A **zero-dependency low-precision numeric substrate** for ML inference in Nim —
one home for the low-precision family (bf16 → MXFP4 → int1), sharing one
conformance-test harness and one version.

It is deliberately a **substrate**, not a framework: dtypes, conversions,
quantization, and hand-written SIMD kernels over flat buffers. The `Tensor` /
GEMM layer lives one level up in [nim-tensor](https://github.com/RazorClient/nim-tensor),
so this stays the clean thing everything else (including other tensor libraries)
can depend on.

## Formats

| family | type(s) | notes |
|---|---|---|
| bfloat16 | `BF16` | top 16 bits of an fp32 (1/8/7) |
| IEEE half | `F16` | binary16 (1/5/10), real subnormals |
| fp8 (OCP) | `F8E4M3`, `F8E5M2` | e4m3fn / e5m2 |
| fp8 (AMD) | `F8E4M3FNUZ`, `F8E5M2FNUZ` | fnuz: bias +1, no Inf/−0, `0x80` NaN |
| **MXFP4** | `F4E2M1` | OCP microscaling 4-bit float (e2m1) |
| **MXFP6** | `F6E2M3`, `F6E3M2` | OCP microscaling 6-bit floats |
| **MX scale** | `E8M0` | shared power-of-two block scale (`2^(b−127)`) |
| integers | `I8`, `I4`, `I1` | + nibble / bit packing (ggml low-nibble-first) |

Every float↔float32 conversion is verified **bit-exact** against `ml_dtypes` /
`numpy` — exhaustive where the code space allows (all 2¹⁶ for bf16/fp16, all 2⁸
for each fp8, every code for the MX formats), differential over millions of
samples otherwise.

On top of the storage types:

- a `LowPrec` **concept** every dtype satisfies (`decode → float32`, `storageBits`);
- generic fp32-accumulate **BLAS-1** (`sum` / `dot` / `nrm2` / `axpy` / …);
- **quantization** — affine `quantize`/`dequantize` with per-tensor, per-group,
  **MX-block** (E8M0 shared scale), and **asymmetric** calibration;
- **SIMD kernels** (NEON + AVX2/FMA/F16C) — batch bf16↔f32 / f16↔f32 convert, and
  **fused dequant-GEMV** for int8, packed int4, and packed **MXFP4** weights (the
  quantized-decode hot path, weights never materialized as fp32).

## Install

Not yet on the Nimble registry — install from git (pin the release tag):

```sh
nimble install https://github.com/RazorClient/nim-lowprec@#v0.1.0
```

## Quickstart

```nim
import nim_lowprec

# bf16: store narrow, widen exactly
let a = toBF16(3.14159'f32)
echo a.toFloat32                 # 3.140625

# BLAS-1 is generic over every dtype (accumulates in fp32)
let xs = @[toBF16(1.0'f32), toBF16(2.0'f32), toBF16(3.0'f32)]
echo dot(xs, xs)                 # 14.0

# quantize fp32 weights -> int8 with per-group scales, and back
let p = calibrateSymmetric(w, groupSize = 32, qmax = 127.0'f32)
var q = newSeq[I8](w.len)
quantize(w, p, q)
```

A fuller runnable example (bf16, BLAS-1, and an MXFP4 fused GEMV) is in
[`examples/quickstart.nim`](examples/quickstart.nim) — `nimble example`.

## Develop

```sh
nimble refs      # generate golden reference vectors (needs python3 + numpy + ml_dtypes)
nimble test      # correctness suite (differential legs skip without the vectors)
nimble simd      # NEON SIMD tests — run locally, on arm64
nimble example   # the quickstart
nimble bench     # microbenchmarks (release mode) — see Benchmarks, below
```

## Benchmarks

Each benchmark times a **scalar reference against the kernel the build selected**,
then compares their checksums and exits non-zero on a mismatch — a speedup can't
come from a path that quietly stopped computing the right answer. Elementwise
kernels must agree bit-for-bit; GEMV and dot are reductions, so those are checked
within a relative tolerance.

```sh
nimble bench        # native SIMD kernels
nimble benchScalar  # the same benchmarks with -d:lpSimd=scalar
nimble benchAll     # both, back to back
```

Apple M1 Pro, Nim 2.2.6, clang 21, `-d:release`, best of 3:

| kernel | N / shape | scalar | NEON | speedup |
|---|---|---|---|---|
| f32 → bf16 | 16M | 905 Melem/s | 3823 Melem/s | 4.2x |
| bf16 → f32 | 16M | 1056 Melem/s | 8163 Melem/s | 7.7x |
| f32 → fp16 | 16M | 427 Melem/s | 4965 Melem/s | 11.6x |
| fp16 → f32 | 16M | 799 Melem/s | 5102 Melem/s | 6.4x |
| fp8 e5m2 → f32 | 16M | 144 Melem/s | 8408 Melem/s | 58x |
| fp8 e4m3 → f32 | 16M | 144 Melem/s | 4264 Melem/s | 29x |
| int8 dequant | 16M, group 128 | 399 Melem/s | 3989 Melem/s | 10.0x |
| bf16 dot | 16M | 2.1 GFLOP/s | 6.4 GFLOP/s | 3.0x |
| int8 dequant-GEMV | 4096² | 2.3 GFLOP/s | 9.2 GFLOP/s | 4.0x |
| int4 dequant-GEMV (packed) | 4096² | 1.7 GFLOP/s | 9.0 GFLOP/s | 5.3x |
| mxfp4 dequant-GEMV (packed) | 4096² | 0.3 GFLOP/s | 9.1 GFLOP/s | 30x |
| ggml Q8_0 GEMV | 4096² | 2.7 GFLOP/s | 10.2 GFLOP/s | 3.8x |
| ggml Q4_0 GEMV | 4096² | 3.4 GFLOP/s | 10.1 GFLOP/s | 3.0x |

Reading the numbers honestly:

- **The scalar references are `proc`s over `openArray` params**, shaped like the
  kernels' own non-SIMD fallback. An inline loop over module-level `seq`s can't be
  auto-vectorized by the C compiler (the globals may alias), which inflated these
  speedups by up to 3x before the references were fixed.
- **The fp8 and mxfp4 rows are the widest for a reason that is only half SIMD:**
  their scalar decode computes magnitudes in float64 via the shared tinyfloat core,
  while the kernels use a hardware convert / byte-LUT. That is a real win for a
  caller today, but it isn't a pure lane-width win.
- **`nimble benchScalar` separates the two.** Rebuilt with `-d:lpSimd=scalar`, the
  mxfp4 GEMV kernel is still 6.2x its reference — that part is the byte-LUT beating
  the float64 decode, so of the 30x native figure roughly 4.8x is the vectorization.
  The bf16 conversions split the same way (1.3–2.0x of their 4.2x/7.7x is codegen,
  not NEON). The fp8 kernels, by contrast, land at 1.00x in the scalar build: their
  entire advantage lives in the SIMD path.
- **The GEMVs sit near 9–10 GFLOP/s on one core regardless of weight width** — the
  fused kernels are already limited by fp32 FMA throughput, not by the decode, so
  going int8 → int4 → fp4 buys memory, not speed.
- **Fusion is worth about as much as vectorization here:** in `bench_ggml`, the
  unfused "dequantize a row to fp32, then dot it" pipeline is *slower* than the
  fused scalar loop (0.4–0.5x), and ~3x slower than the fused NEON kernel.
- **The asymmetric-quant group is a deliberate ~1.00x in *both* builds:**
  `dequantizeBatch` has no zero-point fast path and falls through to the exact
  scalar routine. A speedup appearing there would mean the fallback stopped being
  taken.

**Verification:** the correctness suite runs in CI on **x86_64 (Linux)** and
**arm64 (macOS)**, regenerating the reference vectors per-arch so the differential
tests enforce bit-exactness on both. **x86 SIMD** (AVX2/FMA/F16C) is verified in
its own CI workflow, diffed bit-for-bit against the scalar reference; **NEON** is
verified locally (`nimble simd`) on arm64 — each ISA tested where its hardware
actually exists.

## License

MIT — see [LICENSE](LICENSE).
