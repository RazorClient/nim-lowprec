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
| **NVFP4** | `quantizeNVFP4` | block-16 + FP8-E4M3 scale + fp32 tensor scale, bit-exact vs TransformerEngine's reference |
| **MXFP6** | `F6E2M3`, `F6E3M2` | OCP microscaling 6-bit floats |
| **MX scale** | `E8M0` | shared power-of-two block scale (`2^(b−127)`) |
| integers | `I8`, `I4`, `I1` | + nibble / bit packing (ggml low-nibble-first); **MXINT8** = `calibrateMX` + `I8` |
| ggml blocks | `BlockQ8_0/Q4_0/Q4_K/Q6_K` | quantize (Q*_0) and dequantize, bit-exact vs **gguf-py** |

Every float↔float32 conversion is verified **bit-exact** against `ml_dtypes` /
`numpy` — exhaustive where the code space allows (all 2¹⁶ for bf16/fp16, all 2⁸
for each fp8, every code for the MX formats), differential over millions of
samples otherwise. The block **schemes** are diff-tested end-to-end too: ggml
formats against **gguf-py**, MX block scaling against an independent OCP v1.0
implementation, NVFP4 against TransformerEngine's reference (see CONFORMANCE.md).

On top of the storage types:

- a `LowPrec` **concept** every dtype satisfies (`decode → float32`, `storageBits`);
- generic fp32-accumulate **BLAS-1** (`sum` / `dot` / `nrm2` / `axpy` / …);
- **quantization** — affine `quantize`/`dequantize` with per-tensor, per-group,
  **MX-block** (E8M0 shared scale), and **asymmetric** calibration;
- **SIMD kernels** (NEON + AVX2/FMA/F16C) — batch bf16↔f32 / f16↔f32 / fp8↔f32
  convert (all four fp8 formats, both directions), SIMD **quantize** (Q8_0/Q4_0/
  per-group int8), and **fused dequant-GEMV** for int8, packed int4, packed
  **MXFP4**, and **ggml Q8_0/Q4_0** weights;
- **int8-activation kernels** — SDOT on arm64, maddubs/madd on AVX2 — GEMV and
  multi-column **GEMM** (`dequantGemvQ8`, `dequantGemvI4Q8`, `dequantGemmQ8`, …),
  int32-exact accumulation, ~3x the fp32-activation path, all pinned bit-for-bit
  against integer references;
- optional **row-sharded** forms (`nim_lowprec/parallel`) — per-call threads or a
  caller-owned persistent `LpPool` — bit-identical to the serial kernels.

## Install

Not yet on the Nimble registry — install from git (pin the release tag):

```sh
nimble install https://github.com/RazorClient/nim-lowprec@#v1.0.0
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

A fuller runnable example (bf16, BLAS-1, an MXFP4 fused GEMV, the int8-SDOT
GEMV/GEMM, and the thread pool) is in
[`examples/quickstart.nim`](examples/quickstart.nim) — `nimble example`.

## Develop

```sh
nimble refs      # generate golden reference vectors (needs python3 + numpy + ml_dtypes)
nimble test      # correctness suite (differential legs skip without the vectors)
nimble simd      # NEON SIMD tests (CI runs these on the arm64 macOS runners too)
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
| f32 → bf16 | 16M | 912 Melem/s | 4529 Melem/s | 5.0x |
| bf16 → f32 | 16M | 1072 Melem/s | 9981 Melem/s | 9.3x |
| f32 → fp16 | 16M | 417 Melem/s | 10981 Melem/s | 26x |
| fp16 → f32 | 16M | 795 Melem/s | 10020 Melem/s | 12.6x |
| f32 → fp8 e4m3 (encode) | 16M | 15 Melem/s | 788 Melem/s | 53x |
| fp8 e5m2 → f32 | 16M | 150 Melem/s | 8438 Melem/s | 56x |
| fp8 e4m3 → f32 | 16M | 149 Melem/s | 6374 Melem/s | 43x |
| int8 dequant | 16M, group 128 | 398 Melem/s | 10639 Melem/s | 27x |
| quantize → Q8_0 (encode) | 16M | 427 Melem/s | 4622 Melem/s | 10.8x |
| ggml Q8_0 block dequant | 16.8M | 1513 Melem/s | 6121 Melem/s | 4.1x |
| ggml Q4_0 block dequant | 16.8M | 1566 Melem/s | 6009 Melem/s | 3.8x |
| bf16 dot | 16M | 2.1 GFLOP/s | 12.9 GFLOP/s | 6.1x |
| int8 dequant-GEMV | 4096² | 2.3 GFLOP/s | 26.4 GFLOP/s | 11.3x |
| int4 dequant-GEMV (packed) | 4096² | 1.6 GFLOP/s | 20.5 GFLOP/s | 12.5x |
| mxfp4 dequant-GEMV (packed) | 4096² | 0.3 GFLOP/s | 21.2 GFLOP/s | 70x |
| ggml Q8_0 GEMV | 4096² | 2.7 GFLOP/s | 12.8 GFLOP/s | 4.8x |
| ggml Q4_0 GEMV | 4096² | 3.3 GFLOP/s | 13.5 GFLOP/s | 4.1x |

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
  mxfp4 GEMV kernel is still ~6x its reference — that part is the byte-LUT beating
  the float64 decode, so most of the native figure is decode, not vectorization.
  The bf16 conversions split the same way (a sizeable slice of their speedup is
  codegen, not NEON). The fp8 kernels, by contrast, land at 1.00x in the scalar build: their
  entire advantage lives in the SIMD path.
- **The fp32-activation GEMVs cluster at 20–26 GFLOP/s regardless of weight width**
  — the widen-to-fp32 FMA pipeline is the limit, not the decode, so going
  int8 → int4 → fp4 buys memory, not speed. Escaping that pipeline entirely is what
  the int8-activation path below is for.
- **Fusion is worth about as much as vectorization here:** in `bench_ggml`, the
  unfused "dequantize a row to fp32, then dot it" pipeline is *slower* than the
  fused scalar loop (0.4–0.5x), and ~3x slower than the fused NEON kernel.
- **The asymmetric-quant group is a deliberate ~1.00x in *both* builds:**
  `dequantizeBatch` has no zero-point fast path and falls through to the exact
  scalar routine. A speedup appearing there would mean the fallback stopped being
  taken.

### Multiple cores

The kernels are single-threaded on purpose — the same layering ggml uses, where
`vec_dot_*` is serial and `ggml_graph_compute` shards rows one layer up. A
substrate that spawns its own threads can't see that its caller already has a pool,
and the two oversubscribe.

Row sharding is therefore opt-in, in [`nim_lowprec/parallel`](src/nim_lowprec/parallel.nim):
`parDequantGemv`, `parDequantGemvI4/F4`, `parDequantGemvQ8_0/Q4_0`. They take a
`threads` count, wrap the *same* serial kernels over row ranges, and are
**bit-identical** to the serial call — sharding changes which thread visits a row,
never the arithmetic in it, so `tests/test_parallel.nim` demands exact equality
(and `bench_parallel` runs its checksum check at `tol = 0`).

`nimble bench` → `bench_parallel`, int8 4096², 8 P-cores + 2 E-cores:

| threads | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| GFLOP/s | 26.2 | 44.7 | 77.4 | 101.2 | **111.9** |
| speedup | 1.00x | 1.71x | 2.95x | 3.86x | 4.27x |

(Scaling flattened relative to earlier revisions because the serial kernel got
2.5x faster — the fixed per-call thread-creation cost didn't. The pooled forms in
the same module avoid that cost; the int8-activation kernels reach 192–564 GFLOP/s
pooled, see below.)

Threads are created per call (stdlib `createThread` — no pool, no dependency), so
below `minRowsPerThread` rows per shard these degrade to the serial kernel rather
than pay for spawning. A caller wanting the last slice of performance should keep a
persistent pool and call the serial kernels on row ranges itself; this module
deliberately owns no long-lived threads.

### The two findings that moved everything at once

**Nim's `-d:release` keeps bounds and overflow checks ON** (only `-d:danger` removes
them), and in these inner loops that was not a safety net but the dominant cost:
the L1-resident SDOT GEMV measured 32 GFLOP/s with checks and 138 without — a 4.3x
tax, bigger than every architectural improvement combined. The kernel modules now
compile their loops with `{.push boundChecks: off, overflowChecks: off.}` after
asserting shape preconditions at entry, so callers get full speed in ANY build
mode; the bit-exactness suites compile without `-d:release` and therefore pin
exactly this checks-off code. (The scalar reference rows above are ordinary
`-d:release` Nim with its default checks — deliberately, since that is what naive
caller code gets.)

**Apple clang defaults to `-ffp-contract=fast`**, which silently fuses a multiply
feeding an add into one FMA across statement boundaries — one rounding instead of
two, which broke bit-exactness by one ulp in one row out of five. The kernel and
test modules pin `-ffp-contract=off`; every FMA in the library is now an explicit
intrinsic, never a compiler improvisation.

Beyond those, the disassembly-driven kernel work found: multiple independent
accumulators (FMA/SDOT latency vs throughput), pointer-walking instead of indexed
addressing (clang otherwise recomputes base+offset per load and refuses to
unroll), a row-at-a-time streaming shape for weights past `-d:lpStreamBytes`
(default 12 MB — four parallel weight streams prefetch at ~28 GB/s, one sequential
stream at ~44), and NOT unrolling the 64-column loop further, because at the
dominant groupSize of 128 a wider body executes once per group and its machinery
becomes per-group overhead.

### The int8-activation (SDOT) path

`dequantGemv` and friends take **fp32** activations, so each int8 weight is widened
to fp32 and fed to an FP FMA. `dequantGemvQ8` / `dequantGemvI4Q8` instead take
activations **already quantized to int8** and use `vdotq_s32` (ARMv8.2
FEAT_DotProd, gated by `-d:lpDotProd=auto|on|off`): 16 multiply-accumulates per
instruction, exact int32 accumulation, one fp multiply per group. Produce the
activations with the existing `calibrateSymmetric(x, groupSize, 127.0)` +
`quantize`; that cost is O(K) against the GEMV's O(M·K) and is included in every
number below.

This is a **separate entry point on purpose**: quantizing activations loses
precision and that is the caller's call — the fp32-activation kernels are
untouched. The *sum* itself is more accurate than the fp32 path, not less: int32
group accumulation is exact (g·127·127 caps a group of g ≤ ~132k), so
`tests/test_sdot.nim` pins every kernel **bit-for-bit** against an int64
reference, and the `-d:lpDotProd=off` fallback produces identical bits (it exists
for portability, not speed — on targets without SDOT, x86 included, prefer
`dequantGemv`).

4096², M1 Pro, one core, activation quantization included:

| kernel | GFLOP/s | vs fp32-activation path |
|---|---|---|
| `dequantGemv` — int8 weights, fp32 activations | 26.4 | 1.00x |
| `dequantGemvQ8` — int8 weights, int8 activations | 79.7 | **3.0x** |
| `dequantGemvI4Q8` — **4-bit** weights, int8 activations | 65–100 (shape-dependent) | **2.5–3.8x** |

Weights larger than `-d:lpStreamBytes` (default 12 MB) switch from the 4-row tile
to a row-at-a-time streaming loop — one sequential weight stream prefetches at
~44 GB/s where the tile's four parallel streams manage ~28. Both shapes are
bit-identical; the suite runs twice (`-d:lpStreamBytes=0`) to pin both.

Two siblings round the family out. The **ggml-block-layout** forms
(`dequantGemvQ8_0Q8` / `dequantGemvQ4_0Q8`, activations via `quantizeQ8_0`) are
format-identical to llama.cpp's `vec_dot`, so real GGUF weights run without
repacking — 79/68 GFLOP/s cache-resident, slower than the group-128 kernels when
streaming because the inline per-32 scale means 4x more finalize work. And the
**multi-column GEMM** (`dequantGemmQ8` / `dequantGemmI4Q8`) handles n>1
activation columns by iterating columns *inside* the group loop, so each weight
group is read from DRAM once per row and its n reuses hit L1: single-core it ties
n separate GEMVs (one core can't outrun DRAM either way), but on 8 threads it
reaches **564 GFLOP/s** at n=8 int8 against ~190 for the GEMV path. Every GEMM
column is bit-identical to the corresponding GEMV.

Roofline, measured on this machine: one core sustains 54.9 GB/s of streaming
reads; all cores saturate at ~115 GB/s. The single-column int8 kernel moves
~44 GB/s of weights on one core — close to the single-stream wall, which is
exactly why the multi-column GEMM exists.

### Against other libraries

Same machine, same sizes, all single-core (BLAS pinned to one thread, llama.cpp
patched to `N_THREADS 1` and verified `user ≈ real`). Not part of `nimble bench` —
these need numpy/`ml_dtypes` and a llama.cpp build:

| | nim-lowprec | numpy 2.5 + ml_dtypes | ggml / llama.cpp |
|---|---|---|---|
| bf16 → f32 | 9981 Melem/s | 10114 | 9262 |
| fp16 → f32 | 10020 Melem/s | 10152 | 3648 |
| f32 → fp16 | 10981 Melem/s | 12406 | 2562 |
| fp8 e5m2 → f32 | **8438 Melem/s** | 1067 | — |
| f32 → fp8 (encode) | **788–802 Melem/s** | 798 | — |
| quantize → Q8_0 | **4622 Melem/s** | — | 4622 |
| quantize → Q4_0 | **1047 Melem/s** | — | 995 |
| Q8_0 block dequant | 6121 Melem/s | — | **11248** |
| Q4_0 block dequant | **6009 Melem/s** | — | 6380 |
| bf16 dot | **12.9 GFLOP/s** | 5.6 (widen ×2 + sdot) | 1.9 (no NEON path) |
| fp32 matvec | — | 16.0 (Accelerate sgemv) | — |

Quantized matvec, format-matched, at llama.cpp's own test shape
(`m=4096, k=14336`) — CPU only, one core and then all cores:

| weights | nim-lowprec | llama.cpp (same minute) | verdict |
|---|---|---|---|
| int8, fp32 activations (the old path) | 26.1 | — | — |
| **int8, int8 activations** | **85.9** | 73.8 (q8_0) | **1.16x ahead** |
| **4-bit**, int8 activations | 64.8 | 70.6 (q4_0) | 92% |
| int8, 8 threads (pooled) | 192.1 | 149.6 (q8_0, 10 thr) | **1.28x ahead** |
| **4-bit, 8 threads (pooled)** | **322.3** | 193–294 (q4_0, 10 thr) | **above their whole band** |
| **int8 GEMM, n=8, 8 threads** | **564.0** | ~100 (q4_0 n=8, 10 thr) | **~5.6x ahead** |

Their all-core figure is genuinely unstable — nine samples of the same q4_0 case gave
193–294 GFLOPS, a 1.5x spread from 10 threads landing across P- and E-cores
differently each run; single-core numbers on both sides are stable, so trust those
first. The single-core rows above were taken in the same minute as ours (the machine
drifts thermally over a benchmarking session). The GEMM row is the widest margin:
their n=8 CPU path gains little over n=1, while reading the weight stream once per
row-tile and reusing it across columns in L1 approaches this chip's Metal GPU number
(590 GFLOPS) on CPU cores alone.

**Not in this table: the Metal GPU backend**, which does the same case at 590
GFLOPS. A GPU is faster and always will be — it is a different device, and driving
it belongs a layer above a zero-dependency CPU substrate. Everything here is
CPU-only on purpose.

What this says:

- **Fusion is validated.** Against the pipeline a numpy/torch user actually writes —
  materialize fp32 weights, then call BLAS (3.2 GFLOP/s) — the fused fp32-activation
  kernel is **8x faster on 4x less memory**, and the int8-activation path is 27x.
- **fp8 is a genuine strength**: nothing else here vectorizes fp8 decode at all.
- **The bf16/fp16 conversion gap was lane width, and it's closed** — the loops now
  run 16 elements/iteration and sit at 89–99% of numpy, which is at memory
  bandwidth. fp8 encode (round-to-odd f16 + a 64 KB table) matches numpy exactly;
  the quantize direction matches ggml's q8_0 number to the digit and leads on q4_0.
- **The ggml block dequant is vectorized and roughly at parity** — ahead of ggml on
  Q4_0, 54% on Q8_0. Half the win was decoding the fp16 block scales through the
  hardware converter four at a time instead of the software bit-twiddle (5.3 → 8.0
  Gelem/s on its own).
- **bf16 dot is now the strongest result** — 2.3x numpy, and 6.7x ggml for a dull
  reason: `ggml_vec_dot_bf16` has AVX512-BF16 / AVX512F / AVX2 paths and *no ARM
  path*, so on arm64 it runs its scalar tail. (Their AVX2 path is the same
  shift-into-fp32 trick with two accumulators that `simd/blas1` uses.)
- **llama.cpp's lead on matvec was algorithmic, not tuning** — and it is mostly
  closed. For `q4_0` they set `vec_dot_type = GGML_TYPE_Q8_0`: the *activations* get
  quantized to int8 once, then `ggml_vdotq_s32` (ARM SDOT) does 16 MACs per
  instruction; on top of that `GGML_USE_LLAMAFILE` tiles 4 weight rows per pass
  (`mnpack<4,…>`) so one activation load feeds four dots. Adopting both, over 4-bit
  weights, plus the build-mode and codegen findings above, took the single-core
  matvec from 14% of their throughput to 92% (4-bit) and **116%** (int8) — and
  every multi-thread row now clears their band. The matvec rows above are measured
  at *their* shape so nothing hinges on problem size.
- **A persistent thread pool matters at these speeds.** `LpPool` in
  `nim_lowprec/parallel` starts its workers once; at ~0.4–0.6 ms of serial work per
  matvec, per-call thread creation is a third of the multi-thread runtime. The
  caller owns the pool, so the substrate still has no ambient threads.
- **Not every plausible optimization survived measurement.** Vectorizing the
  per-group finalize was the obvious suspect and was **neutral** — tried, measured,
  reverted, with the numbers recorded in `simd/dequant.nim` so nobody re-derives it.
  The mystery it was meant to solve turned out to be the `-d:release` checks tax
  (see *The two findings* above), found by an ablation ladder, not by guessing.

**Verification:** the correctness suite runs in CI on **x86_64 (Linux)** and
**arm64 (macOS)**, regenerating the reference vectors per-arch so the differential
tests enforce bit-exactness on both. **x86 SIMD** (AVX2/FMA/F16C and the
maddubs int8-dot path) is verified in its own CI workflow, diffed bit-for-bit
against the scalar reference; **NEON** (including SDOT, the streaming paths and
the thread pool) runs on the arm64 macOS runners in the same CI — each ISA tested
on real silicon.

## License

MIT — see [LICENSE](LICENSE).
