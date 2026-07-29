# Package

version       = "0.1.0"
author        = "Tamaghna Choudhuri"
description   = "Zero-dependency low-precision numeric substrate (bf16/fp16/fp8/MXFP4/MXFP6/E8M0/int8/int4/int1) + quantization + NEON/AVX2 kernels for ML inference in Nim"
license       = "MIT"
srcDir        = "src"

# Dependencies

requires "nim >= 2.2.0"

# Tasks

task test, "Run the correctness suite (this is what CI runs)":
  exec "nim c -r --hints:off tests/test_bf16.nim"
  exec "nim c -r --hints:off tests/test_bf16_conformance.nim"
  exec "nim c -r --hints:off tests/test_f16_conformance.nim"
  exec "nim c -r --hints:off tests/test_f8_conformance.nim"
  exec "nim c -r --hints:off tests/test_mx_conformance.nim"
  exec "nim c -r --hints:off tests/test_intx_conformance.nim"
  exec "nim c -r --hints:off tests/test_reduce.nim"
  exec "nim c -r --hints:off tests/test_quant.nim"
  exec "nim c -r --hints:off tests/test_interop.nim"
  exec "nim c -r --hints:off tests/test_mxfp6_pack.nim"
  exec "nim c -r --hints:off tests/test_stochastic.nim"
  exec "nim c -r --hints:off tests/test_ggml.nim"
  exec "nim c -r --hints:off tests/test_parallel.nim"
  exec "nim c -r --hints:off tests/test_sdot.nim"

task simd, "Run NEON SIMD tests (arm64; CI runs this on the macOS runners; x86 SIMD has its own workflow)":
  exec "nim c -r --hints:off tests/test_bf16_simd.nim"
  exec "nim c -r --hints:off tests/test_quant_simd.nim"
  exec "nim c -r --hints:off tests/test_blas1_simd.nim"
  exec "nim c -r --hints:off tests/test_f8_simd.nim"
  # second run of the SDOT suite with the stream threshold at 0, so the
  # row-at-a-time paths (normally only taken above 12 MB of weights) are pinned too
  exec "nim c -r --hints:off -d:lpStreamBytes=0 tests/test_sdot.nim"

task example, "Run the quickstart example":
  exec "nim c -r --hints:off examples/quickstart.nim"

task refs, "Generate reference vectors (needs python3 + numpy + ml_dtypes)":
  exec "python3 tests/gen_reference.py"

# Microbenchmarks. Every bench times a scalar reference against the kernel the
# build selected, cross-checks their checksums, and quits non-zero on a mismatch —
# so a "speedup" can't come from a path that stopped computing the right answer.
const benchmarkFiles = [
  "bench_bf16",     # scalar round-trip latency (dependent chain)
  "bench_simd",     # batch conversions: bf16 / fp16 / fp8
  "bench_dequant",  # elementwise dequant: int8, int4, asymmetric fallback
  "bench_blas1",    # bf16 dot
  "bench_gemv",     # fused dequant-GEMV: int8, packed int4, packed mxfp4
  "bench_ggml",     # ggml Q8_0 / Q4_0: block dequant + fused GEMV
  "bench_parallel", # multi-core scaling of the fused GEMV (caller-side threading)
  "bench_sdot",     # int8-activation GEMV: SDOT + 4-bit weights vs the fp32 path
  "bench_gemm",     # multi-column GEMM: weight-stream reuse across n activation columns
]

proc runBenchmarks(flags: string) =
  for b in benchmarkFiles:
    exec "nim c -r -d:release --hints:off " & flags & " benchmarks/" & b & ".nim"

task bench, "Run microbenchmarks on the native SIMD path":
  runBenchmarks("")

task benchScalar, "Run the same microbenchmarks with SIMD off (-d:lpSimd=scalar)":
  runBenchmarks("-d:lpSimd=scalar")

task benchAll, "Run the microbenchmarks both ways: native SIMD, then scalar fallback":
  echo "==== native SIMD kernels ===="
  runBenchmarks("")
  echo ""
  echo "==== -d:lpSimd=scalar (kernels fall back to portable code) ===="
  runBenchmarks("-d:lpSimd=scalar")
