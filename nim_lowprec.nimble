# Package
version = "1.0.0"
author = "Tamaghna Choudhuri"
description = "Zero-dependency low-precision numeric substrate for ML inference in Nim"
license = "MIT"
srcDir = "src"

# Dependencies
requires "nim >= 2.2.0"

# Tasks

task test, "Run the correctness tests":
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
  exec "nim c -r --hints:off tests/test_scheme_conformance.nim"

task simd, "Run NEON SIMD tests":
  exec "nim c -r --hints:off tests/test_bf16_simd.nim"
  exec "nim c -r --hints:off tests/test_quant_simd.nim"
  exec "nim c -r --hints:off tests/test_blas1_simd.nim"
  exec "nim c -r --hints:off tests/test_f8_simd.nim"
  exec "nim c -r --hints:off -d:lpStreamBytes=0 tests/test_sdot.nim"

task example, "Run the quickstart example":
  exec "nim c -r --hints:off examples/quickstart.nim"

task refs, "Generate reference vectors (needs python3 + numpy + ml_dtypes + gguf)":
  exec "python3 tests/gen_reference.py"

const benchmarkFiles = [
  "bench_bf16", "bench_simd", "bench_dequant", "bench_blas1", "bench_gemv",
  "bench_ggml", "bench_parallel", "bench_sdot", "bench_gemm",
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
