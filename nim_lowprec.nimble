# Package
version = "1.0.0"
author = "Tamaghna Choudhuri"
description = "Zero-dependency low-precision numeric substrate for ML inference in Nim"
license = "MIT"
srcDir = "src"

# Dependencies
requires "nim >= 2.2.0"

const
  testOutDir = "build/tests"
  exampleOutDir = "build/examples"
  benchmarkOutDir = "build/benchmarks"
  nimCacheDir = "build/nimcache"

proc ensureOutDir(dir: string) =
  exec "mkdir -p " & dir

proc runTestFile(file: string, flags = "") =
  ensureOutDir(testOutDir)
  ensureOutDir(nimCacheDir)
  exec "nim c -r --nimcache:" & nimCacheDir & " --outdir:" & testOutDir & " --hints:off " &
    flags & " " & file

proc runExampleFile(file: string, flags = "") =
  ensureOutDir(exampleOutDir)
  ensureOutDir(nimCacheDir)
  exec "nim c -r --nimcache:" & nimCacheDir & " --outdir:" & exampleOutDir &
    " --hints:off " & flags & " " & file

proc runBenchmarkFile(file: string, flags = "") =
  ensureOutDir(benchmarkOutDir)
  ensureOutDir(nimCacheDir)
  exec "nim c -r --nimcache:" & nimCacheDir & " --outdir:" & benchmarkOutDir &
    " -d:release --hints:off " & flags & " " & file

task test, "Run the correctness tests":
  runTestFile("tests/test_bf16.nim")
  runTestFile("tests/test_bf16_conformance.nim")
  runTestFile("tests/test_f16_conformance.nim")
  runTestFile("tests/test_f8_conformance.nim")
  runTestFile("tests/test_mx_conformance.nim")
  runTestFile("tests/test_intx_conformance.nim")
  runTestFile("tests/test_reduce.nim")
  runTestFile("tests/test_quant.nim")
  runTestFile("tests/test_interop.nim")
  runTestFile("tests/test_mxfp6_pack.nim")
  runTestFile("tests/test_stochastic.nim")
  runTestFile("tests/test_ggml.nim")
  runTestFile("tests/test_parallel.nim")
  runTestFile("tests/test_sdot.nim")
  runTestFile("tests/test_scheme_conformance.nim")

task simd, "Run NEON SIMD tests":
  runTestFile("tests/test_bf16_simd.nim")
  runTestFile("tests/test_quant_simd.nim")
  runTestFile("tests/test_blas1_simd.nim")
  runTestFile("tests/test_f8_simd.nim")
  runTestFile("tests/test_sdot.nim", "-d:lpStreamBytes=0")

task example, "Run the quickstart example":
  runExampleFile("examples/quickstart.nim")

task refs, "Generate reference vectors (needs python3 + numpy + ml_dtypes + gguf)":
  exec "python3 tests/gen_reference.py"

const benchmarkFiles = [
  "bench_bf16", "bench_simd", "bench_dequant", "bench_blas1", "bench_gemv",
  "bench_ggml", "bench_parallel", "bench_sdot", "bench_gemm",
]

proc runBenchmarks(flags: string) =
  for b in benchmarkFiles:
    runBenchmarkFile("benchmarks/" & b & ".nim", flags)

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
