# Package

version       = "0.1.0"
author        = "Tamaghna Choudhuri"
description   = "Low-precision floating-point primitives (bf16, fp16, fp8) for ML inference in Nim"
license       = "MIT"
srcDir        = "src"

# Dependencies

requires "nim >= 2.2.0"

# Tasks

task test, "Run the test suite":
  exec "nim c -r --hints:off tests/test_bf16.nim"
  exec "nim c -r --hints:off tests/test_bf16_conformance.nim"
  exec "nim c -r --hints:off tests/test_f16_conformance.nim"

task refs, "Generate reference vectors (needs python3 + numpy + ml_dtypes)":
  exec "python3 tests/gen_reference.py"

task bench, "Run microbenchmarks":
  exec "nim c -r -d:release --hints:off benchmarks/bench_bf16.nim"
