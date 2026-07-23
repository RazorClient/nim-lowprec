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

task bench, "Run microbenchmarks":
  exec "nim c -r -d:release --hints:off benchmarks/bench_bf16.nim"
