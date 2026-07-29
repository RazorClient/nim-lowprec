## nim_lowprec/reduce — BLAS-1 with fp32 accumulation, generic over LowPrec.
##
## This is the "store narrow, accumulate WIDE" contract, written ONCE and valid
## for every dtype that satisfies `LowPrec` (bf16, fp16, fp8×4, I8, …). It is the
## first real consumer of the concept. Scalar reference only — the SIMD paths
## (VDPBF16PS / BFDOT / VNNI) land later and are diffed against these.
##
## Scope: 1-D reductions/vector ops only. GEMV / GEMM / attention are the
## kernel layer's job (they need tiling, packing, threading), not the substrate.

import std/math
import ./common

template foldDecoded(x, contribution: untyped): float32 =
  ## Accumulate `x` into a float32 total (the accumulate-WIDE half of the
  ## contract). `it` is the fp32-decoded element; `contribution` is its addend.
  ## `decode` is resolved by the enclosing generic's `mixin decode`.
  var acc = 0.0'f32
  for v in x:
    let it {.inject.} = decode(v)
    acc += contribution
  acc

func sum*[T: LowPrec](x: openArray[T]): float32 =
  ## Σ xᵢ, accumulated in float32.
  mixin decode
  foldDecoded(x, it)

func asum*[T: LowPrec](x: openArray[T]): float32 =
  ## Σ |xᵢ|.
  mixin decode
  foldDecoded(x, abs(it))

func dot*[T: LowPrec](a, b: openArray[T]): float32 =
  ## Σ aᵢ·bᵢ, accumulated in float32 (the core contract).
  mixin decode
  assert a.len == b.len
  result = 0.0'f32
  for i in 0 ..< a.len:
    result += decode(a[i]) * decode(b[i])

func nrm2*[T: LowPrec](x: openArray[T]): float32 =
  ## Euclidean norm √(Σ xᵢ²).
  mixin decode
  sqrt(foldDecoded(x, it * it))

func absmax*[T: LowPrec](x: openArray[T]): float32 =
  ## max |xᵢ| — the one stateless statistic quantizers key their scale off.
  mixin decode
  result = 0.0'f32
  for v in x:
    let a = abs(decode(v))
    if a > result:
      result = a

func scal*[T: LowPrec](a: float32, x: var openArray[T]) =
  ## x ← a·x  (compute in fp32, store narrow).
  mixin decode, encode
  for i in 0 ..< x.len:
    x[i] = encode(a * decode(x[i]), T)

func axpy*[T: LowPrec](a: float32, x: openArray[T], y: var openArray[T]) =
  ## y ← a·x + y  (compute in fp32, store narrow).
  mixin decode, encode
  assert x.len == y.len
  for i in 0 ..< x.len:
    y[i] = encode(a * decode(x[i]) + decode(y[i]), T)
