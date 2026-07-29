## SIMD batch convert must be BIT-IDENTICAL to the scalar reference. Local only.

import std/unittest
import nim_lowprec
import nim_lowprec/simd/convert

suite "bf16 SIMD == scalar":
  test "we are exercising the intended SIMD path":
    # When an ISA is pinned (-d:lpSimd=avx2 in CI), the backend MUST equal it —
    # this is what proves the avx2 cell really runs avx2, not a silent scalar
    # fallback. With the default "auto", it's the arch's baseline.
    when lpSimd == "auto":
      when defined(arm64) or defined(aarch64):
        check simdBackend == "neon"
      else:
        check simdBackend == "scalar"
    else:
      check simdBackend == lpSimd

  test "toFloat32Batch matches scalar for all 65536 codes (aligned + ragged tail)":
    for total in [65536, 65533]:
      var src = newSeq[BF16](total)
      for i in 0 ..< total:
        src[i] = BF16(uint16(i and 0xffff))
      var simd = newSeq[float32](total)
      toFloat32Batch(src, simd)
      var mism = 0
      for i in 0 ..< total:
        if cast[uint32](simd[i]) != cast[uint32](toFloat32(src[i])):
          inc mism
      check mism == 0

  test "toBF16Batch matches scalar over a 100k fp32 sweep (ragged tail)":
    var src = newSeq[float32](100_003)
    var u = 1'u32
    for i in 0 ..< src.len:
      u = u * 2654435761'u32 + 1'u32
      src[i] = cast[float32](u)
    var simd = newSeq[BF16](src.len)
    toBF16Batch(src, simd)
    var mism = 0
    for i in 0 ..< src.len:
      if bits(simd[i]) != bits(toBF16(src[i])):
        inc mism
    check mism == 0

suite "fp16 SIMD == scalar (NaN-guarded)":
  test "f16 -> f32 for all 65536 codes":
    var src = newSeq[F16](65536)
    for i in 0 ..< 65536:
      src[i] = F16(uint16(i))
    var simd = newSeq[float32](65536)
    toFloat32Batch(src, simd)
    var mism = 0
    for i in 0 ..< 65536:
      let s = toFloat32(src[i])
      if s != s: # scalar produced NaN
        if simd[i] == simd[i]:
          inc mism
          # SIMD must also be NaN
      elif cast[uint32](simd[i]) != cast[uint32](s):
        inc mism
    check mism == 0

  test "f32 -> f16 over a 100k sweep":
    var src = newSeq[float32](100_003)
    var u = 12345'u32
    for i in 0 ..< src.len:
      u = u * 2654435761'u32 + 1'u32
      src[i] = cast[float32](u)
    var simd = newSeq[F16](src.len)
    toF16Batch(src, simd)
    var mism = 0
    for i in 0 ..< src.len:
      if src[i] != src[i]: # NaN input
        if not simd[i].isNaN:
          inc mism
      elif bits(simd[i]) != bits(toF16(src[i])):
        inc mism
    check mism == 0
