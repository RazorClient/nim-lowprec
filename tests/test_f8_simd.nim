## fp8 SIMD convert must be BIT-IDENTICAL to the scalar decode (NaN-guarded).
## Exhaustive over all 256 codes; the ragged length exercises the scalar tail. Local.

import std/unittest
import nim_lowprec
import nim_lowprec/simd/convert

template f8Suite(TT: untyped, nm: static string) =
  suite nm:
    test "toFloat32Batch matches scalar for all 256 codes (aligned + ragged tail)":
      for total in [256, 259]:
        var src = newSeq[TT](total)
        for c in 0 ..< total:
          src[c] = TT(uint8(c and 0xff))
        var simd = newSeq[float32](total)
        toFloat32Batch(src, simd)
        var mism = 0
        for c in 0 ..< total:
          let s = toFloat32(src[c]) # scalar reference
          if s != s: # scalar produced NaN
            if simd[c] == simd[c]:
              inc mism
              # SIMD must also be NaN
          elif cast[uint32](simd[c]) != cast[uint32](s):
            inc mism
        check mism == 0

f8Suite(F8E5M2, "fp8 e5m2 SIMD == scalar")
f8Suite(F8E4M3, "fp8 e4m3 SIMD == scalar")
f8Suite(F8E4M3FNUZ, "fp8 e4m3fnuz SIMD == scalar")
f8Suite(F8E5M2FNUZ, "fp8 e5m2fnuz SIMD == scalar")

import std/math

template f8EncodeSuite(TT, toFn, batchFn: untyped, nm: static string) =
  suite nm:
    test "batch encode matches scalar over sweep + edges":
      # Dense magnitude sweep (all f16 codes as f32 — the LUT's whole domain),
      # plus values BETWEEN f16 codes (rounding exercised), plus every edge the
      # encoder branches on. NaN codes compare as codes, not values.
      var src: seq[float32]
      for h in 0 ..< 65536:
        let v = toFloat32(F16(uint16(h)))
        src.add v
        if abs(v) < 1e30:
          src.add v * 1.0000001'f32
          # off-grid neighbour
      for v in [
        0.0'f32,
        -0.0,
        448.0,
        448.1,
        463.99,
        464.0,
        464.001,
        465.0,
        240.0,
        247.9,
        248.0,
        248.1,
        57344.0,
        61439.0,
        61440.0,
        61441.0,
        1e-8,
        -1e-8,
        6e-8,
        1e30,
        -1e30,
        Inf,
        -Inf,
        NaN,
      ]:
        src.add v
      while src.len mod 8 != 3:
        src.add 1.0'f32
        # ragged tail
      var got = newSeq[TT](src.len)
      batchFn(src, got)
      var mism = 0
      for i in 0 ..< src.len:
        if uint8(got[i]) != uint8(toFn(src[i])):
          inc mism
      check mism == 0

f8EncodeSuite(F8E4M3, toF8E4M3, toF8E4M3Batch, "fp8 e4m3 batch ENCODE == scalar")
f8EncodeSuite(F8E5M2, toF8E5M2, toF8E5M2Batch, "fp8 e5m2 batch ENCODE == scalar")
f8EncodeSuite(
  F8E4M3FNUZ, toF8E4M3FNUZ, toF8E4M3FNUZBatch, "fp8 e4m3fnuz batch ENCODE == scalar"
)
f8EncodeSuite(
  F8E5M2FNUZ, toF8E5M2FNUZ, toF8E5M2FNUZBatch, "fp8 e5m2fnuz batch ENCODE == scalar"
)
