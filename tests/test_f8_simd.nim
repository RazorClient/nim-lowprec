## fp8 SIMD convert must be BIT-IDENTICAL to the scalar decode (NaN-guarded).
## Exhaustive over all 256 codes; the ragged length exercises the scalar tail. Local.

import std/unittest
import nim_lowprec/[float8, simd/convert]

template f8Suite(TT: untyped; nm: static string) =
  suite nm:
    test "toFloat32Batch matches scalar for all 256 codes (aligned + ragged tail)":
      for total in [256, 259]:
        var src = newSeq[TT](total)
        for c in 0 ..< total: src[c] = TT(uint8(c and 0xff))
        var simd = newSeq[float32](total)
        toFloat32Batch(src, simd)
        var mism = 0
        for c in 0 ..< total:
          let s = toFloat32(src[c])                 # scalar reference
          if s != s:                                # scalar produced NaN
            if simd[c] == simd[c]: inc mism         # SIMD must also be NaN
          elif cast[uint32](simd[c]) != cast[uint32](s):
            inc mism
        check mism == 0

f8Suite(F8E5M2, "fp8 e5m2 SIMD == scalar")
f8Suite(F8E4M3, "fp8 e4m3 SIMD == scalar")
f8Suite(F8E4M3FNUZ, "fp8 e4m3fnuz SIMD == scalar")
