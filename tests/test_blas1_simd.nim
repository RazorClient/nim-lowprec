## SIMD bf16 dot vs an fp64 reference (reduction → tolerance, not bit-exact). Local.

import std/unittest
import nim_lowprec/[bfloat16, simd/blas1]

suite "SIMD bf16 dot":

  test "dotBf16 matches fp64 reference (aligned + ragged tail)":
    for n in [256, 259]:                         # 259 exercises the scalar tail
      var a = newSeq[BF16](n)
      var b = newSeq[BF16](n)
      var refv = 0.0'f64
      var u = 99991'u32
      for i in 0 ..< n:
        u = u * 1664525'u32 + 1013904223'u32
        let av = float32(u shr 8) / float32(1'u32 shl 24) * 2.0'f32 - 1.0'f32
        u = u * 1664525'u32 + 1013904223'u32
        let bv = float32(u shr 8) / float32(1'u32 shl 24) * 2.0'f32 - 1.0'f32
        a[i] = toBF16(av); b[i] = toBF16(bv)
        refv += float64(a[i].toFloat32) * float64(b[i].toFloat32)
      let got = dotBf16(a, b)
      check abs(float64(got) - refv) / (abs(refv) + 1e-4) < 1e-3
