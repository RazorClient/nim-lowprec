## Conformance suite for the four fp8 types.
## Only 256 codes each → decode is EXHAUSTIVE; encode is edges + random.
## Generate vectors first: nimble refs (needs ml_dtypes). Differential tests
## skip if vectors are absent. LITTLE-ENDIAN ONLY.

import std/[unittest, streams, os]
import nim_lowprec/float8

func f32bits(u: uint32): float32 = cast[float32](u)
const refDir = currentSourcePath().parentDir

template f8suite(TT, toFn: untyped; nm, decf, encf: static string) =
  suite nm:

    test "decode∘encode round-trips every finite code":
      for c in 0'u8 .. 255'u8:
        let x = TT(c)
        if x.isNaN: continue
        check bits(toFn(x.toFloat32)) == c

    test "decode all 256 vs ml_dtypes":
      let p = refDir / decf
      if not fileExists(p):
        skip()
      else:
        let s = newFileStream(p, fmRead)
        defer: s.close()
        var mm = 0
        for c in 0'u8 .. 255'u8:
          let want = s.readUint32()
          let x = TT(c)
          let v = x.toFloat32
          if x.isNaN:
            if v == v: inc mm
          elif cast[uint32](v) != want:
            inc mm
        check mm == 0

    test "encode f32 vs ml_dtypes (edges + 2M random)":
      let p = refDir / encf
      if not fileExists(p):
        skip()
      else:
        let s = newFileStream(p, fmRead)
        defer: s.close()
        let n = s.readUint32().int
        var ins = newSeq[uint32](n)
        for i in 0 ..< n: ins[i] = s.readUint32()
        var mm = 0
        for i in 0 ..< n:
          let want = s.readUint8()
          let inF = f32bits(ins[i])
          let got = toFn(inF)
          if inF != inF:
            if not got.isNaN: inc mm
          elif bits(got) != want:
            inc mm
        check mm == 0

f8suite(F8E4M3,     toF8E4M3,     "fp8 e4m3fn (OCP)",   "ref_e4m3_to_f32.bin",     "ref_f32_to_e4m3.bin")
f8suite(F8E5M2,     toF8E5M2,     "fp8 e5m2 (OCP)",     "ref_e5m2_to_f32.bin",     "ref_f32_to_e5m2.bin")
f8suite(F8E4M3FNUZ, toF8E4M3FNUZ, "fp8 e4m3fnuz (AMD)", "ref_e4m3fnuz_to_f32.bin", "ref_f32_to_e4m3fnuz.bin")
f8suite(F8E5M2FNUZ, toF8E5M2FNUZ, "fp8 e5m2fnuz (AMD)", "ref_e5m2fnuz_to_f32.bin", "ref_f32_to_e5m2fnuz.bin")
