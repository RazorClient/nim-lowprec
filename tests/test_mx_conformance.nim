## Conformance for the MX formats: MXFP4 (e2m1), MXFP6 (e2m3, e3m2) + E8M0 scale.
## Tiny code spaces → decode is EXHAUSTIVE; encode is finite edges + 2M random.
## Verified vs ml_dtypes (float4_e2m1fn / float6_*fn / float8_e8m0fnu).
## Generate vectors first: `nimble refs`. Differential tests skip if absent.
## fp4/fp6 have NO NaN, so NaN *inputs* to encode are undefined and skipped.
## LITTLE-ENDIAN ONLY.

import std/[unittest, streams, os]
import nim_lowprec

func f32bits(u: uint32): float32 =
  cast[float32](u)
const refDir = currentSourcePath().parentDir / "refs"

template mxFloatSuite(
    TT, toFn: untyped, nm, decf, encf: static string, ncodes: static int
) =
  suite nm:
    test "decode∘encode round-trips every code":
      for c in 0 ..< ncodes:
        let x = TT(uint8(c))
        check bits(toFn(x.toFloat32)) == uint8(c)

    test "decode all codes vs ml_dtypes":
      let p = refDir / decf
      if not fileExists(p):
        skip()
      else:
        let s = newFileStream(p, fmRead)
        defer:
          s.close()
        var mm = 0
        for c in 0 ..< ncodes:
          let want = s.readUint32()
          if cast[uint32](TT(uint8(c)).toFloat32) != want:
            inc mm
        check mm == 0

    test "encode f32 vs ml_dtypes (finite edges + 2M random; NaN skipped)":
      let p = refDir / encf
      if not fileExists(p):
        skip()
      else:
        let s = newFileStream(p, fmRead)
        defer:
          s.close()
        let n = s.readUint32().int
        var ins = newSeq[uint32](n)
        for i in 0 ..< n:
          ins[i] = s.readUint32()
        var mm = 0
        for i in 0 ..< n:
          let want = s.readUint8()
          let inF = f32bits(ins[i])
          if inF != inF:
            continue # NaN input undefined for finite formats
          if bits(toFn(inF)) != want:
            inc mm
        check mm == 0

mxFloatSuite(
  F4E2M1, toF4E2M1, "MXFP4 e2m1", "ref_f4e2m1_to_f32.bin", "ref_f32_to_f4e2m1.bin", 16
)
mxFloatSuite(
  F6E2M3, toF6E2M3, "MXFP6 e2m3", "ref_f6e2m3_to_f32.bin", "ref_f32_to_f6e2m3.bin", 64
)
mxFloatSuite(
  F6E3M2, toF6E3M2, "MXFP6 e3m2", "ref_f6e3m2_to_f32.bin", "ref_f32_to_f6e3m2.bin", 64
)

suite "E8M0 shared scale":
  test "decode all 256 vs ml_dtypes":
    let p = refDir / "ref_e8m0_to_f32.bin"
    if not fileExists(p):
      skip()
    else:
      let s = newFileStream(p, fmRead)
      defer:
        s.close()
      var mm = 0
      for c in 0 .. 255:
        let want = s.readUint32()
        let x = E8M0(uint8(c))
        let v = x.toFloat32
        if x.isNaN:
          if v == v:
            inc mm
            # 0xFF must decode to NaN
        elif cast[uint32](v) != want:
          inc mm
      check mm == 0

  test "encode f32 vs ml_dtypes (positive inputs; 0/inf/nan → 0xFF)":
    let p = refDir / "ref_f32_to_e8m0.bin"
    if not fileExists(p):
      skip()
    else:
      let s = newFileStream(p, fmRead)
      defer:
        s.close()
      let n = s.readUint32().int
      var ins = newSeq[uint32](n)
      for i in 0 ..< n:
        ins[i] = s.readUint32()
      var mm = 0
      for i in 0 ..< n:
        let want = s.readUint8()
        if bits(toE8M0(f32bits(ins[i]))) != want:
          inc mm
      check mm == 0

suite "MX packing":
  test "F4 pack/unpack round-trips (ragged)":
    for total in [16, 15]:
      var vals = newSeq[F4E2M1](total)
      for i in 0 ..< total:
        vals[i] = F4E2M1(uint8(i and 0x0f))
      var packed = newSeq[byte]((total + 1) div 2)
      packF4(vals, packed)
      var un = newSeq[F4E2M1](total)
      unpackF4(packed, un)
      var mm = 0
      for i in 0 ..< total:
        if uint8(un[i]) != uint8(i and 0x0f):
          inc mm
      check mm == 0
