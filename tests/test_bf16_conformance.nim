# Conformance suite for BF16:
#   - self-checking invariants
#   - a differential check against ml_dtypes golden vectors

import std/[unittest, streams, os]
import nim_lowprec/bfloat16

func f32bits(u: uint32): float32 = cast[float32](u)
const refDir = currentSourcePath().parentDir

suite "bf16 invariants (self-checking, no oracle)":

  test "bf16 -> f32 -> bf16 identity for all 65536 patterns":
    for i in 0'u32 .. 65535'u32:
      let b = BF16(uint16(i))
      if b.isNaN: continue                 # NaN payload need not survive
      check bits(toBF16(b.toFloat32)) == bits(b)

  test "NaN survives (the 0x7F800001 trap), canonical & sign-preserving":
    check toBF16(f32bits(0x7f80_0001'u32)).isNaN
    check bits(toBF16(f32bits(0x7f80_0001'u32))) == 0x7fc0'u16
    check bits(toBF16(f32bits(0xff80_0001'u32))) == 0xffc0'u16

  test "Inf preserved with sign":
    check toBF16(f32bits(0x7f80_0000'u32)).isInf
    check bits(toBF16(f32bits(0x7f80_0000'u32))) == 0x7f80'u16
    check bits(toBF16(f32bits(0xff80_0000'u32))) == 0xff80'u16

  test "round-to-nearest-even on exact ties":
    check bits(toBF16(f32bits(0x3f80_8000'u32))) == 0x3f80'u16    # tie -> even (down)
    check bits(toBF16(f32bits(0x3f81_8000'u32))) == 0x3f82'u16    # tie -> even (up)

  test "truncating variant is NaN-safe":
    check toBF16Trunc(f32bits(0x7f80_0001'u32)).isNaN

suite "bf16 differential vs ml_dtypes (skips if vectors absent)":

  test "bf16 -> f32 exhaustive (65536)":
    let path = refDir / "ref_bf16_to_f32.bin"
    if not fileExists(path):
      skip()
    else:
      let s = newFileStream(path, fmRead)
      defer: s.close()
      for i in 0'u32 .. 65535'u32:
        let want = s.readUint32()
        check cast[uint32](BF16(uint16(i)).toFloat32) == want

  test "f32 -> bf16 differential (edges + 4M random)":
    let path = refDir / "ref_f32_to_bf16.bin"
    if not fileExists(path):
      skip()
    else:
      let s = newFileStream(path, fmRead)
      defer: s.close()
      let n = s.readUint32().int
      var ins = newSeq[uint32](n)
      for i in 0 ..< n: ins[i] = s.readUint32()
      var mismatches = 0
      for i in 0 ..< n:
        let want = s.readUint16()
        let inF = f32bits(ins[i])
        let got = toBF16(inF)
        if inF != inF:                       # NaN input: oracle canonicalization may differ
          if not got.isNaN: inc mismatches
        elif bits(got) != want:
          inc mismatches
      check mismatches == 0
