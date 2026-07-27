# Conformance suite for F16 (IEEE binary16):
#   - self-checking invariants
#   - a differential check against numpy.float16 golden vectors

import std/[unittest, streams, os]
import nim_lowprec/float16

func f32bits(u: uint32): float32 = cast[float32](u)
const refDir = currentSourcePath().parentDir

suite "fp16 invariants (self-checking, no oracle)":

  test "f16 -> f32 -> f16 identity for all 65536 patterns":
    for i in 0'u32 .. 65535'u32:
      let h = F16(uint16(i))
      if h.isNaN: continue
      check bits(toF16(h.toFloat32)) == bits(h)

  test "unit values and signs":
    check bits(toF16(1.0'f32)) == 0x3c00'u16
    check bits(toF16(-2.0'f32)) == 0xc000'u16
    check bits(toF16(0.0'f32)) == 0x0000'u16

  test "Inf / NaN preserved":
    check toF16(f32bits(0x7f80_0000'u32)).isInf
    check bits(toF16(f32bits(0x7f80_0000'u32))) == 0x7c00'u16
    check bits(toF16(f32bits(0xff80_0000'u32))) == 0xfc00'u16
    check toF16(f32bits(0x7fc0_0000'u32)).isNaN

  test "overflow saturates to Inf; max normal is exact":
    check bits(toF16(65504.0'f32)) == 0x7bff'u16       # max fp16 normal
    check toF16(70000.0'f32).isInf                     # > 65520 → +Inf

  test "subnormals represented, not flushed; tiny → signed zero":
    check bits(toF16(f32bits(0x33800000'u32))) == 0x0001'u16  # 2^-24 = min subnormal
    check bits(toF16(1e-9'f32)) == 0x0000'u16                 # far below 2^-25 → +0

suite "fp16 differential vs numpy.float16 (skips if vectors absent)":

  test "f16 -> f32 exhaustive (65536)":
    let path = refDir / "ref_f16_to_f32.bin"
    if not fileExists(path):
      skip()
    else:
      let s = newFileStream(path, fmRead)
      defer: s.close()
      var mismatches = 0
      for i in 0'u32 .. 65535'u32:
        let want = s.readUint32()
        let h = F16(uint16(i))
        let v = h.toFloat32
        if h.isNaN:
          if v == v: inc mismatches          # must be NaN (float32 payload not standardized)
        elif cast[uint32](v) != want:
          inc mismatches
      check mismatches == 0

  test "f32 -> f16 differential (edges + 4M random)":
    let path = refDir / "ref_f32_to_f16.bin"
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
        let got = toF16(inF)
        if inF != inF:                          # NaN input
          if not got.isNaN: inc mismatches
        elif bits(got) != want:
          inc mismatches
      check mismatches == 0
