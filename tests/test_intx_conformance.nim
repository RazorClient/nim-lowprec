## Conformance suite for the integer/packing types (self-contained — no oracle
## needed; the value set is small and the packing layout is asserted directly).

import std/unittest
import nim_lowprec

suite "I8":
  test "integer round-trip, clamp, rounding":
    for v in -128 .. 127:
      check toI8(float32(v)).toFloat32 == float32(v)
    check toI8(200.0'f32).toFloat32 == 127.0'f32 # clamp high
    check toI8(-200.0'f32).toFloat32 == -128.0'f32 # clamp low
    check toI8(2.6'f32).toFloat32 == 3.0'f32 # round
    check toI8(-2.6'f32).toFloat32 == -3.0'f32

suite "I4":
  test "all 16 values + two's-complement nibble encoding":
    for v in -8 .. 7:
      check toI4(float32(v)).toFloat32 == float32(v)
    check nibble(toI4(-8.0'f32)) == 0x8'u8
    check nibble(toI4(-1.0'f32)) == 0xf'u8
    check nibble(toI4(7.0'f32)) == 0x7'u8
    check nibble(toI4(0.0'f32)) == 0x0'u8
    check fromNibble(0x8'u8).toFloat32 == -8.0'f32
    check fromNibble(0xf'u8).toFloat32 == -1.0'f32

  test "clamp and round":
    check toI4(100.0'f32).toFloat32 == 7.0'f32
    check toI4(-100.0'f32).toFloat32 == -8.0'f32
    check toI4(2.6'f32).toFloat32 == 3.0'f32

  test "packInt4 layout (ggml low-nibble-first)":
    var packed = newSeq[byte](1)
    packInt4([toI4(3.0'f32), toI4(-2.0'f32)], packed) # low=3(0x3), high=-2(0xE) → 0xE3
    check packed[0] == 0xE3'u8

  test "packInt4 / unpackInt4 round-trip (all 16 values)":
    var vals: seq[I4]
    for v in -8 .. 7:
      vals.add toI4(float32(v))
    var pk = newSeq[byte]((vals.len + 1) div 2)
    packInt4(vals, pk)
    var back = newSeq[I4](vals.len)
    unpackInt4(pk, back)
    for i in 0 ..< vals.len:
      check back[i] == vals[i]

  test "odd-length pack keeps the low nibble":
    var pk = newSeq[byte](1)
    packInt4([toI4(5.0'f32)], pk)
    check (pk[0] and 0x0f'u8) == 0x5'u8

suite "I1 (sign bit)":
  test "sign semantics":
    check toI1(3.0'f32).toFloat32 == 1.0'f32
    check toI1(-3.0'f32).toFloat32 == -1.0'f32

  test "packInt1 layout (LSB-first) + round-trip":
    let signs = [toI1(1.0'f32), toI1(-1.0'f32), toI1(-1.0'f32), toI1(1.0'f32)]
      # +,-,-,+ → 0b0110
    var pk = newSeq[byte](1)
    packInt1(signs, pk)
    check pk[0] == 0x06'u8
    var back = newSeq[I1](4)
    unpackInt1(pk, back)
    for i in 0 ..< 4:
      check back[i].toFloat32 == signs[i].toFloat32
