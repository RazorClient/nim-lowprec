## Round-trip + bit-layout tests for the MXFP6 tight-packing (packF6/unpackF6):
## FOUR 6-bit codes per THREE bytes, LSB-first. Covers both F6E2M3 and F6E3M2
## (both are 6-bit codes in the low 6 bits) and ragged tails.

import std/unittest
import nim_lowprec/mxfloat

func packedLen(n: int): int = (n * 6 + 7) div 8

template roundTripSuite(TT: untyped; nm: static string) =
  suite nm:

    test "round-trips all 64 codes":
      var vals = newSeq[TT](64)
      for c in 0 ..< 64: vals[c] = TT(uint8(c))
      var packed = newSeq[byte](packedLen(64))
      packF6(vals, packed)
      var un = newSeq[TT](64)
      unpackF6(packed, un)
      var mm = 0
      for c in 0 ..< 64:
        if uint8(un[c]) != uint8(c): inc mm
      check mm == 0

    test "round-trips ragged lengths":
      for total in [1, 2, 3, 5, 6, 7, 9, 63, 100]:
        var vals = newSeq[TT](total)
        for i in 0 ..< total: vals[i] = TT(uint8((i * 7 + 1) and 0x3f))
        var packed = newSeq[byte](packedLen(total))
        packF6(vals, packed)
        var un = newSeq[TT](total)
        unpackF6(packed, un)
        var mm = 0
        for i in 0 ..< total:
          if uint8(un[i]) != uint8((i * 7 + 1) and 0x3f): inc mm
        check mm == 0

    test "unused tail bytes are exactly the minimum":
      # a lone value uses 1 byte, two use 2, three use 3, four use 3
      check packedLen(1) == 1
      check packedLen(2) == 2
      check packedLen(3) == 3
      check packedLen(4) == 3
      check packedLen(5) == 4

roundTripSuite(F6E2M3, "MXFP6 e2m3 pack")
roundTripSuite(F6E3M2, "MXFP6 e3m2 pack")

suite "MXFP6 exact bit-layout":

  test "LSB-first 24-bit little-endian group":
    # value0=1, value1=2, value2=3, value3=0
    #   g = 1 | (2<<6) | (3<<12) = 0x0001 | 0x0080 | 0x3000 = 0x3081
    #   little-endian bytes: [0x81, 0x30, 0x00]
    var vals = @[F6E2M3(1), F6E2M3(2), F6E2M3(3), F6E2M3(0)]
    var packed = newSeq[byte](3)
    packF6(vals, packed)
    check packed[0] == 0x81'u8
    check packed[1] == 0x30'u8
    check packed[2] == 0x00'u8

  test "all-ones tiles the 24 bits":
    var vals = @[F6E3M2(0x3f), F6E3M2(0x3f), F6E3M2(0x3f), F6E3M2(0x3f)]
    var packed = newSeq[byte](3)
    packF6(vals, packed)
    check packed[0] == 0xff'u8
    check packed[1] == 0xff'u8
    check packed[2] == 0xff'u8

  test "each field lands in its documented bit slot":
    # value1 = 0x3f (all six bits) with the others 0 → g = 0x3f << 6 = 0x0FC0
    #   bytes: [0xC0, 0x0F, 0x00]  (low 2 bits of value1 in byte0[7:6],
    #   high 4 bits in byte1[3:0])
    var vals = @[F6E2M3(0), F6E2M3(0x3f), F6E2M3(0), F6E2M3(0)]
    var packed = newSeq[byte](3)
    packF6(vals, packed)
    check packed[0] == 0xc0'u8
    check packed[1] == 0x0f'u8
    check packed[2] == 0x00'u8
