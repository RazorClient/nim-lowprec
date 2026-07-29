## ggml Q8_0 / Q4_0 block formats: explicit dequant-formula checks + quantize
## round-trips bounded by the quantization step (+ the fp16-scale error).

import std/[unittest, math]
import nim_lowprec/[float16, ggml]

suite "ggml block formats":

  test "Q8_0 dequant is d·q (hand-built block)":
    var blk: BlockQ8_0
    blk.d = toF16(0.5'f32)
    for i in 0 ..< QK: blk.qs[i] = int8(i - 16)          # -16 .. 15
    var dst = newSeq[float32](QK)
    dequantizeQ8_0([blk], dst)
    for i in 0 ..< QK:
      check dst[i] == 0.5'f32 * float32(i - 16)

  test "Q4_0 dequant is d·(nibble-8), low→lane i, high→lane i+16 (hand-built block)":
    var blk: BlockQ4_0
    blk.d = toF16(2.0'f32)
    for i in 0 ..< QK div 2:
      blk.qs[i] = uint8(i or ((15 - i) shl 4))            # low nibble = i, high = 15-i
    var dst = newSeq[float32](QK)
    dequantizeQ4_0([blk], dst)
    for i in 0 ..< QK div 2:
      check dst[i]            == 2.0'f32 * float32(i - 8)
      check dst[i + QK div 2] == 2.0'f32 * float32((15 - i) - 8)

  test "Q8_0 quantize→dequantize within one step":
    const NB = 4
    var x = newSeq[float32](NB * QK)
    var u = 4242'u32
    for i in 0 ..< x.len:
      u = u * 1664525'u32 + 1013904223'u32
      x[i] = (float32(u shr 8) / float32(1'u32 shl 24) * 2.0'f32 - 1.0'f32) * 5.0'f32
    var blocks = newSeq[BlockQ8_0](NB)
    quantizeQ8_0(x, blocks)
    var y = newSeq[float32](NB * QK)
    dequantizeQ8_0(blocks, y)
    for b in 0 ..< NB:
      var amax = 0.0'f32
      for i in 0 ..< QK: amax = max(amax, abs(x[b * QK + i]))
      let step = amax / 127.0'f32
      for i in 0 ..< QK:
        check abs(y[b * QK + i] - x[b * QK + i]) <= step + amax * 5e-3'f32 + 1e-5'f32

  test "Q4_0 quantize→dequantize within one 4-bit step":
    const NB = 4
    var x = newSeq[float32](NB * QK)
    var u = 99'u32
    for i in 0 ..< x.len:
      u = u * 1664525'u32 + 1013904223'u32
      x[i] = (float32(u shr 8) / float32(1'u32 shl 24) * 2.0'f32 - 1.0'f32) * 3.0'f32
    var blocks = newSeq[BlockQ4_0](NB)
    quantizeQ4_0(x, blocks)
    var y = newSeq[float32](NB * QK)
    dequantizeQ4_0(blocks, y)
    for b in 0 ..< NB:
      var amax = 0.0'f32
      for i in 0 ..< QK: amax = max(amax, abs(x[b * QK + i]))
      let step = amax / 8.0'f32                            # 4-bit resolution
      for i in 0 ..< QK:
        check abs(y[b * QK + i] - x[b * QK + i]) <= step + amax * 5e-3'f32 + 1e-5'f32
