## nim_lowprec/ggml — llama.cpp / GGUF block-quantization formats (Q8_0, Q4_0).
##
## The on-disk layout real quantized LLM weights ship in: a block of QK=32
## weights sharing one fp16 scale. This is the bridge for LOADING a real
## quantized model into the substrate (dequantize → fp32). `quantize*` is the
## inverse, matching ggml's reference algorithm, for round-trips / re-quant.
##
##   Q8_0: fp16 d + 32×int8            →  value[i] = d · q[i]
##   Q4_0: fp16 d + 16 bytes (32×nib)  →  value[i] = d · (nibble[i] − 8)
##         low nibbles are the first 16 lanes, high nibbles the last 16 (ggml order)
##
## (This is the block *layout* + numeric (de)quant — the substrate's concern.
## Parsing a `.gguf` container's headers/metadata is a loader's job, one layer up.)

import std/math
import ./float16

# Q4_K's dequant is `d·q − m`: Apple clang's default -ffp-contract=fast would
# fuse that into one fma (single rounding) and break bit-exactness against the
# gguf-py oracle (plain numpy: two roundings). Same policy as simd/dequant.
{.localPassc: "-ffp-contract=off".}

const QK* = 32 ## ggml block length for Q8_0 / Q4_0

type
  BlockQ8_0* = object
    d*: F16 ## fp16 block scale
    qs*: array[QK, int8] ## 32 quantized weights

  BlockQ4_0* = object
    d*: F16 ## fp16 block scale
    qs*: array[QK div 2, uint8] ## 16 bytes = 32 nibbles (offset-8 encoding)

# ---------------- dequantize (the load path) ----------------

proc dequantizeQ8_0*(blocks: openArray[BlockQ8_0], dst: var openArray[float32]) =
  ## value[i] = d · q[i]. `dst.len` must be `blocks.len * QK`.
  assert dst.len == blocks.len * QK
  var o = 0
  for blk in blocks:
    let d = blk.d.toFloat32
    for i in 0 ..< QK:
      dst[o + i] = d * float32(blk.qs[i])
    o += QK

proc dequantizeQ4_0*(blocks: openArray[BlockQ4_0], dst: var openArray[float32]) =
  ## value[i] = d · (nibble[i] − 8). Low nibble of byte i → lane i, high → lane i+16.
  assert dst.len == blocks.len * QK
  var o = 0
  for blk in blocks:
    let d = blk.d.toFloat32
    for i in 0 ..< QK div 2:
      let b = blk.qs[i]
      dst[o + i] = d * float32(int(b and 0x0f'u8) - 8)
      dst[o + i + QK div 2] = d * float32(int(b shr 4) - 8)
    o += QK

# ---------------- quantize (ggml reference algorithm) ----------------

proc quantizeQ8_0*(src: openArray[float32], blocks: var openArray[BlockQ8_0]) =
  ## d = amax/127 per block; q[i] = round(x[i]/d). `src.len` must be `blocks.len*QK`.
  assert src.len == blocks.len * QK
  var o = 0
  for blk in blocks.mitems:
    var amax = 0.0'f32
    for i in 0 ..< QK:
      let a = abs(src[o + i])
      if a > amax:
        amax = a
    let d = amax / 127.0'f32
    let id = (if d != 0.0'f32: 1.0'f32 / d else: 0.0'f32)
    blk.d = toF16(d)
    for i in 0 ..< QK:
      blk.qs[i] = int8(clamp(round(src[o + i] * id), -127.0'f32, 127.0'f32))
    o += QK

proc quantizeQ4_0*(src: openArray[float32], blocks: var openArray[BlockQ4_0]) =
  ## ggml Q4_0: pick the largest-|·| element KEEPING its sign, d = max/−8;
  ## q[i] = clamp(round(x[i]/d) + 8, 0, 15). `src.len` must be `blocks.len*QK`.
  assert src.len == blocks.len * QK
  var o = 0
  for blk in blocks.mitems:
    var amax = 0.0'f32
    var vmax = 0.0'f32
    for i in 0 ..< QK:
      let v = src[o + i]
      if abs(v) > amax:
        amax = abs(v)
        vmax = v
    let d = vmax / -8.0'f32
    let id = (if d != 0.0'f32: 1.0'f32 / d else: 0.0'f32)
    blk.d = toF16(d)
    for i in 0 ..< QK div 2:
      let q0 = clamp(int(round(src[o + i] * id)) + 8, 0, 15)
      let q1 = clamp(int(round(src[o + i + QK div 2] * id)) + 8, 0, 15)
      blk.qs[i] = uint8(q0 or (q1 shl 4))
    o += QK

# ---------------- k-quant super-blocks (Q4_K, Q6_K) ----------------
#
# The 256-element super-block formats real GGUF models actually ship in
# (Q4_0/Q8_0 are legacy). DEQUANTIZE only — the ingestion direction — ported
# from ggml-quants.c and diff-tested BIT-EXACT against gguf-py's independent
# numpy implementation (tests/test_scheme_conformance.nim). The quantize
# direction (ggml's two-stage scale search) is deliberately not implemented:
# gguf-py provides no oracle for it, and loading models never needs it.

const QK_K* = 256 ## k-quant super-block length

type
  BlockQ4_K* = object
    d*: F16 ## super-block scale for the sub-scales
    dmin*: F16 ## super-block scale for the sub-mins
    scales*: array[12, uint8] ## 8 sub-scales + 8 sub-mins, 6-bit packed
    qs*: array[QK_K div 2, uint8] ## 128 bytes = 256 nibbles

  BlockQ6_K* = object
    ql*: array[QK_K div 2, uint8] ## lower 4 bits of the quants
    qh*: array[QK_K div 4, uint8] ## upper 2 bits of the quants
    scales*: array[QK_K div 16, int8] ## 16 sub-scales, 8-bit
    d*: F16 ## super-block scale

static:
  doAssert sizeof(BlockQ4_K) == 144 and sizeof(BlockQ6_K) == 210

func getScaleMinK4(j: int, q: array[12, uint8]): (uint8, uint8) {.inline.} =
  ## The 6-bit sub-scale/sub-min unpack (ggml-quants.c get_scale_min_k4).
  if j < 4:
    (q[j] and 63'u8, q[j + 4] and 63'u8)
  else:
    (
      (q[j + 4] and 0x0f'u8) or ((q[j - 4] shr 6) shl 4),
      (q[j + 4] shr 4) or ((q[j] shr 6) shl 4),
    )

proc dequantizeQ4_K*(blocks: openArray[BlockQ4_K], dst: var openArray[float32]) =
  ## value = (d·sc)·nibble − (dmin·m), 64-element chunks sharing a (sc, m) pair
  ## per 32 lanes. `dst.len` must be `blocks.len * QK_K`.
  assert dst.len == blocks.len * QK_K
  var o = 0
  for bi in 0 ..< blocks.len:
    let d = blocks[bi].d.toFloat32
    let dmin = blocks[bi].dmin.toFloat32
    var q = 0 # byte index into qs
    var isv = 0
    var j = 0
    while j < QK_K:
      let (sc1, m1) = getScaleMinK4(isv, blocks[bi].scales)
      let d1 = d * float32(sc1)
      let mm1 = dmin * float32(m1)
      let (sc2, m2) = getScaleMinK4(isv + 1, blocks[bi].scales)
      let d2 = d * float32(sc2)
      let mm2 = dmin * float32(m2)
      for l in 0 ..< 32:
        dst[o + l] = d1 * float32(blocks[bi].qs[q + l] and 0x0f'u8) - mm1
      for l in 0 ..< 32:
        dst[o + 32 + l] = d2 * float32(blocks[bi].qs[q + l] shr 4) - mm2
      o += 64
      q += 32
      isv += 2
      j += 64

proc dequantizeQ6_K*(blocks: openArray[BlockQ6_K], dst: var openArray[float32]) =
  ## value = (d·scale)·(q − 32), 6-bit quants split across ql (low 4) and qh
  ## (high 2). `dst.len` must be `blocks.len * QK_K`.
  assert dst.len == blocks.len * QK_K
  var o = 0
  for bi in 0 ..< blocks.len:
    let d = blocks[bi].d.toFloat32
    var ql = 0
    var qh = 0
    var sc = 0
    var n = 0
    while n < QK_K:
      for l in 0 ..< 32:
        let isv = l div 16
        let q1 =
          int(
            int8(
              (blocks[bi].ql[ql + l] and 0x0f'u8) or
                (((blocks[bi].qh[qh + l] shr 0) and 3) shl 4)
            )
          ) - 32
        let q2 =
          int(
            int8(
              (blocks[bi].ql[ql + l + 32] and 0x0f'u8) or
                (((blocks[bi].qh[qh + l] shr 2) and 3) shl 4)
            )
          ) - 32
        let q3 =
          int(
            int8(
              (blocks[bi].ql[ql + l] shr 4) or
                (((blocks[bi].qh[qh + l] shr 4) and 3) shl 4)
            )
          ) - 32
        let q4 =
          int(
            int8(
              (blocks[bi].ql[ql + l + 32] shr 4) or
                (((blocks[bi].qh[qh + l] shr 6) and 3) shl 4)
            )
          ) - 32
        dst[o + l] = (d * float32(blocks[bi].scales[sc + isv])) * float32(q1)
        dst[o + l + 32] = (d * float32(blocks[bi].scales[sc + isv + 2])) * float32(q2)
        dst[o + l + 64] = (d * float32(blocks[bi].scales[sc + isv + 4])) * float32(q3)
        dst[o + l + 96] = (d * float32(blocks[bi].scales[sc + isv + 6])) * float32(q4)
      o += 128
      ql += 64
      qh += 32
      sc += 8
      n += 128
