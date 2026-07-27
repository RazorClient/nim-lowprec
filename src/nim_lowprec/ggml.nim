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

const QK* = 32   ## ggml block length for Q8_0 / Q4_0

type
  BlockQ8_0* = object
    d*:  F16                       ## fp16 block scale
    qs*: array[QK, int8]           ## 32 quantized weights

  BlockQ4_0* = object
    d*:  F16                       ## fp16 block scale
    qs*: array[QK div 2, uint8]    ## 16 bytes = 32 nibbles (offset-8 encoding)

# ---------------- dequantize (the load path) ----------------

proc dequantizeQ8_0*(blocks: openArray[BlockQ8_0]; dst: var openArray[float32]) =
  ## value[i] = d · q[i]. `dst.len` must be `blocks.len * QK`.
  assert dst.len == blocks.len * QK
  var o = 0
  for blk in blocks:
    let d = blk.d.toFloat32
    for i in 0 ..< QK:
      dst[o + i] = d * float32(blk.qs[i])
    o += QK

proc dequantizeQ4_0*(blocks: openArray[BlockQ4_0]; dst: var openArray[float32]) =
  ## value[i] = d · (nibble[i] − 8). Low nibble of byte i → lane i, high → lane i+16.
  assert dst.len == blocks.len * QK
  var o = 0
  for blk in blocks:
    let d = blk.d.toFloat32
    for i in 0 ..< QK div 2:
      let b = blk.qs[i]
      dst[o + i]            = d * float32(int(b and 0x0f'u8) - 8)
      dst[o + i + QK div 2] = d * float32(int(b shr 4) - 8)
    o += QK

# ---------------- quantize (ggml reference algorithm) ----------------

proc quantizeQ8_0*(src: openArray[float32]; blocks: var openArray[BlockQ8_0]) =
  ## d = amax/127 per block; q[i] = round(x[i]/d). `src.len` must be `blocks.len*QK`.
  assert src.len == blocks.len * QK
  var o = 0
  for blk in blocks.mitems:
    var amax = 0.0'f32
    for i in 0 ..< QK:
      let a = abs(src[o + i])
      if a > amax: amax = a
    let d = amax / 127.0'f32
    let id = (if d != 0.0'f32: 1.0'f32 / d else: 0.0'f32)
    blk.d = toF16(d)
    for i in 0 ..< QK:
      blk.qs[i] = int8(clamp(round(src[o + i] * id), -127.0'f32, 127.0'f32))
    o += QK

proc quantizeQ4_0*(src: openArray[float32]; blocks: var openArray[BlockQ4_0]) =
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
      let q0 = clamp(int(round(src[o + i]            * id)) + 8, 0, 15)
      let q1 = clamp(int(round(src[o + i + QK div 2] * id)) + 8, 0, 15)
      blk.qs[i] = uint8(q0 or (q1 shl 4))
    o += QK
