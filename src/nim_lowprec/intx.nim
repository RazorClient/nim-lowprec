## nim_lowprec/intx — small signed integers + bit/nibble packing for quantization.
##
##   I8   signed 8-bit   (-128..127), one byte each
##   I4   signed 4-bit   (-8..7),     packed two-per-byte, ggml LOW-nibble-first
##   I1   a sign bit      (+1 / -1),  packed eight-per-byte, LSB-first
##
## These are STORAGE + PACKING primitives only. The quant *scheme* (per-block
## offsets, scales, super-block layouts) lives one layer up — this module owns
## just the mechanical two's-complement values and the exact bit/nibble order.

import std/math
import ./dtypes

type
  I8* = distinct int8      ## signed 8-bit integer
  I4* = distinct int8      ## signed 4-bit integer, logical value in -8..7
  I1* = distinct uint8     ## a sign bit: 0 → +1, 1 → −1

# ---------- I8 ----------
func toFloat32*(x: I8): float32 {.inline.} = float32(int8(x))
func toI8*(f: float32): I8 {.inline.} =
  ## Round-to-nearest (ties away from zero) then clamp to the int8 range.
  I8(int8(clamp(round(f), -128.0'f32, 127.0'f32)))
func `==`*(a, b: I8): bool {.inline.} = int8(a) == int8(b)
func `$`*(x: I8): string = $int8(x)
func decode*(x: I8): float32 {.inline.} = x.toFloat32
func encode*(f: float32; _: typedesc[I8]): I8 {.inline.} = toI8(f)
func storageBits*(_: typedesc[I8]): int {.inline.} = 8
func dtypeCode*(_: typedesc[I8]): DType {.inline.} = dtI8

# ---------- I4 ----------
func value*(x: I4): int {.inline.} = int(int8(x))                    ## logical -8..7
func toFloat32*(x: I4): float32 {.inline.} = float32(int8(x))
func toI4*(f: float32): I4 {.inline.} =
  I4(int8(clamp(round(f), -8.0'f32, 7.0'f32)))
func nibble*(x: I4): uint8 {.inline.} = uint8(int8(x)) and 0x0f'u8   ## two's-complement 4-bit
func fromNibble*(n: uint8): I4 {.inline.} =
  ## Decode a 4-bit two's-complement nibble (0..15) → -8..7.
  let v = n and 0x0f'u8
  I4(if v >= 8'u8: int8(int(v) - 16) else: int8(v))
func `==`*(a, b: I4): bool {.inline.} = int8(a) == int8(b)
func `$`*(x: I4): string = $int8(x)
func decode*(x: I4): float32 {.inline.} = x.toFloat32
func encode*(f: float32; _: typedesc[I4]): I4 {.inline.} = toI4(f)
func storageBits*(_: typedesc[I4]): int {.inline.} = 4
func dtypeCode*(_: typedesc[I4]): DType {.inline.} = dtI4

# ---------- I1 (sign bit) ----------
func toFloat32*(x: I1): float32 {.inline.} =
  (if (uint8(x) and 1'u8) != 0'u8: -1.0'f32 else: 1.0'f32)
func toI1*(f: float32): I1 {.inline.} =
  ## Negative (including -0.0) → −1; otherwise +1.
  I1(if (cast[uint32](f) shr 31) != 0'u32: 1'u8 else: 0'u8)
func decode*(x: I1): float32 {.inline.} = x.toFloat32
func encode*(f: float32; _: typedesc[I1]): I1 {.inline.} = toI1(f)
func storageBits*(_: typedesc[I1]): int {.inline.} = 1
func dtypeCode*(_: typedesc[I1]): DType {.inline.} = dtI1

# ---------- packing ----------
func packInt4*(src: openArray[I4]; dst: var openArray[byte]) =
  ## Pack signed 4-bit values two per byte — LOW nibble = even index (ggml order).
  ## `dst.len` must be at least `(src.len + 1) div 2`.
  assert dst.len >= (src.len + 1) div 2
  var i = 0
  var j = 0
  while i + 1 < src.len:
    dst[j] = nibble(src[i]) or (nibble(src[i + 1]) shl 4)
    i += 2
    inc j
  if i < src.len:                        # odd tail: low nibble only, high nibble 0
    dst[j] = nibble(src[i])

func unpackInt4*(src: openArray[byte]; dst: var openArray[I4]) =
  ## Inverse of `packInt4` — fills `dst` (two values per source byte).
  var j = 0
  for b in src:
    if j < dst.len:
      dst[j] = fromNibble(b and 0x0f'u8)
      inc j
    if j < dst.len:
      dst[j] = fromNibble(b shr 4)
      inc j

func packInt1*(src: openArray[I1]; dst: var openArray[byte]) =
  ## Pack sign bits eight per byte, LSB-first (element 0 → bit 0).
  ## `dst.len` must be at least `(src.len + 7) div 8`.
  assert dst.len >= (src.len + 7) div 8
  for k in 0 ..< dst.len: dst[k] = 0'u8
  for i in 0 ..< src.len:
    if (uint8(src[i]) and 1'u8) != 0'u8:
      dst[i shr 3] = dst[i shr 3] or (1'u8 shl uint8(i and 7))

func unpackInt1*(src: openArray[byte]; dst: var openArray[I1]) =
  for i in 0 ..< dst.len:
    dst[i] = I1((src[i shr 3] shr uint8(i and 7)) and 1'u8)
