import std/math
import ../dtypes

type
  I8* = distinct int8
  I4* = distinct int8
  I1* = distinct uint8

template defInt8Backed(T, toFn, BITS, DT: untyped) =
  func toFloat32*(x: T): float32 {.inline.} =
    float32(int8(x))
  func `==`*(a, b: T): bool {.inline.} =
    int8(a) == int8(b)
  func `$`*(x: T): string =
    $int8(x)
  func decode*(x: T): float32 {.inline.} =
    x.toFloat32
  func encode*(f: float32, _: typedesc[T]): T {.inline.} =
    toFn(f)
  func storageBits*(_: typedesc[T]): int {.inline.} =
    BITS
  func dtypeCode*(_: typedesc[T]): DType {.inline.} =
    DT

func toI8*(f: float32): I8 {.inline.} =
  I8(int8(clamp(round(f), -128.0'f32, 127.0'f32)))
defInt8Backed(I8, toI8, 8, dtI8)

func value*(x: I4): int {.inline.} =
  int(int8(x))
func toI4*(f: float32): I4 {.inline.} =
  I4(int8(clamp(round(f), -8.0'f32, 7.0'f32)))
func nibble*(x: I4): uint8 {.inline.} =
  uint8(int8(x)) and 0x0f'u8
func fromNibble*(n: uint8): I4 {.inline.} =
  let v = n and 0x0f'u8
  I4(
    if v >= 8'u8:
      int8(int(v) - 16)
    else:
      int8(v)
  )
defInt8Backed(I4, toI4, 4, dtI4)

func toFloat32*(x: I1): float32 {.inline.} =
  (if (uint8(x) and 1'u8) != 0'u8: -1.0'f32 else: 1.0'f32)
func toI1*(f: float32): I1 {.inline.} =
  I1(if (cast[uint32](f) shr 31) != 0'u32: 1'u8 else: 0'u8)
func decode*(x: I1): float32 {.inline.} =
  x.toFloat32
func encode*(f: float32, _: typedesc[I1]): I1 {.inline.} =
  toI1(f)
func storageBits*(_: typedesc[I1]): int {.inline.} =
  1
func dtypeCode*(_: typedesc[I1]): DType {.inline.} =
  dtI1

func packInt4*(src: openArray[I4], dst: var openArray[byte]) =
  assert dst.len >= (src.len + 1) div 2
  var i = 0
  var j = 0
  while i + 1 < src.len:
    dst[j] = nibble(src[i]) or (nibble(src[i + 1]) shl 4)
    i += 2
    inc j
  if i < src.len: # odd tail: low nibble only, high nibble 0
    dst[j] = nibble(src[i])

func unpackInt4*(src: openArray[byte], dst: var openArray[I4]) =
  var j = 0
  for b in src:
    if j < dst.len:
      dst[j] = fromNibble(b and 0x0f'u8)
      inc j
    if j < dst.len:
      dst[j] = fromNibble(b shr 4)
      inc j

func packInt1*(src: openArray[I1], dst: var openArray[byte]) =
  assert dst.len >= (src.len + 7) div 8
  for k in 0 ..< dst.len:
    dst[k] = 0'u8
  for i in 0 ..< src.len:
    if (uint8(src[i]) and 1'u8) != 0'u8:
      dst[i shr 3] = dst[i shr 3] or (1'u8 shl uint8(i and 7))

func unpackInt1*(src: openArray[byte], dst: var openArray[I1]) =
  for i in 0 ..< dst.len:
    dst[i] = I1((src[i shr 3] shr uint8(i and 7)) and 1'u8)
