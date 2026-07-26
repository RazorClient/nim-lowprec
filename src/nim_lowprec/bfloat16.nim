import ./dtypes

type
  BF16* = distinct uint16
    ## A bfloat16 value: the high 16 bits of a `float32`.

const
  bf16Zero*   = BF16(0x0000'u16)
  bf16One*    = BF16(0x3f80'u16)
  bf16Inf*    = BF16(0x7f80'u16)
  bf16NegInf* = BF16(0xff80'u16)
  bf16NaN*    = BF16(0x7fc0'u16)   # canonical quiet NaN (positive)
  bf16NegNaN* = BF16(0xffc0'u16)   # canonical quiet NaN (negative)

func bits*(x: BF16): uint16 {.inline.} =
  uint16(x)

func toFloat32*(x: BF16): float32 {.inline.} =
  cast[float32](uint32(uint16(x)) shl 16)

func toBF16*(x: float32): BF16 {.inline.} =
  let u = cast[uint32](x)
  if (u and 0x7fff_ffff'u32) > 0x7f80_0000'u32:
    # NaN → canonical quiet NaN, sign-preserving. A naive truncation of the low
    # 16 bits could drop the only set mantissa bits and turn a NaN into Inf.
    # Emitting 0x7FC0/0xFFC0 also matches Eigen / ml_dtypes bfloat16, so the
    # result stays bit-exact against the golden oracle.
    return (if (u and 0x8000_0000'u32) != 0'u32: bf16NegNaN else: bf16NaN)
  # Round-to-nearest-even: bias by 0x7FFF, plus 1 when the kept bit is odd.
  let rounding = 0x7fff'u32 + ((u shr 16) and 1'u32)
  BF16(uint16((u + rounding) shr 16))

func toBF16Trunc*(x: float32): BF16 {.inline.} =
  let u = cast[uint32](x)
  if (u and 0x7fff_ffff'u32) > 0x7f80_0000'u32:
    return (if (u and 0x8000_0000'u32) != 0'u32: bf16NegNaN else: bf16NaN)
  BF16(uint16(u shr 16))

func isNaN*(x: BF16): bool {.inline.} = (uint16(x) and 0x7fff'u16) > 0x7f80'u16
func isInf*(x: BF16): bool {.inline.} = (uint16(x) and 0x7fff'u16) == 0x7f80'u16
func signbit*(x: BF16): bool {.inline.} = (uint16(x) and 0x8000'u16) != 0'u16

func `$`*(x: BF16): string = $x.toFloat32
func `==`*(a, b: BF16): bool {.inline.} = a.toFloat32 == b.toFloat32
func `<`*(a, b: BF16): bool {.inline.}  = a.toFloat32 <  b.toFloat32
func `<=`*(a, b: BF16): bool {.inline.} = a.toFloat32 <= b.toFloat32

func `+`*(a, b: BF16): BF16 {.inline.} = toBF16(a.toFloat32 + b.toFloat32)
func `-`*(a, b: BF16): BF16 {.inline.} = toBF16(a.toFloat32 - b.toFloat32)
func `*`*(a, b: BF16): BF16 {.inline.} = toBF16(a.toFloat32 * b.toFloat32)
func `/`*(a, b: BF16): BF16 {.inline.} = toBF16(a.toFloat32 / b.toFloat32)

func decode*(x: BF16): float32 {.inline.} = x.toFloat32
func encode*(f: float32; _: typedesc[BF16]): BF16 {.inline.} = toBF16(f)
func storageBits*(_: typedesc[BF16]): int {.inline.} = 16
func dtypeCode*(_: typedesc[BF16]): DType {.inline.} = dtBF16
