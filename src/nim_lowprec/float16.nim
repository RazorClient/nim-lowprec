import ./dtypes

type
  F16* = distinct uint16
    ## An IEEE-754 binary16 value.

const
  f16Zero*   = F16(0x0000'u16)
  f16One*    = F16(0x3c00'u16)
  f16Inf*    = F16(0x7c00'u16)
  f16NegInf* = F16(0xfc00'u16)
  f16NaN*    = F16(0x7e00'u16)   ## canonical quiet NaN (positive)

func bits*(x: F16): uint16 {.inline.} =
  ## Raw 16-bit storage of `x`.
  uint16(x)

func floatbitsFromHalf(h: uint16): uint32 =
  let h_exp = h and 0x7c00'u16
  let f_sgn = (uint32(h) and 0x8000'u32) shl 16
  case h_exp
  of 0x0000'u16:                              # zero or subnormal
    var h_sig = h and 0x03ff'u16
    if h_sig == 0'u16:
      return f_sgn                             # signed zero
    var e = 0'u32                              # normalize the subnormal
    h_sig = h_sig shl 1
    while (h_sig and 0x0400'u16) == 0'u16:
      h_sig = h_sig shl 1
      inc e
    let f_exp = (127'u32 - 15'u32 - e) shl 23
    let f_sig = uint32(h_sig and 0x03ff'u16) shl 13
    f_sgn + f_exp + f_sig
  of 0x7c00'u16:                              # inf or NaN (payload preserved)
    f_sgn + 0x7f80_0000'u32 + (uint32(h and 0x03ff'u16) shl 13)
  else:                                        # normalized
    f_sgn + ((uint32(h and 0x7fff'u16) + 0x1c000'u32) shl 13)

func toFloat32*(x: F16): float32 {.inline.} =
  # Widen a `F16` to `float32`.
  cast[float32](floatbitsFromHalf(uint16(x)))

func halfbitsFromFloat(f: uint32): uint16 =
  let h_sgn = uint16((f and 0x8000_0000'u32) shr 16)
  var f_exp = f and 0x7f80_0000'u32
  # exponent overflow / Inf / NaN
  if f_exp >= 0x4780_0000'u32:
    if f_exp == 0x7f80_0000'u32:
      let f_sig = f and 0x007f_ffff'u32
      if f_sig != 0'u32:                       # NaN — keep the flag, stay NaN
        var ret = uint16(0x7c00'u32 + (f_sig shr 13))
        if ret == 0x7c00'u16: inc ret
        return h_sgn + ret
      return h_sgn + 0x7c00'u16                # signed Inf
    return h_sgn + 0x7c00'u16                  # overflow to signed Inf
  # exponent underflow -> subnormal half or signed zero
  if f_exp <= 0x3800_0000'u32:
    if f_exp < 0x3300_0000'u32:
      return h_sgn                             # rounds to signed zero
    let fe = f_exp shr 23
    var f_sig = 0x0080_0000'u32 + (f and 0x007f_ffff'u32)
    f_sig = f_sig shr (113'u32 - fe)
    # round-to-nearest-even on the bit beyond half precision
    if ((f_sig and 0x0000_3fff'u32) != 0x0000_1000'u32) or ((f and 0x0000_07ff'u32) != 0'u32):
      f_sig += 0x0000_1000'u32
    return h_sgn + uint16(f_sig shr 13)
  # regular case
  let h_exp = uint16((f_exp - 0x3800_0000'u32) shr 13)
  var f_sig = f and 0x007f_ffff'u32
  if ((f_sig and 0x0000_3fff'u32) != 0x0000_1000'u32) or ((f and 0x0000_07ff'u32) != 0'u32):
    f_sig += 0x0000_1000'u32
  h_sgn + h_exp + uint16(f_sig shr 13)

func toF16*(x: float32): F16 {.inline.} =
  F16(halfbitsFromFloat(cast[uint32](x)))

func toF16*(x: float64): F16 {.inline.} = toF16(x.float32)

func isNaN*(x: F16): bool {.inline.} = (uint16(x) and 0x7fff'u16) > 0x7c00'u16
func isInf*(x: F16): bool {.inline.} = (uint16(x) and 0x7fff'u16) == 0x7c00'u16
func signbit*(x: F16): bool {.inline.} = (uint16(x) and 0x8000'u16) != 0'u16

func `$`*(x: F16): string = $x.toFloat32
func `==`*(a, b: F16): bool {.inline.} = a.toFloat32 == b.toFloat32
func `<`*(a, b: F16): bool {.inline.}  = a.toFloat32 <  b.toFloat32
func `<=`*(a, b: F16): bool {.inline.} = a.toFloat32 <= b.toFloat32

func `+`*(a, b: F16): F16 {.inline.} = toF16(a.toFloat32 + b.toFloat32)
func `-`*(a, b: F16): F16 {.inline.} = toF16(a.toFloat32 - b.toFloat32)
func `*`*(a, b: F16): F16 {.inline.} = toF16(a.toFloat32 * b.toFloat32)
func `/`*(a, b: F16): F16 {.inline.} = toF16(a.toFloat32 / b.toFloat32)

func decode*(x: F16): float32 {.inline.} = x.toFloat32
func encode*(f: float32; _: typedesc[F16]): F16 {.inline.} = toF16(f)
func storageBits*(_: typedesc[F16]): int {.inline.} = 16
func dtypeCode*(_: typedesc[F16]): DType {.inline.} = dtF16
