## nim_lowprec/bfloat16 — bfloat16 primitives for ML inference.
##
## `BF16` stores the upper 16 bits of an IEEE-754 `float32`: 1 sign bit,
## 8 exponent bits, 7 mantissa bits. It trades mantissa precision for the
## full float32 dynamic range, which is why it dominates ML training and
## inference numerics.

type
  BF16* = distinct uint16
    ## A bfloat16 value: the high 16 bits of a `float32`.

func bits*(x: BF16): uint16 {.inline.} =
  ## Raw 16-bit storage of `x`.
  uint16(x)

func toFloat32*(x: BF16): float32 {.inline.} =
  ## Widen a `BF16` to `float32`. Lossless — bf16 is a strict subset.
  let u = uint32(uint16(x)) shl 16
  cast[float32](u)

func toBF16*(x: float32): BF16 {.inline.} =
  ## Narrow a `float32` to `BF16` with round-to-nearest-even.
  ## NaN inputs are preserved as a quiet NaN.
  let u = cast[uint32](x)
  if (u and 0x7fff_ffff'u32) > 0x7f80_0000'u32:
    # NaN: force the top mantissa bit so it stays NaN after truncation.
    return BF16(uint16((u shr 16) or 0x0040'u32) or 0x7fc0'u16)
  # Round-to-nearest-even on the 16 bits we are about to drop.
  let rounding = 0x7fff'u32 + ((u shr 16) and 1'u32)
  BF16(uint16((u + rounding) shr 16))

func `$`*(x: BF16): string =
  ## Render via the widened float32 value.
  $x.toFloat32

func `==`*(a, b: BF16): bool {.inline.} =
  a.toFloat32 == b.toFloat32

func `+`*(a, b: BF16): BF16 {.inline.} = toBF16(a.toFloat32 + b.toFloat32)
func `-`*(a, b: BF16): BF16 {.inline.} = toBF16(a.toFloat32 - b.toFloat32)
func `*`*(a, b: BF16): BF16 {.inline.} = toBF16(a.toFloat32 * b.toFloat32)
func `/`*(a, b: BF16): BF16 {.inline.} = toBF16(a.toFloat32 / b.toFloat32)
