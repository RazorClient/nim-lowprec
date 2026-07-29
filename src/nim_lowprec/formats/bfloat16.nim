## nim_lowprec/bfloat16 — bfloat16 (1/8/7): the high 16 bits of a float32.
##
## Encode is round-to-nearest-even with NaN-safe handling: a naive high-16
## truncation could drop the only set mantissa bits and alias a NaN to Inf, so
## NaN is mapped to a canonical, sign-preserving quiet NaN instead. Verified
## bit-exact against ml_dtypes bfloat16.

import ../common

type BF16* = distinct uint16 ## A bfloat16 value: the high 16 bits of a `float32`.

const
  bf16Zero* = BF16(0x0000'u16)
  bf16One* = BF16(0x3f80'u16)
  bf16Inf* = BF16(0x7f80'u16)
  bf16NegInf* = BF16(0xff80'u16)
  bf16NaN* = BF16(0x7fc0'u16) # canonical quiet NaN (positive)
  bf16NegNaN* = BF16(0xffc0'u16) # canonical quiet NaN (negative)

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

func toBF16Stochastic*(x: float32, rng: var uint32): BF16 =
  ## Stochastically round fp32 → bf16. The result is ALWAYS one of the two bf16
  ## values bracketing `x` (or `x` exactly when it is representable), and the one
  ## chosen with probability proportional to where `x` falls between them, so
  ## `E[result] ≈ x` (unbiased). The low 16 bits of `x` are dithered with a
  ## uniform 16-bit offset, then truncated: a carry into bit 16 (⇒ round up)
  ## happens with probability `frac/65536`, an exact split of the bracket.
  ##
  ## `rng` is an xorshift32 state threaded by the caller and updated in place —
  ## no globals, no side effects. Seed it with a NON-ZERO value (xorshift is
  ## stuck at 0). NaN/Inf fall back to the deterministic `toBF16` (stay NaN/Inf).
  let u = cast[uint32](x)
  if (u and 0x7fff_ffff'u32) >= 0x7f80_0000'u32:
    return toBF16(x) # NaN / Inf → deterministic special
  # advance xorshift32; the high 16 bits are the higher-quality dither
  var r = rng
  r = r xor (r shl 13)
  r = r xor (r shr 17)
  r = r xor (r shl 5)
  rng = r
  let dither = r shr 16 # uniform in [0, 65535]
  BF16(uint16((u + dither) shr 16))

func isNaN*(x: BF16): bool {.inline.} =
  (uint16(x) and 0x7fff'u16) > 0x7f80'u16
func isInf*(x: BF16): bool {.inline.} =
  (uint16(x) and 0x7fff'u16) == 0x7f80'u16
func signbit*(x: BF16): bool {.inline.} =
  (uint16(x) and 0x8000'u16) != 0'u16

# Comparisons, `$`, arithmetic and the LowPrec interface, all routed through
# toFloat32 / toBF16 — see common.defFloatOps.
defFloatOps(BF16, toBF16, 16, dtBF16)
