import std/math
import ../common, ./tinyfloat

type
  F8E4M3* = distinct uint8 # OCP fp8 e4m3fn
  F8E5M2* = distinct uint8 # OCP fp8 e5m2
  F8E4M3FNUZ* = distinct uint8 # AMD/ROCm fp8 e4m3fnuz
  F8E5M2FNUZ* = distinct uint8 # AMD/ROCm fp8 e5m2fnuz

  Fp8Fmt = object
    ebits, mbits, bias, cmax: int # cmax = largest positive FINITE code
    hasInf, fnuz: bool

const
  fmtE4M3 = Fp8Fmt(ebits: 4, mbits: 3, bias: 7, cmax: 0x7E, hasInf: false, fnuz: false)
  fmtE5M2 = Fp8Fmt(ebits: 5, mbits: 2, bias: 15, cmax: 0x7B, hasInf: true, fnuz: false)
  fmtE4M3FNUZ =
    Fp8Fmt(ebits: 4, mbits: 3, bias: 8, cmax: 0x7F, hasInf: false, fnuz: true)
  fmtE5M2FNUZ =
    Fp8Fmt(ebits: 5, mbits: 2, bias: 16, cmax: 0x7F, hasInf: false, fnuz: true)

func mag8(c: int, f: Fp8Fmt): float64 {.inline.} =
  # Positive magnitude of finite code `c`, keyed by this format's bit layout.
  tinyMag(c, f.ebits, f.mbits, f.bias)

func decode8(u: uint8, f: Fp8Fmt): float32 =
  let neg = (u and 0x80'u8) != 0'u8
  let ef = int((u shr f.mbits) and uint8((1 shl f.ebits) - 1))
  let mf = int(u and uint8((1 shl f.mbits) - 1))
  let maxE = (1 shl f.ebits) - 1
  # specials
  if f.fnuz:
    if u == 0x80'u8:
      return cast[float32](0x7fc0_0000'u32) # the sole NaN
  elif f.hasInf:
    if ef == maxE:
      if mf == 0:
        return cast[float32](if neg: 0xff80_0000'u32 else: 0x7f80_0000'u32)
      else:
        return cast[float32](0x7fc0_0000'u32)
  else:
    if ef == maxE and mf == (1 shl f.mbits) - 1:
      return cast[float32](0x7fc0_0000'u32)
  # finite value
  let mag = mag8(int(u and 0x7f'u8), f)
  float32(
    if neg:
      -mag
    else:
      mag
  )

func encode8(x: float32, f: Fp8Fmt): uint8 =
  let xb = cast[uint32](x)
  let signBit = uint8((xb shr 31) and 1'u32) shl 7
  let absb = xb and 0x7fff_ffff'u32
  # NaN input
  if absb > 0x7f80_0000'u32:
    if f.fnuz:
      return 0x80'u8
    elif f.hasInf:
      return signBit or 0x7e'u8 # e5m2 quiet NaN
    else:
      return signBit or 0x7f'u8 # e4m3fn NaN
  let a = abs(x.float64) # Inf stays Inf
  let vmax = mag8(f.cmax, f)
  let eCmax = (f.cmax shr f.mbits) and ((1 shl f.ebits) - 1)
  let mid = vmax + pow2(eCmax - f.bias - f.mbits) * 0.5 # vmax + ulp(cmax)/2
  # past the top boundary → Inf else NaN — NOT saturation
  if a > mid or (a == mid and (f.cmax and 1) == 1):
    if f.hasInf:
      return signBit or uint8(((1 shl f.ebits) - 1) shl f.mbits)
    elif f.fnuz:
      return 0x80'u8
    else:
      return signBit or 0x7f'u8
  if a >= vmax:
    return signBit or uint8(f.cmax)
  # nearest finite code in [0, cmax], round-to-nearest-even
  let code = nearestCode(a, f.ebits, f.mbits, f.bias, f.cmax)
  if code == 0:
    return (
      if f.fnuz: 0x00'u8 else: signBit # rounds to zero (fnuz has no -0)
    )

  signBit or uint8(code)

template defF8(T, toFn, FMT, DT: untyped) =
  func bits*(x: T): uint8 {.inline.} =
    uint8(x)
  func toFloat32*(x: T): float32 {.inline.} =
    decode8(uint8(x), FMT)
  func toFn*(x: float32): T {.inline.} =
    T(encode8(x, FMT))
  func toFn*(x: float64): T {.inline.} =
    toFn(x.float32)
  func isNaN*(x: T): bool {.inline.} =
    classify(x.toFloat32) == fcNan
  func isInf*(x: T): bool {.inline.} =
    classify(x.toFloat32) in {fcInf, fcNegInf}
  func signbit*(x: T): bool {.inline.} =
    (uint8(x) and 0x80'u8) != 0'u8
  defFloatOps(T, toFn, 8, DT)

defF8(F8E4M3, toF8E4M3, fmtE4M3, dtF8E4M3)
defF8(F8E5M2, toF8E5M2, fmtE5M2, dtF8E5M2)
defF8(F8E4M3FNUZ, toF8E4M3FNUZ, fmtE4M3FNUZ, dtF8E4M3FNUZ)
defF8(F8E5M2FNUZ, toF8E5M2FNUZ, fmtE5M2FNUZ, dtF8E5M2FNUZ)
