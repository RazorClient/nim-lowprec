## nim_lowprec/mxfloat — OCP Microscaling (MX) element formats + the E8M0 scale.
##
##   F4E2M1   MXFP4  (1/2/1, bias 1)  values 0,±0.5,±1,±1.5,±2,±3,±4,±6
##   F6E2M3   MXFP6  (1/2/3, bias 1)  max 7.5
##   F6E3M2   MXFP6  (1/3/2, bias 3)  max 28
##   E8M0     shared scale  value = 2^(b-127), 0x00 = 2^-127, 0xFF = NaN, no sign/zero
##
## The three float formats are FINITE-only (no Inf, no NaN): encode is round-to-
## nearest-even and SATURATES on overflow (7.0 → 6.0 in fp4), unlike fp8-e4m3fn
## which rounds overflow into its NaN slot — these formats simply have no NaN slot.
## E8M0 is the per-block power-of-two scale of the MX spec (paired with MXFPk
## elements in `quant`); its encode rounds to the nearest 2^k (linear midpoint
## 1.5·2^k, ties up).
##
## All four are verified bit-exact against ml_dtypes (float4_e2m1fn /
## float6_e2m3fn / float6_e3m2fn / float8_e8m0fnu).

import std/math
import ../common, ./tinyfloat

type
  F4E2M1* = distinct uint8 ## MXFP4 e2m1 (value in low 4 bits)
  F6E2M3* = distinct uint8 ## MXFP6 e2m3 (value in low 6 bits)
  F6E3M2* = distinct uint8 ## MXFP6 e3m2 (value in low 6 bits)
  E8M0* = distinct uint8 ## MX shared scale: 2^(b-127), 0xFF = NaN

  MxFmt = object
    ebits, mbits, bias, cmax: int ## cmax = largest positive finite code

const
  fmtF4E2M1 = MxFmt(ebits: 2, mbits: 1, bias: 1, cmax: 0x07)
  fmtF6E2M3 = MxFmt(ebits: 2, mbits: 3, bias: 1, cmax: 0x1f)
  fmtF6E3M2 = MxFmt(ebits: 3, mbits: 2, bias: 3, cmax: 0x1f)

# ---- format-specific edges layered on the shared tinyfloat core ----

func mxMag(c: int, f: MxFmt): float64 {.inline.} =
  ## Positive magnitude of finite code `c`, keyed by this format's bit layout.
  tinyMag(c, f.ebits, f.mbits, f.bias)

func mxDecode(u: uint8, f: MxFmt): float32 =
  let sbit = 1 shl (f.ebits + f.mbits)
  let neg = (int(u) and sbit) != 0
  let mag = mxMag(int(u) and (sbit - 1), f)
  float32(
    if neg:
      -mag
    else:
      mag
  )

func mxEncode(x: float32, f: MxFmt): uint8 =
  let sbit = uint8(1 shl (f.ebits + f.mbits))
  let xb = cast[uint32](x)
  let sign = (if (xb shr 31) != 0'u32: sbit else: 0'u8)
  let a = abs(x.float64)
  if a != a:
    return 0'u8 # NaN: unrepresentable → 0 (sane default)
  let vmax = mxMag(f.cmax, f)
  if a >= vmax:
    return sign or uint8(f.cmax) # SATURATE (also catches +Inf)
  # nearest finite code in [0, cmax], round-to-nearest-even (shared core)
  let code = nearestCode(a, f.ebits, f.mbits, f.bias, f.cmax)
  if code == 0:
    return sign # rounds to zero
  sign or uint8(code)

# ---- generate the three float formats' API surface via one template ----

template defMx(T, toFn, FMT, BITS, DT: untyped) =
  func bits*(x: T): uint8 {.inline.} =
    uint8(x)
  func toFloat32*(x: T): float32 {.inline.} =
    mxDecode(uint8(x), FMT)
  func toFn*(x: float32): T {.inline.} =
    T(mxEncode(x, FMT))
  func toFn*(x: float64): T {.inline.} =
    toFn(x.float32)
  func isNaN*(x: T): bool {.inline.} =
    false # finite-only
  func isInf*(x: T): bool {.inline.} =
    false
  func signbit*(x: T): bool {.inline.} =
    (uint8(x) and uint8(1 shl (FMT.ebits + FMT.mbits))) != 0'u8
  defFloatOps(T, toFn, BITS, DT)

defMx(F4E2M1, toF4E2M1, fmtF4E2M1, 4, dtF4E2M1)
defMx(F6E2M3, toF6E2M3, fmtF6E2M3, 6, dtF6E2M3)
defMx(F6E3M2, toF6E3M2, fmtF6E3M2, 6, dtF6E3M2)

# ---- E8M0 shared scale (pure exponent; no sign, no zero, 0xFF = NaN) ----

func bits*(x: E8M0): uint8 {.inline.} =
  uint8(x)

func toFloat32*(x: E8M0): float32 {.inline.} =
  let u = uint8(x)
  if u == 0xff'u8:
    return cast[float32](0x7fc0_0000'u32) # NaN
  if u == 0x00'u8:
    return cast[float32](0x0040_0000'u32) # 2^-127 (subnormal)
  cast[float32](uint32(u) shl 23) # 2^(u-127)

func toE8M0*(x: float32): E8M0 =
  ## Round to nearest power of two (linear midpoint 1.5·2^k, ties up).
  ## 0 / Inf / NaN → 0xFF; underflow → 0x00; overflow → 0xFF. Bit-exact vs
  ## ml_dtypes over all NORMAL positive floats; for the out-of-domain case of
  ## float32-SUBNORMAL inputs (~1e-39) we round true-nearest, where ml_dtypes
  ## rounds up — E8M0 is a scale and never sees such inputs in practice.
  let a = abs(x.float64)
  if x != x or a == 0.0 or a == Inf:
    return E8M0(0xff'u8)
  let (frac, exp) = frexp(a) # a = frac·2^exp, frac ∈ [0.5, 1)
  let er = (if frac >= 0.75: exp else: exp - 1)
  let code = er + 127
  if code > 254:
    E8M0(0xff'u8)
  elif code < 0:
    E8M0(0x00'u8)
  else:
    E8M0(uint8(code))

func toE8M0*(x: float64): E8M0 {.inline.} =
  toE8M0(x.float32)
func isNaN*(x: E8M0): bool {.inline.} =
  uint8(x) == 0xff'u8
func decode*(x: E8M0): float32 {.inline.} =
  x.toFloat32
func encode*(f: float32, _: typedesc[E8M0]): E8M0 {.inline.} =
  toE8M0(f)
func storageBits*(_: typedesc[E8M0]): int {.inline.} =
  8
func dtypeCode*(_: typedesc[E8M0]): DType {.inline.} =
  dtE8M0

# ---- packing ----

func packF4*(src: openArray[F4E2M1], dst: var openArray[byte]) =
  ## Pack MXFP4 codes two per byte — LOW nibble = even index (ggml order).
  ## `dst.len` must be at least `(src.len + 1) div 2`.
  assert dst.len >= (src.len + 1) div 2
  var i = 0
  var j = 0
  while i + 1 < src.len:
    dst[j] = (uint8(src[i]) and 0x0f'u8) or ((uint8(src[i + 1]) and 0x0f'u8) shl 4)
    i += 2
    inc j
  if i < src.len:
    dst[j] = uint8(src[i]) and 0x0f'u8

func unpackF4*(src: openArray[byte], dst: var openArray[F4E2M1]) =
  ## Inverse of `packF4` — two values per source byte.
  var j = 0
  for b in src:
    if j < dst.len:
      dst[j] = F4E2M1(b and 0x0f'u8)
      inc j
    if j < dst.len:
      dst[j] = F4E2M1(b shr 4)
      inc j

func packF6*[T: F6E2M3 | F6E3M2](src: openArray[T], dst: var openArray[byte]) =
  ## Tight-pack MXFP6 6-bit codes — FOUR values (4×6 = 24 bits) per THREE bytes,
  ## LSB-first. Within each little-endian 24-bit group: value0 → bits[5:0],
  ## value1 → bits[11:6], value2 → bits[17:12], value3 → bits[23:18]. The three
  ## bytes are that group's low, middle and high octet. A ragged tail (src.len
  ## not a multiple of 4) packs the values it has, with the unused high bits 0,
  ## occupying only the bytes those values need. Works for both F6E2M3 and
  ## F6E3M2 (both are 6-bit codes in the low 6 bits). `dst.len` must be at least
  ## `(src.len * 6 + 7) div 8`.
  assert dst.len >= (src.len * 6 + 7) div 8
  var i = 0 # value index
  var j = 0 # byte index
  while i < src.len:
    let n = min(4, src.len - i) # values in this group (1..4)
    var g = 0'u32
    for k in 0 ..< n:
      g = g or ((uint32(uint8(src[i + k])) and 0x3f'u32) shl (6 * k))
    let nbytes = (n * 6 + 7) div 8 # 1,2,3 → 1,2,3 bytes; 4 → 3 bytes
    for b in 0 ..< nbytes:
      dst[j + b] = uint8((g shr (8 * b)) and 0xff'u32)
    i += n
    j += nbytes

func unpackF6*[T: F6E2M3 | F6E3M2](src: openArray[byte], dst: var openArray[T]) =
  ## Inverse of `packF6` — reconstruct `dst.len` 6-bit codes from the three-byte
  ## groups. Reads only the bytes each (possibly ragged) group needs; never reads
  ## past `src`.
  var i = 0 # byte index
  var j = 0 # value index
  while j < dst.len:
    let n = min(4, dst.len - j) # values in this group (1..4)
    let nbytes = (n * 6 + 7) div 8
    var g = 0'u32
    for b in 0 ..< nbytes:
      if i + b < src.len:
        g = g or (uint32(src[i + b]) shl (8 * b))
    for k in 0 ..< n:
      dst[j + k] = T(uint8((g shr (6 * k)) and 0x3f'u32))
    i += nbytes
    j += n
