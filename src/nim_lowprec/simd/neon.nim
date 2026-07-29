## nim_lowprec/simd/neon — shared arm_neon.h intrinsic bindings.
##
## Thin `importc` wrappers over the NEON intrinsics used by the SIMD kernels
## (`convert`, `dequant`). Everything is guarded by
## `when defined(arm64) or defined(aarch64)`, so on any other target this module
## is empty and the kernels compile their scalar fallbacks instead.
##
## The vector types are `{.bycopy.}` objects mirroring the arm_neon.h `*_t`
## types; the procs map 1:1 onto the intrinsics. Shifts use the variable-count
## `vshlq_*` form (the shift amount is supplied as a dup'd vector) rather than
## the immediate `_n_` intrinsics — this is deliberate, do not "simplify" it.

import ./target

when defined(arm64) or defined(aarch64):
  {.push header: "arm_neon.h".}

  type
    uint16x8*  {.importc: "uint16x8_t",  bycopy.} = object
    uint16x4*  {.importc: "uint16x4_t",  bycopy.} = object
    uint32x4*  {.importc: "uint32x4_t",  bycopy.} = object
    int32x4*   {.importc: "int32x4_t",   bycopy.} = object
    float32x4* {.importc: "float32x4_t", bycopy.} = object
    float16x4* {.importc: "float16x4_t", bycopy.} = object
    int8x16*   {.importc: "int8x16_t",   bycopy.} = object
    uint8x16*  {.importc: "uint8x16_t",  bycopy.} = object
    int8x8*    {.importc: "int8x8_t",    bycopy.} = object
    int16x8*   {.importc: "int16x8_t",   bycopy.} = object
    int16x4*   {.importc: "int16x4_t",   bycopy.} = object
    uint8x8*   {.importc: "uint8x8_t",   bycopy.} = object
    float16x8* {.importc: "float16x8_t", bycopy.} = object

  # A 2-way de-interleaving load: val[0] gets the even-indexed bytes, val[1] the
  # odd ones. That is exactly the layout packed nibbles want — low nibbles are the
  # even columns, high nibbles the odd ones — so the ACTIVATIONS can be split once
  # instead of the weights being zipped back into order per row.
  type int8x16x2* {.importc: "int8x16x2_t", bycopy.} = object
    val*: array[2, int8x16]

  # loads / stores
  proc vld2q_s8*(p: ptr int8): int8x16x2 {.importc.}
  proc vld1q_u16*(p: ptr uint16): uint16x8 {.importc.}
  proc vld1_u16*(p: ptr uint16): uint16x4 {.importc.}
  proc vld1q_f32*(p: ptr float32): float32x4 {.importc.}
  proc vld1q_s8*(p: ptr int8): int8x16 {.importc.}
  proc vld1q_s32*(p: ptr int32): int32x4 {.importc.}
  proc vld1q_u8*(p: ptr uint8): uint8x16 {.importc.}
  proc vld1_u8*(p: ptr uint8): uint8x8 {.importc.}
  proc vst1q_f32*(p: ptr float32; v: float32x4) {.importc.}
  proc vst1_u16*(p: ptr uint16; v: uint16x4) {.importc.}

  # lane extraction / widen / narrow
  proc vget_low_u16*(a: uint16x8): uint16x4 {.importc.}
  proc vget_high_u16*(a: uint16x8): uint16x4 {.importc.}
  proc vget_low_s8*(a: int8x16): int8x8 {.importc.}
  proc vget_high_s8*(a: int8x16): int8x8 {.importc.}
  proc vget_low_s16*(a: int16x8): int16x4 {.importc.}
  proc vget_high_s16*(a: int16x8): int16x4 {.importc.}
  proc vmovl_u8*(a: uint8x8): uint16x8 {.importc.}
  proc vmovl_u16*(a: uint16x4): uint32x4 {.importc.}
  proc vmovl_s8*(a: int8x8): int16x8 {.importc.}
  proc vmovl_s16*(a: int16x4): int32x4 {.importc.}
  proc vmovn_u32*(a: uint32x4): uint16x4 {.importc.}

  # broadcast (dup)
  proc vdupq_n_u32*(v: uint32): uint32x4 {.importc.}
  proc vdupq_n_s32*(v: int32): int32x4 {.importc.}
  proc vdupq_n_f32*(v: float32): float32x4 {.importc.}
  proc vdupq_n_u8*(v: uint8): uint8x16 {.importc.}
  proc vdupq_n_s8*(v: int8): int8x16 {.importc.}

  # arithmetic / convert / fma
  proc vaddq_u32*(a, b: uint32x4): uint32x4 {.importc.}
  proc vabsq_f32*(a: float32x4): float32x4 {.importc.}
  proc vmaxq_f32*(a, b: float32x4): float32x4 {.importc.}
  proc vmaxvq_f32*(a: float32x4): float32 {.importc.}          # horizontal max (aarch64)
  proc vdivq_f32*(a, b: float32x4): float32x4 {.importc.}      # lanewise divide (aarch64)
  proc vcvtaq_s32_f32*(a: float32x4): int32x4 {.importc.}      # round ties-AWAY -> int32 (matches Nim round())
  proc vminq_s32*(a, b: int32x4): int32x4 {.importc.}
  proc vmaxq_s32*(a, b: int32x4): int32x4 {.importc.}
  proc vmovn_s32*(a: int32x4): int16x4 {.importc.}             # narrow (low 16 bits)
  proc vmovn_s16*(a: int16x8): int8x8 {.importc.}
  proc vcombine_s16*(lo, hi: int16x4): int16x8 {.importc.}
  proc vcombine_s8*(lo, hi: int8x8): int8x16 {.importc.}
  proc vst1q_s8*(p: ptr int8; v: int8x16) {.importc.}
  proc vst1q_u8*(p: ptr uint8; v: uint8x16) {.importc.}
  proc vorrq_u8*(a, b: uint8x16): uint8x16 {.importc.}
  proc vreinterpretq_u8_s8*(a: int8x16): uint8x16 {.importc.}
  proc vsubq_s8*(a, b: int8x16): int8x16 {.importc.}   # per-lane int8 subtract (ggml Q4_0 nibble-8)
  proc vcvt_f32_f16*(a: float16x4): float32x4 {.importc.}   # fp16 -> fp32 (hardware)
  proc vcvt_f16_f32*(a: float32x4): float16x4 {.importc.}   # fp32 -> fp16 (hardware, RNE)
  proc vcvtq_f32_s32*(a: int32x4): float32x4 {.importc.}
  proc vmulq_n_f32*(a: float32x4; b: float32): float32x4 {.importc.}
  proc vfmaq_f32*(a, b, c: float32x4): float32x4 {.importc.}   # a + b*c
  proc vaddq_f32*(a, b: float32x4): float32x4 {.importc.}      # lanewise add (folding accumulators)
  proc vaddq_s32*(a, b: int32x4): int32x4 {.importc.}          # lanewise int32 add
  proc vpaddq_s32*(a, b: int32x4): int32x4 {.importc.}         # pairwise add across two vectors
  proc vmulq_f32*(a, b: float32x4): float32x4 {.importc.}      # lanewise fp32 multiply
  proc vaddvq_s32*(a: int32x4): int32 {.importc.}              # horizontal int32 sum (aarch64)
  proc vaddvq_f32*(a: float32x4): float32 {.importc.}          # horizontal sum (aarch64)

  # bitwise / shift / select
  proc vandq_u32*(a, b: uint32x4): uint32x4 {.importc.}
  proc vorrq_u32*(a, b: uint32x4): uint32x4 {.importc.}
  proc vcgtq_u32*(a, b: uint32x4): uint32x4 {.importc.}
  proc vceqq_u32*(a, b: uint32x4): uint32x4 {.importc.}
  proc vcvtq_f32_u32*(a: uint32x4): float32x4 {.importc.}
  proc vcgtq_f32*(a, b: float32x4): uint32x4 {.importc.}
  proc vceqq_f32*(a, b: float32x4): uint32x4 {.importc.}
  proc vst1q_u32*(p: ptr uint32; v: uint32x4) {.importc.}
  proc vbslq_u32*(mask, a, b: uint32x4): uint32x4 {.importc.}
  proc vshlq_u32*(a: uint32x4; b: int32x4): uint32x4 {.importc.}
  proc vandq_u8*(a, b: uint8x16): uint8x16 {.importc.}
  proc vshlq_u8*(a: uint8x16; b: int8x16): uint8x16 {.importc.}   # var shift (b<0 ⇒ right)
  proc vshlq_s8*(a, b: int8x16): int8x16 {.importc.}              # var shift (b<0 ⇒ arith right)

  # reinterpret (bit-casts)
  proc vreinterpret_f16_u16*(a: uint16x4): float16x4 {.importc.}
  proc vreinterpret_u16_f16*(a: float16x4): uint16x4 {.importc.}
  proc vreinterpretq_f32_u32*(a: uint32x4): float32x4 {.importc.}
  proc vreinterpretq_u32_f32*(a: float32x4): uint32x4 {.importc.}
  proc vreinterpretq_s8_u8*(a: uint8x16): int8x16 {.importc.}

  # zip (interleave)
  proc vzip1q_s8*(a, b: int8x16): int8x16 {.importc.}
  proc vzip2q_s8*(a, b: int8x16): int8x16 {.importc.}

  # table lookup (16-entry byte LUT; index lanes ≥ 16 yield 0)
  proc vqtbl1q_s8*(t: int8x16; idx: uint8x16): int8x16 {.importc.}

  # fp8-decode helpers: u16-lane ops + 8-lane fp16 handling
  proc vandq_u16*(a, b: uint16x8): uint16x8 {.importc.}
  proc vorrq_u16*(a, b: uint16x8): uint16x8 {.importc.}
  proc vceqq_u16*(a, b: uint16x8): uint16x8 {.importc.}          # per-lane a==b → 0xFFFF/0
  proc vshlq_u16*(a: uint16x8; b: int16x8): uint16x8 {.importc.} # var shift (b<0 ⇒ right)
  proc vdupq_n_u16*(v: uint16): uint16x8 {.importc.}
  proc vdupq_n_s16*(v: int16): int16x8 {.importc.}
  proc vreinterpretq_f16_u16*(a: uint16x8): float16x8 {.importc.}
  proc vget_low_f16*(a: float16x8): float16x4 {.importc.}
  proc vget_high_f16*(a: float16x8): float16x4 {.importc.}

  {.pop.}

# ---- ARMv8.2 dot product (FEAT_DotProd), gated separately from NEON itself ----
#
# 16 int8 multiply-accumulates into 4 int32 lanes, one instruction: lane j of the
# result gets acc[j] + Σ(a[4j+i] * b[4j+i]) for i in 0..3. Not in baseline ARMv8.0,
# hence the `lpUseDotProd` gate rather than the plain arm64 one.
when lpUseDotProd:
  {.push header: "arm_neon.h".}
  proc vdotq_s32*(acc: int32x4; a, b: int8x16): int32x4 {.importc.}
  {.pop.}
