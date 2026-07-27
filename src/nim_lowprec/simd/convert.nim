## nim_lowprec/simd/convert — vectorized bf16 <-> fp32 and fp16 <-> fp32 batch conversion.
##
## Hand-written NEON (arm_neon.h) + AVX2/F16C (immintrin.h) with a scalar
## fallback, selected at compile time by `-d:lpSimd=` (see simd/target). The
## vector paths reproduce the scalar reference EXACTLY (bf16: RNE + NaN-canonical;
## fp16: hardware vcvt / F16C), so they are bit-identical — enforced by the test
## (SIMD batch == scalar, element for element) on every ISA the suite runs on.
##
## Memory-bandwidth-bound streaming kernels. NEON is baseline on arm64; x86 picks
## its path at build time and CI verifies it. bf16<->fp32: 8 lanes widen / 4 lanes
## narrow. fp16<->fp32: F16C `_mm256_cvtph_ps` / `_mm256_cvtps_ph`, 8 lanes.

import ../bfloat16, ../float16, ../float8
import ./target
export target          # re-export simdBackend (+ the lpUse* selection flags)
when lpUseNeon: import ./neon
when lpUseAvx2: import ./x86

proc toFloat32Batch*(src: openArray[BF16]; dst: var openArray[float32]) =
  ## dst[i] = widen(src[i]). Exact. NEON/AVX2 do 8 lanes/iteration.
  assert src.len == dst.len
  let n = src.len
  when lpUseNeon:
    let sh16 = vdupq_n_s32(16'i32)
    var i = 0
    while i + 8 <= n:
      let v  = vld1q_u16(cast[ptr uint16](unsafeAddr src[i]))
      let lo = vshlq_u32(vmovl_u16(vget_low_u16(v)),  sh16)
      let hi = vshlq_u32(vmovl_u16(vget_high_u16(v)), sh16)
      vst1q_f32(addr dst[i],     vreinterpretq_f32_u32(lo))
      vst1q_f32(addr dst[i + 4], vreinterpretq_f32_u32(hi))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i]); inc i
  elif lpUseAvx2:
    # zero-extend 8×u16 -> 8×u32, << 16, reinterpret as fp32. 8 lanes/iteration.
    let sh16 = mm_cvtsi32_si128(16.cint)
    var i = 0
    while i + 8 <= n:
      let v  = mm_loadu_si128(cast[ptr m128i](unsafeAddr src[i]))
      let w  = mm256_cvtepu16_epi32(v)
      mm256_storeu_ps(addr dst[i], mm256_castsi256_ps(mm256_sll_epi32(w, sh16)))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i]); inc i
  else:
    for i in 0 ..< n: dst[i] = toFloat32(src[i])

proc toBF16Batch*(src: openArray[float32]; dst: var openArray[BF16]) =
  ## dst[i] = narrow(src[i]), RNE + NaN-safe. NEON/AVX2 do 4 lanes/iteration.
  assert src.len == dst.len
  let n = src.len
  when lpUseNeon:
    # 4 lanes/iter: round-to-nearest-even bf16 (add 0x7fff + LSB of the kept
    # half, then >>16) with a NaN-canonical override (qNaN 0x7fc0, sign kept).
    # This is the reference the AVX2 branch below mirrors lane-for-lane.
    let shNeg16 = vdupq_n_s32(-16'i32)
    let one   = vdupq_n_u32(1'u32)
    let c7fff = vdupq_n_u32(0x7fff'u32)
    let cAbs  = vdupq_n_u32(0x7fffffff'u32)
    let cInf  = vdupq_n_u32(0x7f800000'u32)
    let c8000 = vdupq_n_u32(0x8000'u32)
    let c7fc0 = vdupq_n_u32(0x7fc0'u32)
    var i = 0
    while i + 4 <= n:
      let u    = vreinterpretq_u32_f32(vld1q_f32(unsafeAddr src[i]))
      let hi16 = vshlq_u32(u, shNeg16)
      let lsb  = vandq_u32(hi16, one)
      let rounded = vshlq_u32(vaddq_u32(u, vaddq_u32(c7fff, lsb)), shNeg16)
      let nanMask = vcgtq_u32(vandq_u32(u, cAbs), cInf)
      let nanOut  = vorrq_u32(c7fc0, vandq_u32(hi16, c8000))
      let res     = vbslq_u32(nanMask, nanOut, rounded)
      vst1_u16(cast[ptr uint16](addr dst[i]), vmovn_u32(res))
      i += 4
    while i < n:
      dst[i] = toBF16(src[i]); inc i
  elif lpUseAvx2:
    # 4 lanes/iter, 128-bit — mirrors the NEON logic (RNE + NaN-canonical), then
    # packs the low 16 bits of each 32-bit lane into 4×u16 via a byte shuffle.
    let cnt16 = mm_cvtsi32_si128(16.cint)
    let one   = mm_set1_epi32(1)
    let c7fff = mm_set1_epi32(0x7fff)
    let cAbs  = mm_set1_epi32(0x7fffffff)
    let cInf  = mm_set1_epi32(0x7f800000)
    let c8000 = mm_set1_epi32(0x8000)
    let c7fc0 = mm_set1_epi32(0x7fc0)
    var msk: array[16, byte] = [byte 0,1,4,5,8,9,12,13, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80]
    let packMask = mm_loadu_si128(cast[ptr m128i](addr msk[0]))
    var i = 0
    while i + 4 <= n:
      let u    = mm_castps_si128(mm_loadu_ps(unsafeAddr src[i]))
      let hi16 = mm_srl_epi32(u, cnt16)
      let lsb  = mm_and_si128(hi16, one)
      let rounded = mm_srl_epi32(mm_add_epi32(u, mm_add_epi32(c7fff, lsb)), cnt16)
      let nanMask = mm_cmpgt_epi32(mm_and_si128(u, cAbs), cInf)
      let nanOut  = mm_or_si128(c7fc0, mm_and_si128(hi16, c8000))
      let res     = mm_blendv_epi8(rounded, nanOut, nanMask)
      mm_storel_epi64(cast[ptr m128i](addr dst[i]), mm_shuffle_epi8(res, packMask))
      i += 4
    while i < n:
      dst[i] = toBF16(src[i]); inc i
  else:
    for i in 0 ..< n: dst[i] = toBF16(src[i])

proc toFloat32Batch*(src: openArray[F16]; dst: var openArray[float32]) =
  ## fp16 -> fp32 via hardware vcvt (NEON) / F16C (AVX2). Exact.
  assert src.len == dst.len
  let n = src.len
  when lpUseNeon:
    var i = 0
    while i + 4 <= n:
      let h = vreinterpret_f16_u16(vld1_u16(cast[ptr uint16](unsafeAddr src[i])))
      vst1q_f32(addr dst[i], vcvt_f32_f16(h))
      i += 4
    while i < n:
      dst[i] = toFloat32(src[i]); inc i
  elif lpUseAvx2:
    var i = 0
    while i + 8 <= n:
      let h = mm_loadu_si128(cast[ptr m128i](unsafeAddr src[i]))   # 8×fp16 bits
      mm256_storeu_ps(addr dst[i], mm256_cvtph_ps(h))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i]); inc i
  else:
    for i in 0 ..< n: dst[i] = toFloat32(src[i])

proc toF16Batch*(src: openArray[float32]; dst: var openArray[F16]) =
  ## fp32 -> fp16 via hardware vcvt (NEON) / F16C (AVX2), RNE. NaN → hw-canonical.
  assert src.len == dst.len
  let n = src.len
  when lpUseNeon:
    var i = 0
    while i + 4 <= n:
      let h = vcvt_f16_f32(vld1q_f32(unsafeAddr src[i]))
      vst1_u16(cast[ptr uint16](addr dst[i]), vreinterpret_u16_f16(h))
      i += 4
    while i < n:
      dst[i] = toF16(src[i]); inc i
  elif lpUseAvx2:
    var i = 0
    while i + 8 <= n:
      let h = mm256_cvtps_ph(mm256_loadu_ps(unsafeAddr src[i]), 0)  # RNE
      mm_storeu_si128(cast[ptr m128i](addr dst[i]), h)
      i += 8
    while i < n:
      dst[i] = toF16(src[i]); inc i
  else:
    for i in 0 ..< n: dst[i] = toF16(src[i])

proc toFloat32Batch*(src: openArray[F8E5M2]; dst: var openArray[float32]) =
  ## fp8 e5m2 -> fp32. e5m2 IS the top 8 bits of an fp16, so (byte<<8) reinterpreted
  ## as fp16 is exactly the value; hardware fp16->fp32 finishes it. 8 lanes/iter.
  assert src.len == dst.len
  let n = src.len
  when lpUseNeon:
    let sh8 = vdupq_n_s16(8'i16)
    var i = 0
    while i + 8 <= n:
      let f = vreinterpretq_f16_u16(vshlq_u16(vmovl_u8(vld1_u8(cast[ptr uint8](unsafeAddr src[i]))), sh8))
      vst1q_f32(addr dst[i],     vcvt_f32_f16(vget_low_f16(f)))
      vst1q_f32(addr dst[i + 4], vcvt_f32_f16(vget_high_f16(f)))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i]); inc i
  elif lpUseAvx2:
    let sh8 = mm_cvtsi32_si128(8.cint)
    var i = 0
    while i + 8 <= n:
      let u = mm_sll_epi16(mm_cvtepu8_epi16(mm_loadl_epi64(cast[ptr m128i](unsafeAddr src[i]))), sh8)
      mm256_storeu_ps(addr dst[i], mm256_cvtph_ps(u))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i]); inc i
  else:
    for i in 0 ..< n: dst[i] = toFloat32(src[i])

proc toFloat32Batch*(src: openArray[F8E4M3]; dst: var openArray[float32]) =
  ## fp8 e4m3fn -> fp32. e4m3 has 4 exp bits vs fp16's 5, so repack s/e/m into an
  ## fp16 field ((byte&0x80)<<8 for the sign, (byte&0x7f)<<7 for exp+mantissa),
  ## hardware-convert, then ×256 to fix the bias (fp16 15 vs e4m3 7). The e4m3fn
  ## NaN slot ((byte&0x7f)==0x7f) is forced to an fp16 NaN pre-convert (×256·NaN=NaN).
  assert src.len == dst.len
  let n = src.len
  when lpUseNeon:
    let c80 = vdupq_n_u16(0x80'u16)
    let c7f = vdupq_n_u16(0x7f'u16)
    let c7e00 = vdupq_n_u16(0x7e00'u16)
    let sh8 = vdupq_n_s16(8'i16)
    let sh7 = vdupq_n_s16(7'i16)
    var i = 0
    while i + 8 <= n:
      let u = vmovl_u8(vld1_u8(cast[ptr uint8](unsafeAddr src[i])))
      let low7 = vandq_u16(u, c7f)
      let sPart = vshlq_u16(vandq_u16(u, c80), sh8)
      let rest  = vshlq_u16(low7, sh7)
      let nanB  = vandq_u16(vceqq_u16(low7, c7f), c7e00)
      let f = vreinterpretq_f16_u16(vorrq_u16(vorrq_u16(sPart, rest), nanB))
      vst1q_f32(addr dst[i],     vmulq_n_f32(vcvt_f32_f16(vget_low_f16(f)),  256.0'f32))
      vst1q_f32(addr dst[i + 4], vmulq_n_f32(vcvt_f32_f16(vget_high_f16(f)), 256.0'f32))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i]); inc i
  elif lpUseAvx2:
    let c80 = mm_set1_epi16(0x80)
    let c7f = mm_set1_epi16(0x7f)
    let c7e00 = mm_set1_epi16(0x7e00)
    let sh8 = mm_cvtsi32_si128(8.cint)
    let sh7 = mm_cvtsi32_si128(7.cint)
    let scale = mm256_set1_ps(256.0'f32)
    var i = 0
    while i + 8 <= n:
      let u = mm_cvtepu8_epi16(mm_loadl_epi64(cast[ptr m128i](unsafeAddr src[i])))
      let low7 = mm_and_si128(u, c7f)
      let sPart = mm_sll_epi16(mm_and_si128(u, c80), sh8)
      let rest  = mm_sll_epi16(low7, sh7)
      let nanB  = mm_and_si128(mm_cmpeq_epi16(low7, c7f), c7e00)
      let fbits = mm_or_si128(mm_or_si128(sPart, rest), nanB)
      mm256_storeu_ps(addr dst[i], mm256_mul_ps(mm256_cvtph_ps(fbits), scale))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i]); inc i
  else:
    for i in 0 ..< n: dst[i] = toFloat32(src[i])

proc toFloat32Batch*(src: openArray[F8E4M3FNUZ]; dst: var openArray[float32]) =
  ## fp8 e4m3fnuz (AMD) -> fp32. Same s/e/m repack as e4m3, but bias 8 → ×128, and
  ## the ONLY NaN is 0x80 (fnuz: no Inf, no −0, one NaN). The 0x7F slot is FINITE
  ## here (unlike OCP e4m3), so it is not masked.
  assert src.len == dst.len
  let n = src.len
  when lpUseNeon:
    let c80 = vdupq_n_u16(0x80'u16)
    let c7f = vdupq_n_u16(0x7f'u16)
    let c7e00 = vdupq_n_u16(0x7e00'u16)
    let sh8 = vdupq_n_s16(8'i16)
    let sh7 = vdupq_n_s16(7'i16)
    var i = 0
    while i + 8 <= n:
      let u = vmovl_u8(vld1_u8(cast[ptr uint8](unsafeAddr src[i])))
      let sPart = vshlq_u16(vandq_u16(u, c80), sh8)
      let rest  = vshlq_u16(vandq_u16(u, c7f), sh7)
      let nanB  = vandq_u16(vceqq_u16(u, c80), c7e00)          # NaN iff byte == 0x80
      let f = vreinterpretq_f16_u16(vorrq_u16(vorrq_u16(sPart, rest), nanB))
      vst1q_f32(addr dst[i],     vmulq_n_f32(vcvt_f32_f16(vget_low_f16(f)),  128.0'f32))
      vst1q_f32(addr dst[i + 4], vmulq_n_f32(vcvt_f32_f16(vget_high_f16(f)), 128.0'f32))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i]); inc i
  elif lpUseAvx2:
    let c80 = mm_set1_epi16(0x80)
    let c7f = mm_set1_epi16(0x7f)
    let c7e00 = mm_set1_epi16(0x7e00)
    let sh8 = mm_cvtsi32_si128(8.cint)
    let sh7 = mm_cvtsi32_si128(7.cint)
    let scale = mm256_set1_ps(128.0'f32)
    var i = 0
    while i + 8 <= n:
      let u = mm_cvtepu8_epi16(mm_loadl_epi64(cast[ptr m128i](unsafeAddr src[i])))
      let sPart = mm_sll_epi16(mm_and_si128(u, c80), sh8)
      let rest  = mm_sll_epi16(mm_and_si128(u, c7f), sh7)
      let nanB  = mm_and_si128(mm_cmpeq_epi16(u, c80), c7e00)
      let fbits = mm_or_si128(mm_or_si128(sPart, rest), nanB)
      mm256_storeu_ps(addr dst[i], mm256_mul_ps(mm256_cvtph_ps(fbits), scale))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i]); inc i
  else:
    for i in 0 ..< n: dst[i] = toFloat32(src[i])
