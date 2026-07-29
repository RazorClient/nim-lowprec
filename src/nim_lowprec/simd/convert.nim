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
export target # re-export simdBackend (+ the lpUse* selection flags)
when lpUseNeon:
  import ./neon
when lpUseAvx2:
  import ./x86

# The kernels below run with bounds/overflow checks OFF regardless of build mode.
#
# Nim's -d:release keeps both checks enabled (only -d:danger removes them), and in
# these inner loops they are not a safety net but the primary cost: the L1-resident
# SDOT GEMV measured 32 GFLOP/s with checks and 138 without — a 4.3x tax, larger
# than every architectural improvement in this module combined. Every proc here
# asserts its shape preconditions on entry (still active outside -d:release), the
# loop bounds are derived from those same lengths, and the conformance tests pin
# each kernel bit-for-bit — which exercises exactly this checks-off code, since the
# test suite compiles without -d:release.
{.push boundChecks: off, overflowChecks: off.}

proc toFloat32Batch*(src: openArray[BF16], dst: var openArray[float32]) =
  ## dst[i] = widen(src[i]). Exact. NEON/AVX2 do 8 lanes/iteration.
  assert src.len == dst.len
  let n = src.len
  when lpUseNeon:
    let sh16 = vdupq_n_s32(16'i32)
    var i = 0
    while i + 8 <= n:
      let v = vld1q_u16(cast[ptr uint16](unsafeAddr src[i]))
      let lo = vshlq_u32(vmovl_u16(vget_low_u16(v)), sh16)
      let hi = vshlq_u32(vmovl_u16(vget_high_u16(v)), sh16)
      vst1q_f32(addr dst[i], vreinterpretq_f32_u32(lo))
      vst1q_f32(addr dst[i + 4], vreinterpretq_f32_u32(hi))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i])
      inc i
  elif lpUseAvx2:
    # zero-extend 8×u16 -> 8×u32, << 16, reinterpret as fp32. 8 lanes/iteration.
    let sh16 = mm_cvtsi32_si128(16.cint)
    var i = 0
    while i + 8 <= n:
      let v = mm_loadu_si128(cast[ptr m128i](unsafeAddr src[i]))
      let w = mm256_cvtepu16_epi32(v)
      mm256_storeu_ps(addr dst[i], mm256_castsi256_ps(mm256_sll_epi32(w, sh16)))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i])
      inc i
  else:
    for i in 0 ..< n:
      dst[i] = toFloat32(src[i])

proc toBF16Batch*(src: openArray[float32], dst: var openArray[BF16]) =
  ## dst[i] = narrow(src[i]), RNE + NaN-safe. NEON/AVX2 do 4 lanes/iteration.
  assert src.len == dst.len
  let n = src.len
  when lpUseNeon:
    # 4 lanes/iter: round-to-nearest-even bf16 (add 0x7fff + LSB of the kept
    # half, then >>16) with a NaN-canonical override (qNaN 0x7fc0, sign kept).
    # This is the reference the AVX2 branch below mirrors lane-for-lane.
    let shNeg16 = vdupq_n_s32(-16'i32)
    let one = vdupq_n_u32(1'u32)
    let c7fff = vdupq_n_u32(0x7fff'u32)
    let cAbs = vdupq_n_u32(0x7fffffff'u32)
    let cInf = vdupq_n_u32(0x7f800000'u32)
    let c8000 = vdupq_n_u32(0x8000'u32)
    let c7fc0 = vdupq_n_u32(0x7fc0'u32)
    var i = 0
    while i + 4 <= n:
      let u = vreinterpretq_u32_f32(vld1q_f32(unsafeAddr src[i]))
      let hi16 = vshlq_u32(u, shNeg16)
      let lsb = vandq_u32(hi16, one)
      let rounded = vshlq_u32(vaddq_u32(u, vaddq_u32(c7fff, lsb)), shNeg16)
      let nanMask = vcgtq_u32(vandq_u32(u, cAbs), cInf)
      let nanOut = vorrq_u32(c7fc0, vandq_u32(hi16, c8000))
      let res = vbslq_u32(nanMask, nanOut, rounded)
      vst1_u16(cast[ptr uint16](addr dst[i]), vmovn_u32(res))
      i += 4
    while i < n:
      dst[i] = toBF16(src[i])
      inc i
  elif lpUseAvx2:
    # 4 lanes/iter, 128-bit — mirrors the NEON logic (RNE + NaN-canonical), then
    # packs the low 16 bits of each 32-bit lane into 4×u16 via a byte shuffle.
    let cnt16 = mm_cvtsi32_si128(16.cint)
    let one = mm_set1_epi32(1)
    let c7fff = mm_set1_epi32(0x7fff)
    let cAbs = mm_set1_epi32(0x7fffffff)
    let cInf = mm_set1_epi32(0x7f800000)
    let c8000 = mm_set1_epi32(0x8000)
    let c7fc0 = mm_set1_epi32(0x7fc0)
    var msk: array[16, byte] =
      [byte 0, 1, 4, 5, 8, 9, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]
    let packMask = mm_loadu_si128(cast[ptr m128i](addr msk[0]))
    var i = 0
    while i + 4 <= n:
      let u = mm_castps_si128(mm_loadu_ps(unsafeAddr src[i]))
      let hi16 = mm_srl_epi32(u, cnt16)
      let lsb = mm_and_si128(hi16, one)
      let rounded = mm_srl_epi32(mm_add_epi32(u, mm_add_epi32(c7fff, lsb)), cnt16)
      let nanMask = mm_cmpgt_epi32(mm_and_si128(u, cAbs), cInf)
      let nanOut = mm_or_si128(c7fc0, mm_and_si128(hi16, c8000))
      let res = mm_blendv_epi8(rounded, nanOut, nanMask)
      mm_storel_epi64(cast[ptr m128i](addr dst[i]), mm_shuffle_epi8(res, packMask))
      i += 4
    while i < n:
      dst[i] = toBF16(src[i])
      inc i
  else:
    for i in 0 ..< n:
      dst[i] = toBF16(src[i])

proc toFloat32Batch*(src: openArray[F16], dst: var openArray[float32]) =
  ## fp16 -> fp32 via hardware vcvt (NEON) / F16C (AVX2). Exact.
  assert src.len == dst.len
  let n = src.len
  when lpUseNeon:
    var i = 0
    # 16/iteration, 4 independent convert chains: at 4/iteration this loop
    # measured 4.2 Gelem/s against numpy's 10.2 — the ALU was idle between the
    # dependent load->convert->store triples.
    while i + 16 <= n:
      let u0 = vld1q_u16(cast[ptr uint16](unsafeAddr src[i]))
      let u1 = vld1q_u16(cast[ptr uint16](unsafeAddr src[i + 8]))
      let f0 = vreinterpretq_f16_u16(u0)
      let f1 = vreinterpretq_f16_u16(u1)
      vst1q_f32(addr dst[i], vcvt_f32_f16(vget_low_f16(f0)))
      vst1q_f32(addr dst[i + 4], vcvt_f32_f16(vget_high_f16(f0)))
      vst1q_f32(addr dst[i + 8], vcvt_f32_f16(vget_low_f16(f1)))
      vst1q_f32(addr dst[i + 12], vcvt_f32_f16(vget_high_f16(f1)))
      i += 16
    while i + 4 <= n:
      let h = vreinterpret_f16_u16(vld1_u16(cast[ptr uint16](unsafeAddr src[i])))
      vst1q_f32(addr dst[i], vcvt_f32_f16(h))
      i += 4
    while i < n:
      dst[i] = toFloat32(src[i])
      inc i
  elif lpUseAvx2:
    var i = 0
    while i + 8 <= n:
      let h = mm_loadu_si128(cast[ptr m128i](unsafeAddr src[i])) # 8×fp16 bits
      mm256_storeu_ps(addr dst[i], mm256_cvtph_ps(h))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i])
      inc i
  else:
    for i in 0 ..< n:
      dst[i] = toFloat32(src[i])

proc toF16Batch*(src: openArray[float32], dst: var openArray[F16]) =
  ## fp32 -> fp16 via hardware vcvt (NEON) / F16C (AVX2), RNE. NaN → hw-canonical.
  assert src.len == dst.len
  let n = src.len
  when lpUseNeon:
    var i = 0
    while i + 16 <= n: # 4 independent chains (see widen note above)
      let h0 = vcvt_f16_f32(vld1q_f32(unsafeAddr src[i]))
      let h1 = vcvt_f16_f32(vld1q_f32(unsafeAddr src[i + 4]))
      let h2 = vcvt_f16_f32(vld1q_f32(unsafeAddr src[i + 8]))
      let h3 = vcvt_f16_f32(vld1q_f32(unsafeAddr src[i + 12]))
      vst1_u16(cast[ptr uint16](addr dst[i]), vreinterpret_u16_f16(h0))
      vst1_u16(cast[ptr uint16](addr dst[i + 4]), vreinterpret_u16_f16(h1))
      vst1_u16(cast[ptr uint16](addr dst[i + 8]), vreinterpret_u16_f16(h2))
      vst1_u16(cast[ptr uint16](addr dst[i + 12]), vreinterpret_u16_f16(h3))
      i += 16
    while i + 4 <= n:
      let h = vcvt_f16_f32(vld1q_f32(unsafeAddr src[i]))
      vst1_u16(cast[ptr uint16](addr dst[i]), vreinterpret_u16_f16(h))
      i += 4
    while i < n:
      dst[i] = toF16(src[i])
      inc i
  elif lpUseAvx2:
    var i = 0
    while i + 8 <= n:
      let h = mm256_cvtps_ph(mm256_loadu_ps(unsafeAddr src[i]), 0) # RNE
      mm_storeu_si128(cast[ptr m128i](addr dst[i]), h)
      i += 8
    while i < n:
      dst[i] = toF16(src[i])
      inc i
  else:
    for i in 0 ..< n:
      dst[i] = toF16(src[i])

proc toFloat32Batch*(src: openArray[F8E5M2], dst: var openArray[float32]) =
  ## fp8 e5m2 -> fp32. e5m2 IS the top 8 bits of an fp16, so (byte<<8) reinterpreted
  ## as fp16 is exactly the value; hardware fp16->fp32 finishes it. 8 lanes/iter.
  assert src.len == dst.len
  let n = src.len
  when lpUseNeon:
    let sh8 = vdupq_n_s16(8'i16)
    var i = 0
    while i + 8 <= n:
      let f = vreinterpretq_f16_u16(
        vshlq_u16(vmovl_u8(vld1_u8(cast[ptr uint8](unsafeAddr src[i]))), sh8)
      )
      vst1q_f32(addr dst[i], vcvt_f32_f16(vget_low_f16(f)))
      vst1q_f32(addr dst[i + 4], vcvt_f32_f16(vget_high_f16(f)))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i])
      inc i
  elif lpUseAvx2:
    let sh8 = mm_cvtsi32_si128(8.cint)
    var i = 0
    while i + 8 <= n:
      let u = mm_sll_epi16(
        mm_cvtepu8_epi16(mm_loadl_epi64(cast[ptr m128i](unsafeAddr src[i]))), sh8
      )
      mm256_storeu_ps(addr dst[i], mm256_cvtph_ps(u))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i])
      inc i
  else:
    for i in 0 ..< n:
      dst[i] = toFloat32(src[i])

proc toFloat32Batch*(src: openArray[F8E4M3], dst: var openArray[float32]) =
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
      let rest = vshlq_u16(low7, sh7)
      let nanB = vandq_u16(vceqq_u16(low7, c7f), c7e00)
      let f = vreinterpretq_f16_u16(vorrq_u16(vorrq_u16(sPart, rest), nanB))
      vst1q_f32(addr dst[i], vmulq_n_f32(vcvt_f32_f16(vget_low_f16(f)), 256.0'f32))
      vst1q_f32(addr dst[i + 4], vmulq_n_f32(vcvt_f32_f16(vget_high_f16(f)), 256.0'f32))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i])
      inc i
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
      let rest = mm_sll_epi16(low7, sh7)
      let nanB = mm_and_si128(mm_cmpeq_epi16(low7, c7f), c7e00)
      let fbits = mm_or_si128(mm_or_si128(sPart, rest), nanB)
      mm256_storeu_ps(addr dst[i], mm256_mul_ps(mm256_cvtph_ps(fbits), scale))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i])
      inc i
  else:
    for i in 0 ..< n:
      dst[i] = toFloat32(src[i])

proc toFloat32Batch*(src: openArray[F8E4M3FNUZ], dst: var openArray[float32]) =
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
      let rest = vshlq_u16(vandq_u16(u, c7f), sh7)
      let nanB = vandq_u16(vceqq_u16(u, c80), c7e00) # NaN iff byte == 0x80
      let f = vreinterpretq_f16_u16(vorrq_u16(vorrq_u16(sPart, rest), nanB))
      vst1q_f32(addr dst[i], vmulq_n_f32(vcvt_f32_f16(vget_low_f16(f)), 128.0'f32))
      vst1q_f32(addr dst[i + 4], vmulq_n_f32(vcvt_f32_f16(vget_high_f16(f)), 128.0'f32))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i])
      inc i
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
      let rest = mm_sll_epi16(mm_and_si128(u, c7f), sh7)
      let nanB = mm_and_si128(mm_cmpeq_epi16(u, c80), c7e00)
      let fbits = mm_or_si128(mm_or_si128(sPart, rest), nanB)
      mm256_storeu_ps(addr dst[i], mm256_mul_ps(mm256_cvtph_ps(fbits), scale))
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i])
      inc i
  else:
    for i in 0 ..< n:
      dst[i] = toFloat32(src[i])

{.pop.}

proc toFloat32Batch*(src: openArray[F8E5M2FNUZ], dst: var openArray[float32]) =
  ## fp8 e5m2fnuz (AMD) -> fp32. e5m2fnuz is ALMOST the top byte of an fp16 —
  ## same field widths — but bias 16 instead of 15, so the f16 reinterpretation
  ## is exactly 2× the true value: multiply by 0.5. Two exceptions get fixed up
  ## lanewise: codes with an all-ones exponent field are FINITE here (f16 reads
  ## them as Inf/NaN), and 0x80 is the single fnuz NaN (f16 reads −0).
  assert src.len == dst.len
  let n = src.len
  when lpUseNeon:
    let sh8 = vdupq_n_s16(8'i16)
    let e31 = vdupq_n_u32(0x7c'u32)
    let m3 = vdupq_n_u32(0x03'u32)
    let s80 = vdupq_n_u32(0x80'u32)
    let four = vdupq_n_u32(4'u32)
    let sh13 = vdupq_n_s32(13'i32) # ×8192 for the (m+4)·8192 finite top values
    let sh24 = vdupq_n_s32(24'i32) # byte sign bit -> f32 sign bit
    let qnan = vdupq_n_u32(0x7fc0_0000'u32)
    var i = 0
    while i + 8 <= n:
      let u16v = vmovl_u8(vld1_u8(cast[ptr uint8](unsafeAddr src[i])))
      let f = vreinterpretq_f16_u16(vshlq_u16(u16v, sh8))
      template half(getHalfU, getHalfF, off: untyped) =
        let u32v = vmovl_u16(getHalfU(u16v))
        let normal =
          vreinterpretq_u32_f32(vmulq_n_f32(vcvt_f32_f16(getHalfF(f)), 0.5'f32))
        # finite all-ones-exponent codes: ±(m+4)·8192
        let topMask = vceqq_u32(vandq_u32(u32v, e31), e31)
        let mag = vreinterpretq_u32_f32(
          vcvtq_f32_u32(vshlq_u32(vaddq_u32(vandq_u32(u32v, m3), four), sh13))
        )
        let fixed = vorrq_u32(mag, vshlq_u32(vandq_u32(u32v, s80), sh24))
        var bits32 = vbslq_u32(topMask, fixed, normal)
        bits32 = vbslq_u32(vceqq_u32(u32v, s80), qnan, bits32) # 0x80 -> NaN
        vst1q_f32(addr dst[i + off], vreinterpretq_f32_u32(bits32))

      half(vget_low_u16, vget_low_f16, 0)
      half(vget_high_u16, vget_high_f16, 4)
      i += 8
    while i < n:
      dst[i] = toFloat32(src[i])
      inc i
  else:
    # scalar path (also the AVX2 build for now: this format is rare enough that
    # the x86 vector variant hasn't been written; correctness is identical)
    for i in 0 ..< n:
      dst[i] = toFloat32(src[i])

# ---------------- fp8 ENCODE (f32 -> fp8), via round-to-odd f16 + a 64 KB table ----------------
#
# There is no direct hardware path. The construction: convert f32 to f16, look
# the f16 code up in a table built from the scalar encoder over all 2^16 codes.
# The naive version of that is WRONG: hardware f32->f16 rounds to nearest-even,
# and a value just past an fp8 tie point collapses onto the tie (t is
# f16-representable, everything within half-ulp16 of it lands ON it), turning
# "round up" into "tie -> even" — the tests caught 128 such cases per format.
#
# The fix is Boldo–Melquiond: round f32 to f16 with ROUND-TO-ODD, then the
# table's round-to-nearest is exact (f16's 11 bits >= fp8's p+2). Round-to-odd
# is synthesized from the RNE result h: if the conversion was exact, keep h;
# otherwise take the truncated-toward-zero magnitude (h, minus one if RNE
# rounded away from zero) and force its last bit to 1. Exactness and direction
# come from two vector compares against the widened-back h.
#
# Values past the format's top boundary (`mid`, the scalar encoder's own
# constant) are pre-substituted with ±Inf so the table lands on the scalar's
# past-top code. Tables build lazily per thread ({.threadvar.}, ~1 ms each).

import ../float8 as f8mod

template defF8Encode(T, toFn, batchName: untyped, mid: static float32) =
  var lut {.threadvar.}: seq[uint8] # threadvar: no race, one build per thread

  proc batchName*(src: openArray[float32], dst: var openArray[T]) =
    ## f32 -> fp8, bit-identical to the scalar `toFn` for every input,
    ## including NaN, ±Inf, past-top and subnormal values.
    assert src.len == dst.len
    if lut.len == 0:
      lut = newSeq[uint8](65536)
      for h in 0 ..< 65536:
        lut[h] = uint8(toFn(toFloat32(F16(uint16(h)))))
    let n = src.len
    when lpUseNeon:
      let midv = vdupq_n_f32(mid)
      let infv = vdupq_n_u32(0x7f80_0000'u32)
      let signv = vdupq_n_u32(0x8000_0000'u32)
      var hbuf {.noinit.}: array[8, uint16]
      var eqbuf {.noinit.}: array[8, uint32]
      var awbuf {.noinit.}: array[8, uint32]
      var i = 0
      while i + 8 <= n:
        template prep(off: untyped) =
          let x0 = vld1q_f32(unsafeAddr src[i + off])
          let past = vcgtq_f32(vabsq_f32(x0), midv)
          let infS = vorrq_u32(infv, vandq_u32(vreinterpretq_u32_f32(x0), signv))
          let x =
            vreinterpretq_f32_u32(vbslq_u32(past, infS, vreinterpretq_u32_f32(x0)))
          let h = vcvt_f16_f32(x) # RNE
          let back = vcvt_f32_f16(h) # exact widen of h
          vst1_u16(addr hbuf[off], vreinterpret_u16_f16(h))
          vst1q_u32(addr eqbuf[off], vceqq_f32(back, x)) # conversion exact?
          vst1q_u32(addr awbuf[off], vcgtq_f32(vabsq_f32(back), vabsq_f32(x)))
            # rounded away?

        prep(0)
        prep(4)
        for j in 0 ..< 8:
          var h = hbuf[j]
          if eqbuf[j] == 0'u32: # inexact -> round-to-odd
            var mag = h and 0x7fff'u16
            if awbuf[j] != 0'u32:
              dec mag
              # undo the away-rounding
            h = (h and 0x8000'u16) or (mag or 1'u16)
          dst[i + j] = T(lut[h])
        i += 8
      while i < n:
        dst[i] = toFn(src[i])
        inc i
    else:
      for i in 0 ..< n:
        dst[i] = toFn(src[i])

# `mid` = the scalar encoder's past-top boundary: vmax + ulp(cmax)/2.
defF8Encode(F8E4M3, toF8E4M3, toF8E4M3Batch, 464.0'f32)
defF8Encode(F8E5M2, toF8E5M2, toF8E5M2Batch, 61440.0'f32)
defF8Encode(F8E4M3FNUZ, toF8E4M3FNUZ, toF8E4M3FNUZBatch, 248.0'f32)
defF8Encode(F8E5M2FNUZ, toF8E5M2FNUZ, toF8E5M2FNUZBatch, 61440.0'f32)
