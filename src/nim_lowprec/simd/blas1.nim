## nim_lowprec/simd/blas1 — vectorized BLAS-1 for the bandwidth-bound hot dtype (bf16).
##
## fp32-accumulate `dot` over narrow bf16 storage; NEON + AVX2 + scalar fallback,
## selected by `-d:lpSimd=`. A reduction, so the fp summation order differs from
## the scalar `reduce.dot` — checked against an fp64 reference within a tolerance,
## not bit-for-bit. The bf16 widen is the same "shift the 16 stored bits into the
## fp32 high half" used by simd/convert, so no new intrinsics are needed.

import ../bfloat16
import ./target
when lpUseNeon: import ./neon
when lpUseAvx2: import ./x86

proc dotBf16*(a, b: openArray[BF16]): float32 =
  ## Σ decode(aᵢ)·decode(bᵢ), accumulated in fp32. 8 lanes/iteration.
  assert a.len == b.len
  let n = a.len
  var i = 0
  when lpUseNeon:
    let sh16 = vdupq_n_s32(16'i32)
    var acc = vdupq_n_f32(0.0'f32)
    while i + 8 <= n:
      let av = vld1q_u16(cast[ptr uint16](unsafeAddr a[i]))
      let bv = vld1q_u16(cast[ptr uint16](unsafeAddr b[i]))
      acc = vfmaq_f32(acc,
        vreinterpretq_f32_u32(vshlq_u32(vmovl_u16(vget_low_u16(av)), sh16)),
        vreinterpretq_f32_u32(vshlq_u32(vmovl_u16(vget_low_u16(bv)), sh16)))
      acc = vfmaq_f32(acc,
        vreinterpretq_f32_u32(vshlq_u32(vmovl_u16(vget_high_u16(av)), sh16)),
        vreinterpretq_f32_u32(vshlq_u32(vmovl_u16(vget_high_u16(bv)), sh16)))
      i += 8
    result = vaddvq_f32(acc)
  elif lpUseAvx2:
    let sh16 = mm_cvtsi32_si128(16.cint)
    var acc = mm256_setzero_ps()
    while i + 8 <= n:
      let av = mm_loadu_si128(cast[ptr m128i](unsafeAddr a[i]))
      let bv = mm_loadu_si128(cast[ptr m128i](unsafeAddr b[i]))
      let af = mm256_castsi256_ps(mm256_sll_epi32(mm256_cvtepu16_epi32(av), sh16))
      let bf = mm256_castsi256_ps(mm256_sll_epi32(mm256_cvtepu16_epi32(bv), sh16))
      acc = mm256_fmadd_ps(af, bf, acc)
      i += 8
    result = hsum256(acc)
  while i < n:                                   # scalar tail (and the whole scalar build)
    result += toFloat32(a[i]) * toFloat32(b[i]); inc i
