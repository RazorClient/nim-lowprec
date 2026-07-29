## nim_lowprec/simd/blas1 — vectorized BLAS-1 for the bandwidth-bound hot dtype (bf16).
##
## fp32-accumulate `dot` over narrow bf16 storage; NEON + AVX2 + scalar fallback,
## selected by `-d:lpSimd=`. A reduction, so the fp summation order differs from
## the scalar `reduce.dot` — checked against an fp64 reference within a tolerance,
## not bit-for-bit. The bf16 widen is the same "shift the 16 stored bits into the
## fp32 high half" used by simd/convert, so no new intrinsics are needed.

import ../bfloat16
import ../float16
import ./target
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

proc dotBf16*(a, b: openArray[BF16]): float32 =
  ## Σ decode(aᵢ)·decode(bᵢ), accumulated in fp32. 8 lanes/iteration.
  assert a.len == b.len
  let n = a.len
  var i = 0
  # Two independent accumulators, not one: `vfmaq_f32`/`vfmadd` has ~4-cycle
  # latency but issues several per cycle, so a single accumulator makes each FMA
  # wait on the previous one and the loop runs at latency instead of throughput.
  # Only the summation order changes — this reduction is already tolerance-checked
  # against an fp64 reference, never bit-for-bit.
  when lpUseNeon:
    let sh16 = vdupq_n_s32(16'i32)
    var acc0 = vdupq_n_f32(0.0'f32)
    var acc1 = vdupq_n_f32(0.0'f32)
    while i + 8 <= n:
      let av = vld1q_u16(cast[ptr uint16](unsafeAddr a[i]))
      let bv = vld1q_u16(cast[ptr uint16](unsafeAddr b[i]))
      acc0 = vfmaq_f32(
        acc0,
        vreinterpretq_f32_u32(vshlq_u32(vmovl_u16(vget_low_u16(av)), sh16)),
        vreinterpretq_f32_u32(vshlq_u32(vmovl_u16(vget_low_u16(bv)), sh16)),
      )
      acc1 = vfmaq_f32(
        acc1,
        vreinterpretq_f32_u32(vshlq_u32(vmovl_u16(vget_high_u16(av)), sh16)),
        vreinterpretq_f32_u32(vshlq_u32(vmovl_u16(vget_high_u16(bv)), sh16)),
      )
      i += 8
    result = vaddvq_f32(vaddq_f32(acc0, acc1))
  elif lpUseAvx2:
    let sh16 = mm_cvtsi32_si128(16.cint)
    var acc0 = mm256_setzero_ps()
    var acc1 = mm256_setzero_ps()
    while i + 16 <= n: # 2×8 lanes, one per accumulator
      let av0 = mm_loadu_si128(cast[ptr m128i](unsafeAddr a[i]))
      let bv0 = mm_loadu_si128(cast[ptr m128i](unsafeAddr b[i]))
      let av1 = mm_loadu_si128(cast[ptr m128i](unsafeAddr a[i + 8]))
      let bv1 = mm_loadu_si128(cast[ptr m128i](unsafeAddr b[i + 8]))
      acc0 = mm256_fmadd_ps(
        mm256_castsi256_ps(mm256_sll_epi32(mm256_cvtepu16_epi32(av0), sh16)),
        mm256_castsi256_ps(mm256_sll_epi32(mm256_cvtepu16_epi32(bv0), sh16)),
        acc0,
      )
      acc1 = mm256_fmadd_ps(
        mm256_castsi256_ps(mm256_sll_epi32(mm256_cvtepu16_epi32(av1), sh16)),
        mm256_castsi256_ps(mm256_sll_epi32(mm256_cvtepu16_epi32(bv1), sh16)),
        acc1,
      )
      i += 16
    while i + 8 <= n: # odd 8-lane block, if any
      let av = mm_loadu_si128(cast[ptr m128i](unsafeAddr a[i]))
      let bv = mm_loadu_si128(cast[ptr m128i](unsafeAddr b[i]))
      acc0 = mm256_fmadd_ps(
        mm256_castsi256_ps(mm256_sll_epi32(mm256_cvtepu16_epi32(av), sh16)),
        mm256_castsi256_ps(mm256_sll_epi32(mm256_cvtepu16_epi32(bv), sh16)),
        acc0,
      )
      i += 8
    result = hsum256(mm256_add_ps(acc0, acc1))
  while i < n: # scalar tail (and the whole scalar build)
    result += toFloat32(a[i]) * toFloat32(b[i])
    inc i

proc dotF16*(a, b: openArray[F16]): float32 =
  ## Σ decode(aᵢ)·decode(bᵢ) over IEEE fp16, accumulated in fp32 — the hardware
  ## widen (vcvt / F16C) straight into the same two-accumulator FMA shape as
  ## `dotBf16`. Reduction: tolerance-tested against an fp64 reference.
  assert a.len == b.len
  let n = a.len
  var i = 0
  when lpUseNeon:
    var acc0 = vdupq_n_f32(0.0'f32)
    var acc1 = vdupq_n_f32(0.0'f32)
    while i + 8 <= n:
      let af = vreinterpretq_f16_u16(vld1q_u16(cast[ptr uint16](unsafeAddr a[i])))
      let bf = vreinterpretq_f16_u16(vld1q_u16(cast[ptr uint16](unsafeAddr b[i])))
      acc0 =
        vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(af)), vcvt_f32_f16(vget_low_f16(bf)))
      acc1 = vfmaq_f32(
        acc1, vcvt_f32_f16(vget_high_f16(af)), vcvt_f32_f16(vget_high_f16(bf))
      )
      i += 8
    result = vaddvq_f32(vaddq_f32(acc0, acc1))
  elif lpUseAvx2:
    var acc0 = mm256_setzero_ps()
    var acc1 = mm256_setzero_ps()
    while i + 16 <= n:
      let a0 = mm256_cvtph_ps(mm_loadu_si128(cast[ptr m128i](unsafeAddr a[i])))
      let b0 = mm256_cvtph_ps(mm_loadu_si128(cast[ptr m128i](unsafeAddr b[i])))
      let a1 = mm256_cvtph_ps(mm_loadu_si128(cast[ptr m128i](unsafeAddr a[i + 8])))
      let b1 = mm256_cvtph_ps(mm_loadu_si128(cast[ptr m128i](unsafeAddr b[i + 8])))
      acc0 = mm256_fmadd_ps(a0, b0, acc0)
      acc1 = mm256_fmadd_ps(a1, b1, acc1)
      i += 16
    while i + 8 <= n:
      let a0 = mm256_cvtph_ps(mm_loadu_si128(cast[ptr m128i](unsafeAddr a[i])))
      let b0 = mm256_cvtph_ps(mm_loadu_si128(cast[ptr m128i](unsafeAddr b[i])))
      acc0 = mm256_fmadd_ps(a0, b0, acc0)
      i += 8
    result = hsum256(mm256_add_ps(acc0, acc1))
  while i < n:
    result += toFloat32(a[i]) * toFloat32(b[i])
    inc i

{.pop.}
