## nim_lowprec/simd/dequant — the quantized-numerics kernel module. Five families:
##
## ELEMENTWISE (bit-exact vs their scalar references):
##   dequantizeBatch          int8/int4 value·scale -> fp32 (per-group / per-tensor)
##   dequantizeQ8_0Batch/Q4_0Batch   ggml block dequant (hw fp16 scale decode, 4/iter)
##   quantizeQ8_0Batch/Q4_0Batch     ggml block QUANTIZE (ties-away = vcvtaq)
##   quantizeBatch            fp32 -> per-group int8 (the SDOT activation encode)
##
## fp32-ACTIVATION fused GEMV (reduction: fp64-reference, tolerance-tested):
##   dequantGemv / dequantGemvI4 / dequantGemvF4 / dequantGemvQ8_0 / dequantGemvQ4_0
##   — weights dequantised IN-REGISTER, never materialised as fp32.
##
## int8-ACTIVATION GEMV — SDOT on arm64, maddubs/madd on AVX2 (bit-exact vs an
## int64 reference; int32 group sums are exact):
##   dequantGemvQ8 / dequantGemvI4Q8            group-scale formats (the fast path;
##                                              tile or stream by -d:lpStreamBytes)
##   dequantGemvQ8_0Q8 / dequantGemvQ4_0Q8      ggml block layouts (GGUF without repack)
##
## MULTI-COLUMN GEMM (n>1 activations; each column bit-identical to its GEMV):
##   dequantGemmQ8 / dequantGemmI4Q8
##
## TESTING DISCIPLINE: elementwise and integer-accumulated kernels are pinned
## bit-for-bit; only the fp32-FMA reductions use a tolerance (their summation
## order legitimately differs from scalar). NEON + AVX2, scalar fallback elsewhere;
## see the pragma blocks below for why checks and fp contraction are pinned off.

import ../formats/intx, ../formats/float16
import ../quantization/quant, ../quantization/ggml
import ./target

# Apple clang defaults to -ffp-contract=fast, which silently fuses a vector
# multiply feeding a vector add into ONE fma (single rounding) — across statement
# boundaries. That broke bit-exactness against the reference by one ulp in one
# row out of five in the ggml-layout GEMV. Every fp contraction in this module is
# therefore explicit: where an FMA is wanted the code says vfmaq_f32/_mm256_fmadd,
# and the C compiler is not allowed to invent others.
{.localPassc: "-ffp-contract=off".}
when lpUseNeon:
  import ./neon
when lpUseAvx2:
  import ./x86

# 2× the MXFP4 (e2m1) value for each 4-bit code — all integers, so fp4 decode is
# a byte-LUT straight into the int8 widen+FMA path; the ×2 is undone by halving
# the block scale. Codes 8..15 are the negatives (code 8 = -0 → 0).
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

const mxfp4Lut2x: array[16, int8] =
  [int8 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]

when lpUseNeon:
  # widen one int8x16 (16 weights for columns `col..col+15`) and FMA into FOUR
  # INDEPENDENT accumulators.
  #
  # Why four and not one: `vfmaq_f32` has ~4-cycle latency but the core can issue
  # several per cycle, so four FMAs chained through a single accumulator register
  # stall on each other and the loop runs at FMA LATENCY instead of throughput.
  # Four separate registers issue back-to-back. This is worth ~1.6x on the int8
  # GEMV (measured), costs nothing but three registers, and only changes the
  # summation ORDER — which these reductions already tolerate (they are checked
  # against an fp64 reference, not bit-for-bit; see the note at the top).
  template fmaBlock(a0, a1, a2, a3, w, x, col: untyped) =
    let l16 = vmovl_s8(vget_low_s8(w))
    let h16 = vmovl_s8(vget_high_s8(w))
    a0 = vfmaq_f32(
      a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(l16))), vld1q_f32(unsafeAddr x[col])
    )
    a1 = vfmaq_f32(
      a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(l16))), vld1q_f32(unsafeAddr x[col + 4])
    )
    a2 = vfmaq_f32(
      a2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(h16))), vld1q_f32(unsafeAddr x[col + 8])
    )
    a3 = vfmaq_f32(
      a3,
      vcvtq_f32_s32(vmovl_s16(vget_high_s16(h16))),
      vld1q_f32(unsafeAddr x[col + 12]),
    )

  # Fold the four accumulators down to one fp32: two lanewise adds, then one
  # horizontal sum (cheaper than four vaddvq_f32).
  template foldAcc(a0, a1, a2, a3: untyped): float32 =
    vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)))

  # DECLARES the four accumulators, initialized. Declaring them uninitialized
  # (`var a0: float32x4`) and assigning after costs a redundant zeroing store of
  # each register per group — worth ~1.5x on the int8 GEMV, so keep the
  # initializer on the declaration.
  template acc4(a0, a1, a2, a3: untyped) =
    var a0 = vdupq_n_f32(0.0'f32)
    var a1 = vdupq_n_f32(0.0'f32)
    var a2 = vdupq_n_f32(0.0'f32)
    var a3 = vdupq_n_f32(0.0'f32)

  # widen one int8x16 -> 4×float32x4, multiply by scalar s, store 16 floats at
  # dst[di..di+15]. The value*scale multiply is IEEE-commutative, so this is
  # bit-identical to the scalar `d * float32(q)` regardless of operand order.
  template widenMulStore(v, dst, di, s: untyped) =
    let loH = vmovl_s8(vget_low_s8(v))
    let hiH = vmovl_s8(vget_high_s8(v))
    vst1q_f32(addr dst[di], vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(loH))), s))
    vst1q_f32(
      addr dst[di + 4], vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(loH))), s)
    )
    vst1q_f32(
      addr dst[di + 8], vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hiH))), s)
    )
    vst1q_f32(
      addr dst[di + 12], vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hiH))), s)
    )

  # widen 16 int8 (base i) -> 4×float32x4, multiply by scalar s, store to dst.
  template blockS8(q, dst, i, s: untyped) =
    let v = vld1q_s8(cast[ptr int8](unsafeAddr q[i]))
    widenMulStore(v, dst, i, s)

when lpUseAvx2:
  # widen 16 int8 (columns col..col+15) -> 2×(8×f32) and FMA the activations into
  # TWO INDEPENDENT accumulators — same dependency-chain reasoning as the NEON
  # `fmaBlock` above (vfmadd has ~4-cycle latency, several issue per cycle).
  template fmaBlockX86(a0, a1, v, x, col: untyped) =
    a0 = mm256_fmadd_ps(
      mm256_cvtepi32_ps(mm256_cvtepi8_epi32(v)), mm256_loadu_ps(unsafeAddr x[col]), a0
    )
    a1 = mm256_fmadd_ps(
      mm256_cvtepi32_ps(mm256_cvtepi8_epi32(mm_unpackhi_epi64(v, v))),
      mm256_loadu_ps(unsafeAddr x[col + 8]),
      a1,
    )

  # widen one m128i of 16 int8 -> 2×(8×f32), multiply by the dup'd scale sv,
  # store 16 floats at dst[di..di+15].
  template widenMulStoreX86(v, dst, di, sv: untyped) =
    mm256_storeu_ps(
      addr dst[di], mm256_mul_ps(mm256_cvtepi32_ps(mm256_cvtepi8_epi32(v)), sv)
    )
    mm256_storeu_ps(
      addr dst[di + 8],
      mm256_mul_ps(mm256_cvtepi32_ps(mm256_cvtepi8_epi32(mm_unpackhi_epi64(v, v))), sv),
    )

  # widen 16 int8 (base i) -> 2×(8×f32), multiply by scalar s, store to dst.
  template blockS8X86(q, dst, i, s: untyped) =
    let v = mm_loadu_si128(cast[ptr m128i](unsafeAddr q[i]))
    let sv = mm256_set1_ps(s)
    widenMulStoreX86(v, dst, i, sv)

# ---------------- dequantizeBatch (elementwise, bit-exact vs scalar) ----------------

template defDequantBatch(T: untyped) =
  proc dequantizeBatch*(q: openArray[T], p: QParams, dst: var openArray[float32]) =
    assert q.len == dst.len
    # The SIMD fast path handles the symmetric case only (no zero-point); its
    # group loop walks the per-group scales in order. Asymmetric quant and
    # non-SIMD builds fall through to the exact scalar `dequantize` at the end.
    when lpUseNeon:
      if p.zeroPoint.len == 0:
        let n = q.len
        let gsize = if p.scheme == qPerTensor: n else: p.groupSize
        var base = 0
        var g = 0
        while base < n:
          let s = p.scale[if p.scheme == qPerTensor: 0 else: g]
          let hi = min(base + gsize, n)
          var i = base
          while i + 16 <= hi:
            blockS8(q, dst, i, s)
            i += 16
          while i < hi:
            dst[i] = float32(int8(q[i])) * s
            inc i
          base = hi
          inc g
        return
    elif lpUseAvx2:
      if p.zeroPoint.len == 0:
        let n = q.len
        let gsize = if p.scheme == qPerTensor: n else: p.groupSize
        var base = 0
        var g = 0
        while base < n:
          let s = p.scale[if p.scheme == qPerTensor: 0 else: g]
          let hi = min(base + gsize, n)
          var i = base
          while i + 16 <= hi:
            blockS8X86(q, dst, i, s)
            i += 16
          while i < hi:
            dst[i] = float32(int8(q[i])) * s
            inc i
          base = hi
          inc g
        return
    dequantize(q, p, dst) # asymmetric / non-SIMD build: bit-exact scalar path

defDequantBatch(I8)
defDequantBatch(I4)

# ---------------- fused dequant-GEMV (reduction, tolerance-tested) ----------------

proc dequantGemv*(
    wq: openArray[I8],
    scales: openArray[float32],
    groupSize: int,
    x: openArray[float32],
    y: var openArray[float32],
) =
  ## y[m] = Σ_k (int8(W[m,k]) * scale[m, k div groupSize]) * x[k].
  ## W is M×K row-major (`wq`), `scales` is M×(K div groupSize) row-major,
  ## x is length K, y is length M. Weights are dequantised IN-REGISTER — never
  ## written out as an fp32 matrix (that's the whole point of fused GEMV).
  let m = y.len
  let k = x.len
  let gpr = k div groupSize # groups per row
  assert wq.len == m * k
  assert scales.len == m * gpr
  assert k mod groupSize == 0
  for row in 0 ..< m:
    let wbase = row * k
    let sbase = row * gpr
    var total = 0.0'f32
    for g in 0 ..< gpr:
      let s = scales[sbase + g]
      let gstart = g * groupSize
      let gend = gstart + groupSize
      var partial = 0.0'f32
      var kk = gstart
      when lpUseNeon:
        acc4(a0, a1, a2, a3)
        while kk + 16 <= gend:
          let wv = vld1q_s8(cast[ptr int8](unsafeAddr wq[wbase + kk]))
          fmaBlock(a0, a1, a2, a3, wv, x, kk)
            # widen 16 int8 + 4× FMA (shared with the I4 GEMV)
          kk += 16
        partial = foldAcc(a0, a1, a2, a3)
      elif lpUseAvx2:
        var b0 = mm256_setzero_ps()
        var b1 = mm256_setzero_ps()
        while kk + 16 <= gend:
          let v = mm_loadu_si128(cast[ptr m128i](unsafeAddr wq[wbase + kk]))
          fmaBlockX86(b0, b1, v, x, kk)
          kk += 16
        partial = hsum256(mm256_add_ps(b0, b1))
      while kk < gend: # scalar tail within group
        partial += float32(int8(wq[wbase + kk])) * x[kk]
        inc kk
      total += partial * s
    y[row] = total

proc dequantGemvI4*(
    wpacked: openArray[byte],
    scales: openArray[float32],
    groupSize: int,
    x: openArray[float32],
    y: var openArray[float32],
) =
  ## Fused GEMV with PACKED int4 weights (2 per byte, ggml low-nibble-first).
  ## W is M×K; `wpacked` is M×(K div 2) bytes row-major; `scales` is
  ## M×(K div groupSize); x is length K, y is length M. Nibbles are unpacked +
  ## sign-extended + dequantised entirely IN-REGISTER — never materialised.
  let m = y.len
  let k = x.len
  let gpr = k div groupSize
  let rowBytes = k div 2
  assert wpacked.len == m * rowBytes
  assert scales.len == m * gpr
  assert k mod groupSize == 0
  assert k mod 2 == 0
  when lpUseNeon:
    let m0f = vdupq_n_u8(0x0f'u8)
    let sh4pos = vdupq_n_s8(4'i8)
    let sh4neg = vdupq_n_s8(-4'i8)
  when lpUseAvx2:
    let mask0f = mm_set1_epi8(0x0f)
    let eight = mm_set1_epi8(8)
    let cnt4 = mm_cvtsi32_si128(4.cint)
  for row in 0 ..< m:
    let wbase = row * rowBytes
    let sbase = row * gpr
    var total = 0.0'f32
    for g in 0 ..< gpr:
      let s = scales[sbase + g]
      let gstart = g * groupSize
      let gend = gstart + groupSize
      var partial = 0.0'f32
      var kk = gstart
      when lpUseNeon:
        acc4(a0, a1, a2, a3)
        while kk + 32 <= gend: # 16 bytes = 32 int4 columns
          let pv = vld1q_u8(cast[ptr uint8](unsafeAddr wpacked[wbase + (kk shr 1)]))
          let loN = vandq_u8(pv, m0f) # low nibbles  (even columns)
          let hiN = vshlq_u8(pv, sh4neg) # high nibbles (odd columns), >>4
          let loS = vshlq_s8(vshlq_s8(vreinterpretq_s8_u8(loN), sh4pos), sh4neg)
            # sign-extend 4-bit
          let hiS = vshlq_s8(vshlq_s8(vreinterpretq_s8_u8(hiN), sh4pos), sh4neg)
          let wA = vzip1q_s8(loS, hiS) # columns kk .. kk+15, in order
          let wB = vzip2q_s8(loS, hiS) # columns kk+16 .. kk+31
          fmaBlock(a0, a1, a2, a3, wA, x, kk)
          fmaBlock(a0, a1, a2, a3, wB, x, kk + 16)
          kk += 32
        partial = foldAcc(a0, a1, a2, a3)
      elif lpUseAvx2:
        var b0 = mm256_setzero_ps()
        var b1 = mm256_setzero_ps()
        while kk + 32 <= gend: # 16 bytes = 32 int4 columns
          let pv =
            mm_loadu_si128(cast[ptr m128i](unsafeAddr wpacked[wbase + (kk shr 1)]))
          let loN = mm_and_si128(pv, mask0f) # low nibbles  (even columns)
          let hiN = mm_and_si128(mm_srl_epi16(pv, cnt4), mask0f)
            # high nibbles (odd columns), >>4
          # sign-extend a 4-bit value: (n xor 8) - 8  (per byte, shift-free)
          let loS = mm_sub_epi8(mm_xor_si128(loN, eight), eight)
          let hiS = mm_sub_epi8(mm_xor_si128(hiN, eight), eight)
          let wA = mm_unpacklo_epi8(loS, hiS) # columns kk .. kk+15, in order
          let wB = mm_unpackhi_epi8(loS, hiS) # columns kk+16 .. kk+31
          fmaBlockX86(b0, b1, wA, x, kk)
          fmaBlockX86(b0, b1, wB, x, kk + 16)
          kk += 32
        partial = hsum256(mm256_add_ps(b0, b1))
      while kk < gend: # scalar tail
        let b = wpacked[wbase + (kk shr 1)]
        let nib =
          if (kk and 1) == 0:
            b and 0x0f'u8
          else:
            b shr 4
        partial += fromNibble(nib).toFloat32 * x[kk]
        inc kk
      total += partial * s
    y[row] = total

proc dequantGemvF4*(
    wpacked: openArray[byte],
    scales: openArray[float32],
    groupSize: int,
    x: openArray[float32],
    y: var openArray[float32],
) =
  ## Fused GEMV with PACKED MXFP4 weights (2 per byte, ggml low-nibble-first) and
  ## per-block E8M0 scales (from `calibrateMX`). W is M×K; `wpacked` is M×(K div 2)
  ## bytes; `scales` is M×(K div groupSize) — the power-of-two block scales; x is
  ## length K, y is length M. fp4 codes are decoded to 2×value via a byte-LUT
  ## in-register, folded through the int8 widen+FMA path, and the ×2 is cancelled
  ## by halving the block scale.
  let m = y.len
  let k = x.len
  let gpr = k div groupSize
  let rowBytes = k div 2
  assert wpacked.len == m * rowBytes
  assert scales.len == m * gpr
  assert k mod groupSize == 0
  assert k mod 2 == 0
  when lpUseNeon:
    let lut = vld1q_s8(unsafeAddr mxfp4Lut2x[0])
    let m0f = vdupq_n_u8(0x0f'u8)
    let sh4neg = vdupq_n_s8(-4'i8)
  when lpUseAvx2:
    let lut = mm_loadu_si128(cast[ptr m128i](unsafeAddr mxfp4Lut2x[0]))
    let mask0f = mm_set1_epi8(0x0f)
    let cnt4 = mm_cvtsi32_si128(4.cint)
  for row in 0 ..< m:
    let wbase = row * rowBytes
    let sbase = row * gpr
    var total = 0.0'f32
    for g in 0 ..< gpr:
      let s = scales[sbase + g] * 0.5'f32 # halve to undo the LUT's ×2
      let gstart = g * groupSize
      let gend = gstart + groupSize
      var partial = 0.0'f32
      var kk = gstart
      when lpUseNeon:
        acc4(a0, a1, a2, a3)
        while kk + 32 <= gend: # 16 bytes = 32 fp4 columns
          let pv = vld1q_u8(cast[ptr uint8](unsafeAddr wpacked[wbase + (kk shr 1)]))
          let loN = vandq_u8(pv, m0f) # low nibbles  (even columns)
          let hiN = vshlq_u8(pv, sh4neg) # high nibbles (odd columns), >>4
          let loV = vqtbl1q_s8(lut, loN) # 2×value (even columns)
          let hiV = vqtbl1q_s8(lut, hiN) # 2×value (odd columns)
          let wA = vzip1q_s8(loV, hiV) # columns kk .. kk+15, in order
          let wB = vzip2q_s8(loV, hiV) # columns kk+16 .. kk+31
          fmaBlock(a0, a1, a2, a3, wA, x, kk)
          fmaBlock(a0, a1, a2, a3, wB, x, kk + 16)
          kk += 32
        partial = foldAcc(a0, a1, a2, a3)
      elif lpUseAvx2:
        var b0 = mm256_setzero_ps()
        var b1 = mm256_setzero_ps()
        while kk + 32 <= gend:
          let pv =
            mm_loadu_si128(cast[ptr m128i](unsafeAddr wpacked[wbase + (kk shr 1)]))
          let loN = mm_and_si128(pv, mask0f)
          let hiN = mm_and_si128(mm_srl_epi16(pv, cnt4), mask0f)
          let loV = mm_shuffle_epi8(lut, loN) # 2×value (even columns)
          let hiV = mm_shuffle_epi8(lut, hiN) # 2×value (odd columns)
          let wA = mm_unpacklo_epi8(loV, hiV) # columns kk .. kk+15
          let wB = mm_unpackhi_epi8(loV, hiV) # columns kk+16 .. kk+31
          fmaBlockX86(b0, b1, wA, x, kk)
          fmaBlockX86(b0, b1, wB, x, kk + 16)
          kk += 32
        partial = hsum256(mm256_add_ps(b0, b1))
      while kk < gend: # scalar tail
        let b = wpacked[wbase + (kk shr 1)]
        let nib =
          if (kk and 1) == 0:
            b and 0x0f'u8
          else:
            b shr 4
        partial += float32(mxfp4Lut2x[int(nib)]) * x[kk]
        inc kk
      total += partial * s
    y[row] = total

# ---------------- ggml block dequant (elementwise, bit-exact vs scalar) ----------------
#
# Vectorized forms of ggml.nim's `dequantizeQ8_0` / `dequantizeQ4_0` — the load
# path real GGUF weights come through. This was the single largest measured gap
# against ggml itself: the scalar loop ran ~1.55 Gelem/s where their NEON kernels
# do 11.2 (Q8_0) / 6.4 (Q4_0). Same convention as `dequantizeBatch`: elementwise,
# so the result must be BIT-IDENTICAL to the scalar reference (the only fp op is
# value×scale, and IEEE multiplication is commutative).

# The block scale is an fp16, and the scalar `toFloat32(F16)` decode is a branchy
# bit-manipulation — per 32-element block that overhead is comparable to the
# block's whole widen+mul work. So the batch kernels decode FOUR block scales at a
# time through the hardware converter (`vcvt_f32_f16`, baseline ARMv8 — the same
# instruction simd/convert relies on). fp16→fp32 is exact on any path, so the
# result is still bit-identical to the software decode.
when lpUseNeon:
  template decodeD4(blocks, i, dsf: untyped) =
    var dbuf {.noinit.}: array[4, uint16]
    dbuf[0] = bits(blocks[i].d)
    dbuf[1] = bits(blocks[i + 1].d)
    dbuf[2] = bits(blocks[i + 2].d)
    dbuf[3] = bits(blocks[i + 3].d)
    vst1q_f32(addr dsf[0], vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(addr dbuf[0]))))

proc dequantizeQ8_0Batch*(blocks: openArray[BlockQ8_0], dst: var openArray[float32]) =
  ## value[i] = d·q[i]. `dst.len` must be `blocks.len * QK`. Bit-exact.
  assert dst.len == blocks.len * QK
  when lpUseNeon:
    var dsf {.noinit.}: array[4, float32]
    var bi = 0
    while bi + 4 <= blocks.len:
      decodeD4(blocks, bi, dsf)
      for j in 0 ..< 4:
        let o = (bi + j) * QK
        widenMulStore(
          vld1q_s8(cast[ptr int8](unsafeAddr blocks[bi + j].qs[0])), dst, o, dsf[j]
        )
        widenMulStore(
          vld1q_s8(cast[ptr int8](unsafeAddr blocks[bi + j].qs[16])),
          dst,
          o + 16,
          dsf[j],
        )
      bi += 4
    while bi < blocks.len:
      let d = blocks[bi].d.toFloat32
      let o = bi * QK
      widenMulStore(vld1q_s8(cast[ptr int8](unsafeAddr blocks[bi].qs[0])), dst, o, d)
      widenMulStore(
        vld1q_s8(cast[ptr int8](unsafeAddr blocks[bi].qs[16])), dst, o + 16, d
      )
      inc bi
  elif lpUseAvx2:
    for bi in 0 ..< blocks.len:
      let sv = mm256_set1_ps(blocks[bi].d.toFloat32)
      let o = bi * QK
      widenMulStoreX86(
        mm_loadu_si128(cast[ptr m128i](unsafeAddr blocks[bi].qs[0])), dst, o, sv
      )
      widenMulStoreX86(
        mm_loadu_si128(cast[ptr m128i](unsafeAddr blocks[bi].qs[16])), dst, o + 16, sv
      )
  else:
    dequantizeQ8_0(blocks, dst)

proc dequantizeQ4_0Batch*(blocks: openArray[BlockQ4_0], dst: var openArray[float32]) =
  ## value[i] = d·(nibble[i] − 8), ggml lane order (low nibbles → lanes 0..15,
  ## high → 16..31). `dst.len` must be `blocks.len * QK`. Bit-exact.
  assert dst.len == blocks.len * QK
  when lpUseNeon:
    let m0f = vdupq_n_u8(0x0f'u8)
    let sh4neg = vdupq_n_s8(-4'i8)
    let eight = vdupq_n_s8(8'i8)
    template oneQ4(bi, d: untyped) =
      let o = bi * QK
      let pv = vld1q_u8(cast[ptr uint8](unsafeAddr blocks[bi].qs[0]))
      let loV = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pv, m0f)), eight)
      let hiV = vsubq_s8(vreinterpretq_s8_u8(vshlq_u8(pv, sh4neg)), eight)
      widenMulStore(loV, dst, o, d)
      widenMulStore(hiV, dst, o + 16, d)

    var dsf {.noinit.}: array[4, float32]
    var bi = 0
    while bi + 4 <= blocks.len:
      decodeD4(blocks, bi, dsf)
      for j in 0 ..< 4:
        oneQ4(bi + j, dsf[j])
      bi += 4
    while bi < blocks.len:
      oneQ4(bi, blocks[bi].d.toFloat32)
      inc bi
  elif lpUseAvx2:
    let mask0f = mm_set1_epi8(0x0f)
    let eight = mm_set1_epi8(8)
    let cnt4 = mm_cvtsi32_si128(4.cint)
    for bi in 0 ..< blocks.len:
      let sv = mm256_set1_ps(blocks[bi].d.toFloat32)
      let o = bi * QK
      let pv = mm_loadu_si128(cast[ptr m128i](unsafeAddr blocks[bi].qs[0]))
      let loV = mm_sub_epi8(mm_and_si128(pv, mask0f), eight)
      let hiV = mm_sub_epi8(mm_and_si128(mm_srl_epi16(pv, cnt4), mask0f), eight)
      widenMulStoreX86(loV, dst, o, sv)
      widenMulStoreX86(hiV, dst, o + 16, sv)
  else:
    dequantizeQ4_0(blocks, dst)

# ---------------- fused GEMV over ggml block formats (Q8_0, Q4_0) ----------------

proc dequantGemvQ8_0*(
    w: openArray[BlockQ8_0],
    blocksPerRow: int,
    x: openArray[float32],
    y: var openArray[float32],
) =
  ## y[m] = Σ_blocks d_block · Σ_i (q[i]·x[i]) over ggml Q8_0 weights. `w` is
  ## M×blocksPerRow blocks (row-major); x is length blocksPerRow·QK; y is M.
  ## Reuses the int8 widen+FMA path; the fp16 block scale is applied per block.
  let m = y.len
  let bpr = blocksPerRow
  assert w.len == m * bpr
  assert x.len == bpr * QK
  for row in 0 ..< m:
    var total = 0.0'f32
    for bi in 0 ..< bpr:
      let blk = w[row * bpr + bi]
      let base = bi * QK
      var partial = 0.0'f32
      when lpUseNeon:
        acc4(a0, a1, a2, a3)
        let w0 = vld1q_s8(cast[ptr int8](unsafeAddr blk.qs[0]))
        let w1 = vld1q_s8(cast[ptr int8](unsafeAddr blk.qs[16]))
        fmaBlock(a0, a1, a2, a3, w0, x, base)
        fmaBlock(a0, a1, a2, a3, w1, x, base + 16)
        partial = foldAcc(a0, a1, a2, a3)
      elif lpUseAvx2:
        var b0 = mm256_setzero_ps()
        var b1 = mm256_setzero_ps()
        let v0 = mm_loadu_si128(cast[ptr m128i](unsafeAddr blk.qs[0]))
        let v1 = mm_loadu_si128(cast[ptr m128i](unsafeAddr blk.qs[16]))
        fmaBlockX86(b0, b1, v0, x, base)
        fmaBlockX86(b0, b1, v1, x, base + 16)
        partial = hsum256(mm256_add_ps(b0, b1))
      else:
        for i in 0 ..< QK:
          partial += float32(blk.qs[i]) * x[base + i]
      total += partial * blk.d.toFloat32
    y[row] = total

proc dequantGemvQ4_0*(
    w: openArray[BlockQ4_0],
    blocksPerRow: int,
    x: openArray[float32],
    y: var openArray[float32],
) =
  ## Same over ggml Q4_0 weights: value = d·(nibble−8). Low nibbles are the first
  ## 16 lanes of a block, high nibbles the last 16 (ggml order — grouped, not
  ## interleaved). `w` is M×blocksPerRow blocks; x is blocksPerRow·QK; y is M.
  let m = y.len
  let bpr = blocksPerRow
  assert w.len == m * bpr
  assert x.len == bpr * QK
  when lpUseNeon:
    let m0f = vdupq_n_u8(0x0f'u8)
    let sh4neg = vdupq_n_s8(-4'i8)
    let eight = vdupq_n_s8(8'i8)
  when lpUseAvx2:
    let mask0f = mm_set1_epi8(0x0f)
    let cnt4 = mm_cvtsi32_si128(4.cint)
    let eight = mm_set1_epi8(8)
  for row in 0 ..< m:
    var total = 0.0'f32
    for bi in 0 ..< bpr:
      let blk = w[row * bpr + bi]
      let base = bi * QK
      var partial = 0.0'f32
      when lpUseNeon:
        acc4(a0, a1, a2, a3)
        let pv = vld1q_u8(cast[ptr uint8](unsafeAddr blk.qs[0]))
        let loV = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pv, m0f)), eight)
          # low nibbles − 8
        let hiV = vsubq_s8(vreinterpretq_s8_u8(vshlq_u8(pv, sh4neg)), eight)
          # high nibbles − 8
        fmaBlock(a0, a1, a2, a3, loV, x, base) # lanes 0..15
        fmaBlock(a0, a1, a2, a3, hiV, x, base + 16) # lanes 16..31
        partial = foldAcc(a0, a1, a2, a3)
      elif lpUseAvx2:
        var b0 = mm256_setzero_ps()
        var b1 = mm256_setzero_ps()
        let pv = mm_loadu_si128(cast[ptr m128i](unsafeAddr blk.qs[0]))
        let loV = mm_sub_epi8(mm_and_si128(pv, mask0f), eight)
        let hiV = mm_sub_epi8(mm_and_si128(mm_srl_epi16(pv, cnt4), mask0f), eight)
        fmaBlockX86(b0, b1, loV, x, base)
        fmaBlockX86(b0, b1, hiV, x, base + 16)
        partial = hsum256(mm256_add_ps(b0, b1))
      else:
        for i in 0 ..< QK div 2:
          let b = blk.qs[i]
          partial += float32(int(b and 0x0f'u8) - 8) * x[base + i]
          partial += float32(int(b shr 4) - 8) * x[base + i + QK div 2]
      total += partial * blk.d.toFloat32
    y[row] = total

# ---------------- int8-ACTIVATION GEMV (SDOT path) ----------------
#
# The kernels above widen int8 weights to fp32 and do FP FMA against fp32
# activations: ~1 vector instruction per weight, and 4 FMAs per 16 weights. These
# two instead take the activations ALREADY QUANTIZED to int8 and use `vdotq_s32`
# (ARMv8.2 FEAT_DotProd) — 16 multiply-accumulates per instruction, int32
# accumulate, and ONE fp multiply per group instead of four per 16 lanes. Measured
# 2.1x over the fp32-FMA path on an M1 Pro, plus 1.6x more from the 4-row tiling
# below, which is the same shape llamafile/tinyBLAS uses (`mnpack<4,...>`).
#
# WHY THIS IS A SEPARATE ENTRY POINT, not a faster `dequantGemv`: quantizing the
# activations loses precision. That is the CALLER's tradeoff, so they pass `xq` and
# its scales in, and the fp32-activation kernels stay exactly as accurate as before.
# Use `calibrateSymmetric(x, groupSize, 127.0) + quantize` to produce them; the
# activation vector is length K, so that cost is O(K) against the GEMV's O(M·K).
#
# ACCURACY of the sum itself is BETTER than the fp32 path, not worse: int32
# accumulation of a group is EXACT. A group of `g` weights bounds |Σ| at
# g·127·127, so anything up to g = 132_000 cannot overflow.
#
# The `lpUseDotProd == false` fallback accumulates in int32 too, so it produces
# BIT-IDENTICAL results — the tests demand exact equality across both builds.
#
# BUT the fallback exists for PORTABILITY, NOT SPEED: its plain integer loop does
# not auto-vectorize, so on a build without FEAT_DotProd it measures ~1.2 GFLOP/s
# against the fp32-activation kernel's ~2.0 (scalar build, 4096²). On such a target
# — including all of x86 today, since AVX2 has no int8 dot product and this path
# would need `pmaddubsw`+`madd` or AVX512-VNNI — prefer `dequantGemv`. Check
# `lpUseDotProd` if you want to choose at compile time.

template applyScales(isum: int32, ws, xs: float32): float32 =
  ## The one fp expression every path must share, in the same order, or the SDOT,
  ## vector-finalize and fallback results would not be bit-identical.
  float32(isum) * ws * xs

# TRIED AND REJECTED — do not "optimize" this back without measuring.
#
# The per-group finalize here is four independent `vaddvq_s32 → int→float → mul →
# mul → add` chains, one per row of the tile. That looks like the bottleneck: this
# kernel moves ~14 GB/s of weights against a measured 54.9 GB/s single-core stream
# ceiling, so it is plainly not load-bound. Replacing the four chains with a single
# vector one — `vpaddq_s32` twice to fold the four accumulators into one int32x4 of
# row sums, then one `vmulq/vmulq/vaddq` — was NEUTRAL: int8 unchanged (27.7 → 28.0
# GFLOP/s at k=14336), packed int4 +8% at 4096² but −4% at k=14336. The strided
# gather of four rows' scales into a vector costs about what the shorter dependency
# chain saves.
#
# So the finalize is NOT the limiter, and what is remains unidentified — the loop
# also sits ~3.6x off the SDOT issue rate, so the next step is hardware counters
# (Instruments / perf), not another guess. The `vpaddq_s32` / `vmulq_f32` /
# `vld1q_s32` bindings stay in `neon.nim`; they are correct and cost nothing.

# How the M-loop is shaped depends on where the weights live, and the two regimes
# want OPPOSITE shapes (measured, M1 Pro, int8 GEMV):
#
#                          4-row tile   1-row stream
#   cache-resident (int4 4096², 8.4 MB)   86-103        70    GFLOP/s
#   DRAM streaming (k=14336, 29-58 MB)    54-58       81-88   GFLOP/s
#
# The tile amortizes the activation load over 4 rows but reads 4 weight streams at
# once; the prefetcher sustains one sequential stream at ~44 GB/s and the 4-stream
# pattern at only ~28. So: weights bigger than `lpStreamBytes` (default 12 MB, the
# M1 P-cluster L2 — override with -d:lpStreamBytes=N) go row-at-a-time; smaller
# stay on the tile. Both shapes produce BIT-IDENTICAL results: group sums are exact
# int32 either way, and the per-group finalize is the same expression — the tests
# run with -d:lpStreamBytes=0 to pin the stream path too.
const lpStreamBytes {.intdefine.}: int = 12_582_912

when lpUseDotProd:
  proc gemvQ8Stream(
      wq: openArray[I8],
      wScales: openArray[float32],
      groupSize: int,
      xq: openArray[I8],
      xScales: openArray[float32],
      y: var openArray[float32],
  ) =
    ## Row-at-a-time int8×int8: one sequential weight stream, 4 SDOT chains over
    ## 64 columns per iteration, walking raw addresses.
    ##
    ## Two codegen lessons are baked into this shape (measured, M1 Pro, k=14336):
    ## indexing (`wq[wbase+kk]`) makes clang recompute base+offset per load and
    ## skip unrolling — 58 GFLOP/s vs ~90 for this pointer-walking form. And do
    ## NOT "optimize" this into a 128-column unroll: at the dominant groupSize of
    ## 128 that loop body executes once per group, so its extra accumulators,
    ## folds and tail checks all become per-group overhead — measured 20% slower.
    let m = y.len
    let k = xq.len
    let gpr = k div groupSize
    let wbase0 = cast[int](unsafeAddr wq[0])
    let xbase0 = cast[int](unsafeAddr xq[0])
    for row in 0 ..< m:
      var wa = wbase0 + row * k
      var xa = xbase0
      var total = 0.0'f32
      for g in 0 ..< gpr:
        var a0 = vdupq_n_s32(0'i32)
        var a1 = vdupq_n_s32(0'i32)
        var a2 = vdupq_n_s32(0'i32)
        var a3 = vdupq_n_s32(0'i32)
        var left = groupSize
        while left >= 64:
          a0 = vdotq_s32(a0, vld1q_s8(cast[ptr int8](wa)), vld1q_s8(cast[ptr int8](xa)))
          a1 = vdotq_s32(
            a1, vld1q_s8(cast[ptr int8](wa + 16)), vld1q_s8(cast[ptr int8](xa + 16))
          )
          a2 = vdotq_s32(
            a2, vld1q_s8(cast[ptr int8](wa + 32)), vld1q_s8(cast[ptr int8](xa + 32))
          )
          a3 = vdotq_s32(
            a3, vld1q_s8(cast[ptr int8](wa + 48)), vld1q_s8(cast[ptr int8](xa + 48))
          )
          wa += 64
          xa += 64
          left -= 64
        while left >= 16:
          a0 = vdotq_s32(a0, vld1q_s8(cast[ptr int8](wa)), vld1q_s8(cast[ptr int8](xa)))
          wa += 16
          xa += 16
          left -= 16
        var isum = vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)))
        while left > 0:
          isum += int32(cast[ptr int8](wa)[]) * int32(cast[ptr int8](xa)[])
          wa += 1
          xa += 1
          dec left
        total += applyScales(isum, wScales[row * gpr + g], xScales[g])
      y[row] = total

proc dequantGemvQ8*(
    wq: openArray[I8],
    wScales: openArray[float32],
    groupSize: int,
    xq: openArray[I8],
    xScales: openArray[float32],
    y: var openArray[float32],
) =
  ## y[m] = Σ_g (Σ_k w·x as an exact int32) · wScale[m,g] · xScale[g], with int8
  ## weights AND int8 activations. W is M×K row-major; `wScales` is M×(K div
  ## groupSize); `xq` is length K with `xScales` of length (K div groupSize).
  ## Rows are processed in tiles of 4 so one activation load feeds four dots.
  let m = y.len
  let k = xq.len
  let gpr = k div groupSize
  assert wq.len == m * k
  assert wScales.len == m * gpr
  assert xScales.len == gpr
  assert k mod groupSize == 0
  when lpUseDotProd:
    if wq.len > lpStreamBytes:
      gemvQ8Stream(wq, wScales, groupSize, xq, xScales, y)
      return
    var row = 0
    while row + 4 <= m: # ---- 4-row tile ----
      let w0 = row * k
      let w1 = w0 + k
      let w2 = w1 + k
      let w3 = w2 + k
      var t0 = 0.0'f32
      var t1 = 0.0'f32
      var t2 = 0.0'f32
      var t3 = 0.0'f32
      for g in 0 ..< gpr:
        var a0 = vdupq_n_s32(0'i32)
        var a1 = vdupq_n_s32(0'i32)
        var a2 = vdupq_n_s32(0'i32)
        var a3 = vdupq_n_s32(0'i32)
        var kk = g * groupSize
        let gend = kk + groupSize
        while kk + 16 <= gend:
          let xv = vld1q_s8(cast[ptr int8](unsafeAddr xq[kk])) # loaded ONCE for 4 rows
          a0 = vdotq_s32(a0, vld1q_s8(cast[ptr int8](unsafeAddr wq[w0 + kk])), xv)
          a1 = vdotq_s32(a1, vld1q_s8(cast[ptr int8](unsafeAddr wq[w1 + kk])), xv)
          a2 = vdotq_s32(a2, vld1q_s8(cast[ptr int8](unsafeAddr wq[w2 + kk])), xv)
          a3 = vdotq_s32(a3, vld1q_s8(cast[ptr int8](unsafeAddr wq[w3 + kk])), xv)
          kk += 16
        var s0 = vaddvq_s32(a0)
        var s1 = vaddvq_s32(a1)
        var s2 = vaddvq_s32(a2)
        var s3 = vaddvq_s32(a3)
        while kk < gend: # group tail (groupSize not a multiple of 16)
          let xvs = int32(int8(xq[kk]))
          s0 += int32(int8(wq[w0 + kk])) * xvs
          s1 += int32(int8(wq[w1 + kk])) * xvs
          s2 += int32(int8(wq[w2 + kk])) * xvs
          s3 += int32(int8(wq[w3 + kk])) * xvs
          inc kk
        let xs = xScales[g]
        t0 += applyScales(s0, wScales[row * gpr + g], xs)
        t1 += applyScales(s1, wScales[(row + 1) * gpr + g], xs)
        t2 += applyScales(s2, wScales[(row + 2) * gpr + g], xs)
        t3 += applyScales(s3, wScales[(row + 3) * gpr + g], xs)
      y[row] = t0
      y[row + 1] = t1
      y[row + 2] = t2
      y[row + 3] = t3
      row += 4
    while row < m: # ---- rows that don't fill a tile ----
      let wbase = row * k
      var total = 0.0'f32
      for g in 0 ..< gpr:
        var a0 = vdupq_n_s32(0'i32)
        var kk = g * groupSize
        let gend = kk + groupSize
        while kk + 16 <= gend:
          a0 = vdotq_s32(
            a0,
            vld1q_s8(cast[ptr int8](unsafeAddr wq[wbase + kk])),
            vld1q_s8(cast[ptr int8](unsafeAddr xq[kk])),
          )
          kk += 16
        var s0 = vaddvq_s32(a0)
        while kk < gend:
          s0 += int32(int8(wq[wbase + kk])) * int32(int8(xq[kk]))
          inc kk
        total += applyScales(s0, wScales[row * gpr + g], xScales[g])
      y[row] = total
      inc row
  elif lpUseAvx2:
    # AVX2 stand-in for SDOT: abs/sign + maddubs + madd (see x86.nim). The
    # integer group sums are exact and the finalize is `applyScales`, so this
    # path produces the SAME BITS as the NEON and scalar paths — the x86 CI leg
    # proves it on real hardware.
    let ones = mm256_set1_epi16(1)
    for row in 0 ..< m:
      let wbase = row * k
      var total = 0.0'f32
      for g in 0 ..< gpr:
        var acc = mm256_setzero_si256()
        var kk = g * groupSize
        let gend = kk + groupSize
        while kk + 32 <= gend:
          let wv = mm256_loadu_si256(cast[ptr m256i](unsafeAddr wq[wbase + kk]))
          let xv = mm256_loadu_si256(cast[ptr m256i](unsafeAddr xq[kk]))
          let p16 =
            mm256_maddubs_epi16(mm256_sign_epi8(wv, wv), mm256_sign_epi8(xv, wv))
          acc = mm256_add_epi32(acc, mm256_madd_epi16(p16, ones))
          kk += 32
        var isum = hsum256i(acc)
        while kk < gend:
          isum += int32(int8(wq[wbase + kk])) * int32(int8(xq[kk]))
          inc kk
        total += applyScales(isum, wScales[row * gpr + g], xScales[g])
      y[row] = total
  else:
    # No int8-dot hardware: same integer arithmetic, so the same bits out.
    for row in 0 ..< m:
      let wbase = row * k
      var total = 0.0'f32
      for g in 0 ..< gpr:
        var isum = 0'i32
        for kk in g * groupSize ..< (g + 1) * groupSize:
          isum += int32(int8(wq[wbase + kk])) * int32(int8(xq[kk]))
        total += applyScales(isum, wScales[row * gpr + g], xScales[g])
      y[row] = total

when lpUseDotProd:
  proc gemvI4Q8Stream(
      wpacked: openArray[byte],
      wScales: openArray[float32],
      groupSize: int,
      xq: openArray[I8],
      xScales: openArray[float32],
      y: var openArray[float32],
  ) =
    ## Row-at-a-time packed int4 × int8: 32 weight bytes (64 columns) per
    ## iteration, 4 SDOT chains, all loads before any unpack. Pointer-walking and
    ## deliberately NOT unrolled further — see gemvQ8Stream's shape note.
    let m = y.len
    let k = xq.len
    let gpr = k div groupSize
    let rowBytes = k div 2
    let m0f = vdupq_n_u8(0x0f'u8)
    let sh4pos = vdupq_n_s8(4'i8)
    let sh4neg = vdupq_n_s8(-4'i8)
    let wbase0 = cast[int](unsafeAddr wpacked[0])
    let xbase0 = cast[int](unsafeAddr xq[0])
    for row in 0 ..< m:
      var wa = wbase0 + row * rowBytes
      var xa = xbase0
      var total = 0.0'f32
      for g in 0 ..< gpr:
        var a0 = vdupq_n_s32(0'i32)
        var a1 = vdupq_n_s32(0'i32)
        var a2 = vdupq_n_s32(0'i32)
        var a3 = vdupq_n_s32(0'i32)
        var left = groupSize
        while left >= 64:
          let xvA = vld2q_s8(cast[ptr int8](xa))
          let xvB = vld2q_s8(cast[ptr int8](xa + 32))
          let pvA = vld1q_u8(cast[ptr uint8](wa))
          let pvB = vld1q_u8(cast[ptr uint8](wa + 16))
          let loA =
            vshlq_s8(vshlq_s8(vreinterpretq_s8_u8(vandq_u8(pvA, m0f)), sh4pos), sh4neg)
          let hiA = vshlq_s8(
            vshlq_s8(vreinterpretq_s8_u8(vshlq_u8(pvA, sh4neg)), sh4pos), sh4neg
          )
          let loB =
            vshlq_s8(vshlq_s8(vreinterpretq_s8_u8(vandq_u8(pvB, m0f)), sh4pos), sh4neg)
          let hiB = vshlq_s8(
            vshlq_s8(vreinterpretq_s8_u8(vshlq_u8(pvB, sh4neg)), sh4pos), sh4neg
          )
          a0 = vdotq_s32(a0, loA, xvA.val[0])
          a1 = vdotq_s32(a1, hiA, xvA.val[1])
          a2 = vdotq_s32(a2, loB, xvB.val[0])
          a3 = vdotq_s32(a3, hiB, xvB.val[1])
          wa += 32
          xa += 64
          left -= 64
        while left >= 32:
          let xv = vld2q_s8(cast[ptr int8](xa))
          let pv = vld1q_u8(cast[ptr uint8](wa))
          let loS =
            vshlq_s8(vshlq_s8(vreinterpretq_s8_u8(vandq_u8(pv, m0f)), sh4pos), sh4neg)
          let hiS = vshlq_s8(
            vshlq_s8(vreinterpretq_s8_u8(vshlq_u8(pv, sh4neg)), sh4pos), sh4neg
          )
          a0 = vdotq_s32(a0, loS, xv.val[0])
          a1 = vdotq_s32(a1, hiS, xv.val[1])
          wa += 16
          xa += 32
          left -= 32
        var isum = vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)))
        var col = 0
        while left > 0: # nibble tail
          let b = cast[ptr uint8](wa + (col shr 1))[]
          isum +=
            int32(
              int8(
                fromNibble(
                  if (col and 1) == 1:
                    b shr 4
                  else:
                    b and 0x0f'u8
                )
              )
            ) * int32(cast[ptr int8](xa + col)[])
          inc col
          dec left
        wa += (col + 1) shr 1
        xa += col
        total += applyScales(isum, wScales[row * gpr + g], xScales[g])
      y[row] = total

proc dequantGemvI4Q8*(
    wpacked: openArray[byte],
    wScales: openArray[float32],
    groupSize: int,
    xq: openArray[I8],
    xScales: openArray[float32],
    y: var openArray[float32],
) =
  ## The same SDOT path over PACKED int4 weights (2 per byte, low nibble = even
  ## column). Halving the weight bytes is what actually closes the gap to
  ## llama.cpp: at 4096² the fp32-FMA int8 kernel is already at this machine's
  ## single-core streaming limit, so past that point only bytes-per-weight buys
  ## anything.
  ##
  ## The activations are de-interleaved ONCE per 32 columns with `vld2q_s8` —
  ## val[0] holds the even columns (the low nibbles), val[1] the odd ones — instead
  ## of zipping the weights back into column order for every row.
  let m = y.len
  let k = xq.len
  let gpr = k div groupSize
  let rowBytes = k div 2
  assert wpacked.len == m * rowBytes
  assert wScales.len == m * gpr
  assert xScales.len == gpr
  assert k mod groupSize == 0
  assert k mod 2 == 0
  when lpUseDotProd:
    if wpacked.len > lpStreamBytes:
      gemvI4Q8Stream(wpacked, wScales, groupSize, xq, xScales, y)
      return

  template nib(b: byte, odd: bool): int32 =
    ## One packed weight, sign-extended from 4 bits, for the scalar paths.
    int32(
      int8(
        fromNibble(
          if odd:
            b shr 4
          else:
            b and 0x0f'u8
        )
      )
    )

  when lpUseDotProd:
    let m0f = vdupq_n_u8(0x0f'u8)
    let sh4pos = vdupq_n_s8(4'i8)
    let sh4neg = vdupq_n_s8(-4'i8)

    # 16 packed bytes -> two int8x16 of sign-extended nibbles, dotted against the
    # pre-split activations. Two SDOTs cover 32 columns of one row.
    template dotPacked(acc, wbase, kk, xv: untyped) =
      let pv = vld1q_u8(cast[ptr uint8](unsafeAddr wpacked[wbase + (kk shr 1)]))
      let loS =
        vshlq_s8(vshlq_s8(vreinterpretq_s8_u8(vandq_u8(pv, m0f)), sh4pos), sh4neg)
      let hiS =
        vshlq_s8(vshlq_s8(vreinterpretq_s8_u8(vshlq_u8(pv, sh4neg)), sh4pos), sh4neg)
      acc = vdotq_s32(acc, loS, xv.val[0])
      acc = vdotq_s32(acc, hiS, xv.val[1])

    var row = 0
    while row + 4 <= m: # ---- 4-row tile ----
      let b0 = row * rowBytes
      let b1 = b0 + rowBytes
      let b2 = b1 + rowBytes
      let b3 = b2 + rowBytes
      var t0 = 0.0'f32
      var t1 = 0.0'f32
      var t2 = 0.0'f32
      var t3 = 0.0'f32
      for g in 0 ..< gpr:
        var a0 = vdupq_n_s32(0'i32)
        var a1 = vdupq_n_s32(0'i32)
        var a2 = vdupq_n_s32(0'i32)
        var a3 = vdupq_n_s32(0'i32)
        var kk = g * groupSize
        let gend = kk + groupSize
        while kk + 32 <= gend: # 16 bytes = 32 int4 columns
          let xv = vld2q_s8(cast[ptr int8](unsafeAddr xq[kk])) # split ONCE for 4 rows
          dotPacked(a0, b0, kk, xv)
          dotPacked(a1, b1, kk, xv)
          dotPacked(a2, b2, kk, xv)
          dotPacked(a3, b3, kk, xv)
          kk += 32
        var s0 = vaddvq_s32(a0)
        var s1 = vaddvq_s32(a1)
        var s2 = vaddvq_s32(a2)
        var s3 = vaddvq_s32(a3)
        while kk < gend: # group tail
          let odd = (kk and 1) == 1
          let bi = kk shr 1
          let xvs = int32(int8(xq[kk]))
          s0 += nib(wpacked[b0 + bi], odd) * xvs
          s1 += nib(wpacked[b1 + bi], odd) * xvs
          s2 += nib(wpacked[b2 + bi], odd) * xvs
          s3 += nib(wpacked[b3 + bi], odd) * xvs
          inc kk
        let xs = xScales[g]
        t0 += applyScales(s0, wScales[row * gpr + g], xs)
        t1 += applyScales(s1, wScales[(row + 1) * gpr + g], xs)
        t2 += applyScales(s2, wScales[(row + 2) * gpr + g], xs)
        t3 += applyScales(s3, wScales[(row + 3) * gpr + g], xs)
      y[row] = t0
      y[row + 1] = t1
      y[row + 2] = t2
      y[row + 3] = t3
      row += 4
    while row < m: # ---- tile remainder ----
      let wbase = row * rowBytes
      var total = 0.0'f32
      for g in 0 ..< gpr:
        var a0 = vdupq_n_s32(0'i32)
        var kk = g * groupSize
        let gend = kk + groupSize
        while kk + 32 <= gend:
          let xv = vld2q_s8(cast[ptr int8](unsafeAddr xq[kk]))
          dotPacked(a0, wbase, kk, xv)
          kk += 32
        var s0 = vaddvq_s32(a0)
        while kk < gend:
          s0 += nib(wpacked[wbase + (kk shr 1)], (kk and 1) == 1) * int32(int8(xq[kk]))
          inc kk
        total += applyScales(s0, wScales[row * gpr + g], xScales[g])
      y[row] = total
      inc row
  elif lpUseAvx2:
    # Packed nibbles on AVX2: unpack 16 bytes into 32 sign-extended int8 in
    # COLUMN order using the 128-bit unpacklo/hi (which do NOT cross lanes),
    # then combine into one 256-bit register so a contiguous 32-column x load
    # lines up. abs/sign+maddubs as in the int8 kernel; values in [-8,7] are far
    # inside the saturation bound. Bit-identical to the other paths.
    let mask0f = mm_set1_epi8(0x0f)
    let eight8 = mm_set1_epi8(8)
    let cnt4 = mm_cvtsi32_si128(4.cint)
    let ones = mm256_set1_epi16(1)
    for row in 0 ..< m:
      let wbase = row * rowBytes
      var total = 0.0'f32
      for g in 0 ..< gpr:
        var acc = mm256_setzero_si256()
        var kk = g * groupSize
        let gend = kk + groupSize
        while kk + 32 <= gend: # 16 bytes = 32 columns
          let pv =
            mm_loadu_si128(cast[ptr m128i](unsafeAddr wpacked[wbase + (kk shr 1)]))
          let loS = mm_sub_epi8(mm_xor_si128(mm_and_si128(pv, mask0f), eight8), eight8)
          let hiS = mm_sub_epi8(
            mm_xor_si128(mm_and_si128(mm_srl_epi16(pv, cnt4), mask0f), eight8), eight8
          )
          let wv = mm256_set_m128i(
            mm_unpackhi_epi8(loS, hiS), # cols kk+16..kk+31
            mm_unpacklo_epi8(loS, hiS),
          ) # cols kk..kk+15
          let xv = mm256_loadu_si256(cast[ptr m256i](unsafeAddr xq[kk]))
          let p16 =
            mm256_maddubs_epi16(mm256_sign_epi8(wv, wv), mm256_sign_epi8(xv, wv))
          acc = mm256_add_epi32(acc, mm256_madd_epi16(p16, ones))
          kk += 32
        var isum = hsum256i(acc)
        while kk < gend:
          isum += nib(wpacked[wbase + (kk shr 1)], (kk and 1) == 1) * int32(
            int8(xq[kk])
          )
          inc kk
        total += applyScales(isum, wScales[row * gpr + g], xScales[g])
      y[row] = total
  else:
    for row in 0 ..< m:
      let wbase = row * rowBytes
      var total = 0.0'f32
      for g in 0 ..< gpr:
        var isum = 0'i32
        for kk in g * groupSize ..< (g + 1) * groupSize:
          isum += nib(wpacked[wbase + (kk shr 1)], (kk and 1) == 1) * int32(
            int8(xq[kk])
          )
        total += applyScales(isum, wScales[row * gpr + g], xScales[g])
      y[row] = total

# ---------------- int8-activation GEMV over ggml BLOCK layouts ----------------
#
# `dequantGemvQ8_0Q8` / `dequantGemvQ4_0Q8`: SDOT GEMV where BOTH sides use the
# ggml block format — weights as BlockQ8_0/BlockQ4_0 and ACTIVATIONS quantized to
# BlockQ8_0 (use `quantizeQ8_0`). This is format-identical to llama.cpp's
# `vec_dot_q8_0_q8_0` / `q4_0_q8_0`: the fp16 scale travels inline in each
# 34/18-byte block, so weights and scales are ONE stream with perfect locality,
# unlike the separate scale array of `dequantGemvQ8`.
#
# EXACTNESS: per-block sums are exact int32 (32·127·127 max). The fp accumulation
# order is defined as: blocks are taken 4 at a time; lane l of a float32x4 total
# accumulates float32(isum)·(wd·xd) for blocks with index ≡ l (mod 4); leftover
# blocks are added to lane (index mod 4); the row result is vaddvq_f32 of the
# totals — i.e. (l0+l1)+(l2+l3). The tests mirror this order with an int64/float
# reference, so the kernels are pinned BIT-FOR-BIT, same as dequantGemvQ8.
#
# The fp16 block scales are decoded through the hardware converter 4 blocks at a
# time (exact, and the software decode is ~half the cost of a whole block — see
# dequantizeQ8_0Batch). Activation scales are decoded once per call.

when lpUseDotProd:
  static:
    doAssert sizeof(BlockQ8_0) == 34 and sizeof(BlockQ4_0) == 18

  proc decodeXd(xq: openArray[BlockQ8_0]): seq[float32] =
    ## The activation blocks' fp16 scales, decoded once per matvec.
    result = newSeq[float32](xq.len)
    for i in 0 ..< xq.len:
      result[i] = xq[i].d.toFloat32

  # decode the fp16 scales of w-blocks b, b+1, b+2, b+3 (stride `stride` bytes,
  # d at offset 0) into one float32x4 — hardware convert, exact.
  template gatherD4(base, stride: int): float32x4 =
    var dbuf {.noinit.}: array[4, uint16]
    dbuf[0] = cast[ptr uint16](base)[]
    dbuf[1] = cast[ptr uint16](base + stride)[]
    dbuf[2] = cast[ptr uint16](base + 2 * stride)[]
    dbuf[3] = cast[ptr uint16](base + 3 * stride)[]
    vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(addr dbuf[0])))

proc dequantGemvQ8_0Q8*(
    w: openArray[BlockQ8_0],
    blocksPerRow: int,
    xq: openArray[BlockQ8_0],
    y: var openArray[float32],
) =
  ## y[m] = Σ_blocks (Σ int8·int8 exact) · wd·xd over ggml Q8_0 weights AND Q8_0
  ## activations. `w` is M×blocksPerRow row-major; `xq` is blocksPerRow blocks.
  let m = y.len
  let bpr = blocksPerRow
  assert w.len == m * bpr
  assert xq.len == bpr
  when lpUseDotProd:
    const WB = 34 # sizeof BlockQ8_0
    let xd = decodeXd(xq)
    let wbase0 = cast[int](unsafeAddr w[0])
    let xbase0 = cast[int](unsafeAddr xq[0])
    for row in 0 ..< m:
      var wa = wbase0 + row * bpr * WB
      var xa = xbase0
      var totals = vdupq_n_f32(0.0'f32)
      var bi = 0
      while bi + 4 <= bpr:
        let wd4 = gatherD4(wa, WB)
        let xd4 = vld1q_f32(unsafeAddr xd[bi])
        var s0 = vdupq_n_s32(0'i32)
        var s1 = vdupq_n_s32(0'i32)
        var s2 = vdupq_n_s32(0'i32)
        var s3 = vdupq_n_s32(0'i32)
        # two SDOT per block; each block's 32 columns land in ONE accumulator so
        # the pairwise fold below yields per-block sums in lanes 0..3
        s0 = vdotq_s32(
          vdotq_s32(
            s0, vld1q_s8(cast[ptr int8](wa + 2)), vld1q_s8(cast[ptr int8](xa + 2))
          ),
          vld1q_s8(cast[ptr int8](wa + 18)),
          vld1q_s8(cast[ptr int8](xa + 18)),
        )
        s1 = vdotq_s32(
          vdotq_s32(
            s1,
            vld1q_s8(cast[ptr int8](wa + WB + 2)),
            vld1q_s8(cast[ptr int8](xa + WB + 2)),
          ),
          vld1q_s8(cast[ptr int8](wa + WB + 18)),
          vld1q_s8(cast[ptr int8](xa + WB + 18)),
        )
        s2 = vdotq_s32(
          vdotq_s32(
            s2,
            vld1q_s8(cast[ptr int8](wa + 2 * WB + 2)),
            vld1q_s8(cast[ptr int8](xa + 2 * WB + 2)),
          ),
          vld1q_s8(cast[ptr int8](wa + 2 * WB + 18)),
          vld1q_s8(cast[ptr int8](xa + 2 * WB + 18)),
        )
        s3 = vdotq_s32(
          vdotq_s32(
            s3,
            vld1q_s8(cast[ptr int8](wa + 3 * WB + 2)),
            vld1q_s8(cast[ptr int8](xa + 3 * WB + 2)),
          ),
          vld1q_s8(cast[ptr int8](wa + 3 * WB + 18)),
          vld1q_s8(cast[ptr int8](xa + 3 * WB + 18)),
        )
        let sums = vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3))
        totals = vaddq_f32(totals, vmulq_f32(vcvtq_f32_s32(sums), vmulq_f32(wd4, xd4)))
        wa += 4 * WB
        xa += 4 * WB
        bi += 4
      var tailTotals: array[4, float32]
      vst1q_f32(addr tailTotals[0], totals)
      while bi < bpr: # leftover blocks -> lane (bi mod 4)
        var a = vdupq_n_s32(0'i32)
        a = vdotq_s32(
          vdotq_s32(
            a, vld1q_s8(cast[ptr int8](wa + 2)), vld1q_s8(cast[ptr int8](xa + 2))
          ),
          vld1q_s8(cast[ptr int8](wa + 18)),
          vld1q_s8(cast[ptr int8](xa + 18)),
        )
        let wd = cast[ptr F16](wa)[].toFloat32
        tailTotals[bi and 3] =
          tailTotals[bi and 3] + float32(vaddvq_s32(a)) * (wd * xd[bi])
        wa += WB
        xa += WB
        inc bi
      y[row] = (tailTotals[0] + tailTotals[1]) + (tailTotals[2] + tailTotals[3])
  else:
    # Portability fallback: same arithmetic, same summation order, same bits.
    for row in 0 ..< m:
      var lanes: array[4, float32]
      for bi in 0 ..< bpr:
        var isum = 0'i32
        for i in 0 ..< QK:
          isum += int32(w[row * bpr + bi].qs[i]) * int32(xq[bi].qs[i])
        lanes[bi and 3] =
          lanes[bi and 3] +
          float32(isum) * (w[row * bpr + bi].d.toFloat32 * xq[bi].d.toFloat32)
      y[row] = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])

proc dequantGemvQ4_0Q8*(
    w: openArray[BlockQ4_0],
    blocksPerRow: int,
    xq: openArray[BlockQ8_0],
    y: var openArray[float32],
) =
  ## Same over ggml Q4_0 weights: value = d·(nibble−8), low nibbles are lanes
  ## 0..15 of the block, high nibbles lanes 16..31 (ggml grouped order, so the
  ## activation block needs NO de-interleave). Activations are Q8_0 blocks.
  let m = y.len
  let bpr = blocksPerRow
  assert w.len == m * bpr
  assert xq.len == bpr
  when lpUseDotProd:
    const WB = 18 # sizeof BlockQ4_0
    const XB = 34
    let m0f = vdupq_n_u8(0x0f'u8)
    let sh4neg = vdupq_n_s8(-4'i8)
    let eight = vdupq_n_s8(8'i8)
    let xd = decodeXd(xq)
    let wbase0 = cast[int](unsafeAddr w[0])
    let xbase0 = cast[int](unsafeAddr xq[0])
    template blockDot(acc, wAddr, xAddr: untyped) =
      let pv = vld1q_u8(cast[ptr uint8](wAddr + 2))
      let loV = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pv, m0f)), eight)
      let hiV = vsubq_s8(vreinterpretq_s8_u8(vshlq_u8(pv, sh4neg)), eight)
      acc = vdotq_s32(
        vdotq_s32(acc, loV, vld1q_s8(cast[ptr int8](xAddr + 2))),
        hiV,
        vld1q_s8(cast[ptr int8](xAddr + 18)),
      )

    for row in 0 ..< m:
      var wa = wbase0 + row * bpr * WB
      var xa = xbase0
      var totals = vdupq_n_f32(0.0'f32)
      var bi = 0
      while bi + 4 <= bpr:
        let wd4 = gatherD4(wa, WB)
        let xd4 = vld1q_f32(unsafeAddr xd[bi])
        var s0 = vdupq_n_s32(0'i32)
        var s1 = vdupq_n_s32(0'i32)
        var s2 = vdupq_n_s32(0'i32)
        var s3 = vdupq_n_s32(0'i32)
        blockDot(s0, wa, xa)
        blockDot(s1, wa + WB, xa + XB)
        blockDot(s2, wa + 2 * WB, xa + 2 * XB)
        blockDot(s3, wa + 3 * WB, xa + 3 * XB)
        let sums = vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3))
        totals = vaddq_f32(totals, vmulq_f32(vcvtq_f32_s32(sums), vmulq_f32(wd4, xd4)))
        wa += 4 * WB
        xa += 4 * XB
        bi += 4
      var tailTotals: array[4, float32]
      vst1q_f32(addr tailTotals[0], totals)
      while bi < bpr:
        var a = vdupq_n_s32(0'i32)
        blockDot(a, wa, xa)
        let wd = cast[ptr F16](wa)[].toFloat32
        tailTotals[bi and 3] =
          tailTotals[bi and 3] + float32(vaddvq_s32(a)) * (wd * xd[bi])
        wa += WB
        xa += XB
        inc bi
      y[row] = (tailTotals[0] + tailTotals[1]) + (tailTotals[2] + tailTotals[3])
  else:
    for row in 0 ..< m:
      var lanes: array[4, float32]
      for bi in 0 ..< bpr:
        var isum = 0'i32
        for i in 0 ..< QK div 2:
          let b = w[row * bpr + bi].qs[i]
          isum += (int32(b and 0x0f'u8) - 8) * int32(xq[bi].qs[i])
          isum += (int32(b shr 4) - 8) * int32(xq[bi].qs[i + QK div 2])
        lanes[bi and 3] =
          lanes[bi and 3] +
          float32(isum) * (w[row * bpr + bi].d.toFloat32 * xq[bi].d.toFloat32)
      y[row] = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])

# ---------------- multi-column GEMM (n>1), int8 activations ----------------
#
# Y = W · X for N activation columns at once — the prompt-processing / batch>1
# shape. The point is WEIGHT TRAFFIC: a GEMV per column reads the whole weight
# matrix from DRAM N times; here the column loop sits INSIDE the group loop, so
# each 128-byte weight group is read from DRAM once per row and its N reuses are
# L1 hits. Per column the arithmetic (4 SDOT chains, one finalize per group) is
# IDENTICAL to the single-column stream kernels, so each output column is
# bit-for-bit what `dequantGemvQ8`/`dequantGemvI4Q8` produce for that column —
# the tests check exactly that.
#
# Layout: `xq` is K×n column-major (column c at xq[c*K .. c*K+K-1]); `xScales`
# is n×gpr column-major (c*gpr + g); `y` is M×n ROW-major (y[row*n + c]), so a
# row range — the unit of thread sharding — is a contiguous block.

proc dequantGemmQ8*(
    wq: openArray[I8],
    wScales: openArray[float32],
    groupSize: int,
    xq: openArray[I8],
    xScales: openArray[float32],
    n: int,
    y: var openArray[float32],
) =
  ## y[row·n + c] = Σ_g (Σ_k w·x_c exact int32) · wScale[row,g] · xScale[c,g].
  let m = y.len div n
  let k = xq.len div n
  let gpr = k div groupSize
  assert wq.len == m * k
  assert wScales.len == m * gpr
  assert xq.len == n * k
  assert xScales.len == n * gpr
  assert y.len == m * n
  assert k mod groupSize == 0
  when lpUseDotProd:
    let wbase0 = cast[int](unsafeAddr wq[0])
    let xbase0 = cast[int](unsafeAddr xq[0])
    for row in 0 ..< m:
      let wrow = wbase0 + row * k
      for g in 0 ..< gpr:
        let wg = wrow + g * groupSize # this group's weights: L1-hot across columns
        let ws = wScales[row * gpr + g]
        for c in 0 ..< n:
          var xa = xbase0 + c * k + g * groupSize
          var wa = wg
          var a0 = vdupq_n_s32(0'i32)
          var a1 = vdupq_n_s32(0'i32)
          var a2 = vdupq_n_s32(0'i32)
          var a3 = vdupq_n_s32(0'i32)
          var left = groupSize
          while left >= 64:
            a0 =
              vdotq_s32(a0, vld1q_s8(cast[ptr int8](wa)), vld1q_s8(cast[ptr int8](xa)))
            a1 = vdotq_s32(
              a1, vld1q_s8(cast[ptr int8](wa + 16)), vld1q_s8(cast[ptr int8](xa + 16))
            )
            a2 = vdotq_s32(
              a2, vld1q_s8(cast[ptr int8](wa + 32)), vld1q_s8(cast[ptr int8](xa + 32))
            )
            a3 = vdotq_s32(
              a3, vld1q_s8(cast[ptr int8](wa + 48)), vld1q_s8(cast[ptr int8](xa + 48))
            )
            wa += 64
            xa += 64
            left -= 64
          while left >= 16:
            a0 =
              vdotq_s32(a0, vld1q_s8(cast[ptr int8](wa)), vld1q_s8(cast[ptr int8](xa)))
            wa += 16
            xa += 16
            left -= 16
          var isum = vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)))
          while left > 0:
            isum += int32(cast[ptr int8](wa)[]) * int32(cast[ptr int8](xa)[])
            wa += 1
            xa += 1
            dec left
          let term = applyScales(isum, ws, xScales[c * gpr + g])
          if g == 0:
            y[row * n + c] = term
          else:
            y[row * n + c] = y[row * n + c] + term
  else:
    for row in 0 ..< m:
      for c in 0 ..< n:
        var total = 0.0'f32
        for g in 0 ..< gpr:
          var isum = 0'i32
          for kk in g * groupSize ..< (g + 1) * groupSize:
            isum += int32(int8(wq[row * k + kk])) * int32(int8(xq[c * k + kk]))
          total += applyScales(isum, wScales[row * gpr + g], xScales[c * gpr + g])
        y[row * n + c] = total

proc dequantGemmI4Q8*(
    wpacked: openArray[byte],
    wScales: openArray[float32],
    groupSize: int,
    xq: openArray[I8],
    xScales: openArray[float32],
    n: int,
    y: var openArray[float32],
) =
  ## The same over PACKED int4 weights (2 per byte, low nibble = even column).
  ## Same layouts and the same bit-for-bit-per-column guarantee.
  let m = y.len div n
  let k = xq.len div n
  let gpr = k div groupSize
  let rowBytes = k div 2
  assert wpacked.len == m * rowBytes
  assert wScales.len == m * gpr
  assert xq.len == n * k
  assert xScales.len == n * gpr
  assert y.len == m * n
  assert k mod groupSize == 0
  assert k mod 2 == 0
  when lpUseDotProd:
    let m0f = vdupq_n_u8(0x0f'u8)
    let sh4pos = vdupq_n_s8(4'i8)
    let sh4neg = vdupq_n_s8(-4'i8)
    let wbase0 = cast[int](unsafeAddr wpacked[0])
    let xbase0 = cast[int](unsafeAddr xq[0])
    for row in 0 ..< m:
      let wrow = wbase0 + row * rowBytes
      for g in 0 ..< gpr:
        let wg = wrow + (g * groupSize) shr 1
        let ws = wScales[row * gpr + g]
        for c in 0 ..< n:
          var xa = xbase0 + c * k + g * groupSize
          var wa = wg
          var a0 = vdupq_n_s32(0'i32)
          var a1 = vdupq_n_s32(0'i32)
          var a2 = vdupq_n_s32(0'i32)
          var a3 = vdupq_n_s32(0'i32)
          var left = groupSize
          while left >= 64:
            let xvA = vld2q_s8(cast[ptr int8](xa))
            let xvB = vld2q_s8(cast[ptr int8](xa + 32))
            let pvA = vld1q_u8(cast[ptr uint8](wa))
            let pvB = vld1q_u8(cast[ptr uint8](wa + 16))
            let loA = vshlq_s8(
              vshlq_s8(vreinterpretq_s8_u8(vandq_u8(pvA, m0f)), sh4pos), sh4neg
            )
            let hiA = vshlq_s8(
              vshlq_s8(vreinterpretq_s8_u8(vshlq_u8(pvA, sh4neg)), sh4pos), sh4neg
            )
            let loB = vshlq_s8(
              vshlq_s8(vreinterpretq_s8_u8(vandq_u8(pvB, m0f)), sh4pos), sh4neg
            )
            let hiB = vshlq_s8(
              vshlq_s8(vreinterpretq_s8_u8(vshlq_u8(pvB, sh4neg)), sh4pos), sh4neg
            )
            a0 = vdotq_s32(a0, loA, xvA.val[0])
            a1 = vdotq_s32(a1, hiA, xvA.val[1])
            a2 = vdotq_s32(a2, loB, xvB.val[0])
            a3 = vdotq_s32(a3, hiB, xvB.val[1])
            wa += 32
            xa += 64
            left -= 64
          while left >= 32:
            let xv = vld2q_s8(cast[ptr int8](xa))
            let pv = vld1q_u8(cast[ptr uint8](wa))
            let loS =
              vshlq_s8(vshlq_s8(vreinterpretq_s8_u8(vandq_u8(pv, m0f)), sh4pos), sh4neg)
            let hiS = vshlq_s8(
              vshlq_s8(vreinterpretq_s8_u8(vshlq_u8(pv, sh4neg)), sh4pos), sh4neg
            )
            a0 = vdotq_s32(a0, loS, xv.val[0])
            a1 = vdotq_s32(a1, hiS, xv.val[1])
            wa += 16
            xa += 32
            left -= 32
          var isum = vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)))
          var col = 0
          while left > 0:
            let b = cast[ptr uint8](wa + (col shr 1))[]
            isum +=
              int32(
                int8(
                  fromNibble(
                    if (col and 1) == 1:
                      b shr 4
                    else:
                      b and 0x0f'u8
                  )
                )
              ) * int32(cast[ptr int8](xa + col)[])
            inc col
            dec left
          let term = applyScales(isum, ws, xScales[c * gpr + g])
          if g == 0:
            y[row * n + c] = term
          else:
            y[row * n + c] = y[row * n + c] + term
  else:
    template nib(b: byte, odd: bool): int32 =
      int32(
        int8(
          fromNibble(
            if odd:
              b shr 4
            else:
              b and 0x0f'u8
          )
        )
      )

    for row in 0 ..< m:
      for c in 0 ..< n:
        var total = 0.0'f32
        for g in 0 ..< gpr:
          var isum = 0'i32
          for kk in g * groupSize ..< (g + 1) * groupSize:
            isum +=
              nib(wpacked[row * rowBytes + (kk shr 1)], (kk and 1) == 1) *
              int32(int8(xq[c * k + kk]))
          total += applyScales(isum, wScales[row * gpr + g], xScales[c * gpr + g])
        y[row * n + c] = total

# ---------------- quantize direction (elementwise, bit-exact vs scalar) ----------------
#
# The encode side was the largest gap left in the library: scalar `quantizeQ8_0`
# ran 428 Melem/s against ggml's 4622. The vector path reproduces the scalar
# semantics EXACTLY: `round()` is round-to-nearest-ties-AWAY in Nim, and
# `vcvtaq_s32_f32` is the same rounding in hardware, so no fp gymnastics are
# needed. x86 has no ties-away convert (only RNE), so these fall back to the
# scalar procs there rather than ship a near-miss — the AVX2 CI leg therefore
# tests the same bits as arm64.

when lpUseNeon:
  # round 4 floats ties-away, clamp to [lo,hi] as int32
  template rndClamp(v, lo, hi: untyped): int32x4 =
    vminq_s32(vmaxq_s32(vcvtaq_s32_f32(v), vdupq_n_s32(lo)), vdupq_n_s32(hi))

  # 16 floats at `src+i` scaled by idv, rounded/clamped, narrowed to int8x16
  template quant16(src, i, idv, lo, hi: untyped): int8x16 =
    let r0 = rndClamp(vmulq_f32(vld1q_f32(unsafeAddr src[i]), idv), lo, hi)
    let r1 = rndClamp(vmulq_f32(vld1q_f32(unsafeAddr src[i + 4]), idv), lo, hi)
    let r2 = rndClamp(vmulq_f32(vld1q_f32(unsafeAddr src[i + 8]), idv), lo, hi)
    let r3 = rndClamp(vmulq_f32(vld1q_f32(unsafeAddr src[i + 12]), idv), lo, hi)
    vcombine_s8(
      vmovn_s16(vcombine_s16(vmovn_s32(r0), vmovn_s32(r1))),
      vmovn_s16(vcombine_s16(vmovn_s32(r2), vmovn_s32(r3))),
    )

  template absMax32(src, o: untyped): float32 =
    ## |·|-max of 32 floats. Max is order-independent, so the vector reduce is
    ## bit-exact vs the scalar scan.
    var mx = vabsq_f32(vld1q_f32(unsafeAddr src[o]))
    mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(unsafeAddr src[o + 4])))
    mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(unsafeAddr src[o + 8])))
    mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(unsafeAddr src[o + 12])))
    mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(unsafeAddr src[o + 16])))
    mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(unsafeAddr src[o + 20])))
    mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(unsafeAddr src[o + 24])))
    mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(unsafeAddr src[o + 28])))
    vmaxvq_f32(mx)

proc quantizeQ8_0Batch*(src: openArray[float32], blocks: var openArray[BlockQ8_0]) =
  ## Vector form of `quantizeQ8_0` — bit-exact (same amax, same d, same
  ## multiply-by-inverse, same ties-away rounding, same clamp).
  assert src.len == blocks.len * QK
  when lpUseNeon:
    for bi in 0 ..< blocks.len:
      let o = bi * QK
      let amax = absMax32(src, o)
      let d = amax / 127.0'f32
      let id = (if d != 0.0'f32: 1.0'f32 / d else: 0.0'f32)
      blocks[bi].d = toF16(d)
      let idv = vdupq_n_f32(id)
      vst1q_s8(cast[ptr int8](addr blocks[bi].qs[0]), quant16(src, o, idv, -127, 127))
      vst1q_s8(
        cast[ptr int8](addr blocks[bi].qs[16]), quant16(src, o + 16, idv, -127, 127)
      )
  else:
    quantizeQ8_0(src, blocks)

proc quantizeQ4_0Batch*(src: openArray[float32], blocks: var openArray[BlockQ4_0]) =
  ## Vector form of `quantizeQ4_0`. The max-|·|-KEEPING-SIGN search stays scalar
  ## on purpose: the scalar loop's "first strict > wins" tie rule is
  ## order-dependent, and a lane-parallel search can pick a different (equal-|·|,
  ## opposite-sign) element and flip d's sign. Quantization itself is vectorized.
  assert src.len == blocks.len * QK
  when lpUseNeon:
    for bi in 0 ..< blocks.len:
      let o = bi * QK
      var amax = 0.0'f32
      var vmax = 0.0'f32
      for i in 0 ..< QK:
        let v = src[o + i]
        if abs(v) > amax:
          amax = abs(v)
          vmax = v
      let d = vmax / -8.0'f32
      let id = (if d != 0.0'f32: 1.0'f32 / d else: 0.0'f32)
      blocks[bi].d = toF16(d)
      let idv = vdupq_n_f32(id)
      # scalar does clamp(round(x·id)+8, 0, 15): same here — round ties-away,
      # ADD 8, clamp in int32, narrow. Then byte = q0 | (q1 shl 4) lanewise.
      template q16(base: untyped): int8x16 =
        let e8 = vdupq_n_s32(8'i32)
        let r0 = vminq_s32(
          vmaxq_s32(
            vaddq_s32(
              vcvtaq_s32_f32(vmulq_f32(vld1q_f32(unsafeAddr src[base]), idv)), e8
            ),
            vdupq_n_s32(0),
          ),
          vdupq_n_s32(15),
        )
        let r1 = vminq_s32(
          vmaxq_s32(
            vaddq_s32(
              vcvtaq_s32_f32(vmulq_f32(vld1q_f32(unsafeAddr src[base + 4]), idv)), e8
            ),
            vdupq_n_s32(0),
          ),
          vdupq_n_s32(15),
        )
        let r2 = vminq_s32(
          vmaxq_s32(
            vaddq_s32(
              vcvtaq_s32_f32(vmulq_f32(vld1q_f32(unsafeAddr src[base + 8]), idv)), e8
            ),
            vdupq_n_s32(0),
          ),
          vdupq_n_s32(15),
        )
        let r3 = vminq_s32(
          vmaxq_s32(
            vaddq_s32(
              vcvtaq_s32_f32(vmulq_f32(vld1q_f32(unsafeAddr src[base + 12]), idv)), e8
            ),
            vdupq_n_s32(0),
          ),
          vdupq_n_s32(15),
        )
        vcombine_s8(
          vmovn_s16(vcombine_s16(vmovn_s32(r0), vmovn_s32(r1))),
          vmovn_s16(vcombine_s16(vmovn_s32(r2), vmovn_s32(r3))),
        )

      let lo = vreinterpretq_u8_s8(q16(o)) # lanes 0..15  -> low nibbles
      let hi = vreinterpretq_u8_s8(q16(o + 16)) # lanes 16..31 -> high nibbles
      vst1q_u8(
        cast[ptr uint8](addr blocks[bi].qs[0]),
        vorrq_u8(lo, vshlq_u8(hi, vdupq_n_s8(4'i8))),
      )
  else:
    quantizeQ4_0(src, blocks)

proc quantizeBatch*(x: openArray[float32], p: QParams, dst: var openArray[I8]) =
  ## Vector form of `quantize[I8]` for the symmetric case — the activation-side
  ## encode ahead of every int8-SDOT matvec. Bit-exact: the scalar path DIVIDES
  ## (`x / scale`), so this divides too (`vdivq_f32`), then rounds ties-away and
  ## clamps to [-128, 127] exactly like `toI8`. Asymmetric quant and non-NEON
  ## builds fall through to the exact scalar proc.
  assert x.len == dst.len
  when lpUseNeon:
    if p.zeroPoint.len == 0:
      let n = x.len
      let gsize = if p.scheme == qPerTensor: n else: p.groupSize
      var base = 0
      var g = 0
      while base < n:
        let sv = vdupq_n_f32(p.scale[if p.scheme == qPerTensor: 0 else: g])
        let hi = min(base + gsize, n)
        var i = base
        while i + 16 <= hi:
          template r4(off: untyped): int32x4 =
            rndClamp(vdivq_f32(vld1q_f32(unsafeAddr x[i + off]), sv), -128, 127)

          let v = vcombine_s8(
            vmovn_s16(vcombine_s16(vmovn_s32(r4(0)), vmovn_s32(r4(4)))),
            vmovn_s16(vcombine_s16(vmovn_s32(r4(8)), vmovn_s32(r4(12)))),
          )
          vst1q_s8(cast[ptr int8](addr dst[i]), v)
          i += 16
        while i < hi:
          dst[i] = toI8(x[i] / p.scale[if p.scheme == qPerTensor: 0 else: g])
          inc i
        base = hi
        inc g
      return
  quantize(x, p, dst)

{.pop.}
