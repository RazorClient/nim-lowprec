## nim_lowprec/simd/dequant — vectorized dequant + the fused dequant-GEMV.
##
## `dequantizeBatch` : value(q) * scale -> fp32 (bit-exact vs scalar `dequantize`).
## `dequantGemv`     : y = W_q · x, int8 weights × fp32 activations, with dequant
##                     folded IN-REGISTER (weights never materialised as fp32) —
##                     the quantized-decode hot path.
## `dequantGemvI4`   : the same fused GEMV over PACKED int4 weights (2 per byte);
##                     nibbles are unpacked + sign-extended in-register too.
## `dequantGemvF4`   : the same for PACKED MXFP4 weights + E8M0 block scales;
##                     fp4 codes are LUT-decoded in-register (no sign-extend).
##
## NOTE ON TESTING: dequant is elementwise → bit-exact vs scalar. The GEMV is a
## REDUCTION → fp summation order differs from a scalar loop, so it is NOT
## bit-exact; it's checked against an fp64 reference within a tolerance. That
## split (exact for elementwise, tolerance for reductions) is the correct
## discipline. NEON + AVX2 paths, scalar fallback elsewhere.

import ../intx, ../quant, ../float16, ../ggml
import ./target
when lpUseNeon: import ./neon
when lpUseAvx2: import ./x86

# 2× the MXFP4 (e2m1) value for each 4-bit code — all integers, so fp4 decode is
# a byte-LUT straight into the int8 widen+FMA path; the ×2 is undone by halving
# the block scale. Codes 8..15 are the negatives (code 8 = -0 → 0).
const mxfp4Lut2x: array[16, int8] =
  [int8 0, 1, 2, 3, 4, 6, 8, 12,   0, -1, -2, -3, -4, -6, -8, -12]

when lpUseNeon:
  # widen one int8x16 (16 weights for columns `col..col+15`) and FMA into `acc`.
  template fmaBlock(acc, w, x, col: untyped) =
    let l16 = vmovl_s8(vget_low_s8(w))
    let h16 = vmovl_s8(vget_high_s8(w))
    acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(l16))),  vld1q_f32(unsafeAddr x[col]))
    acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(l16))), vld1q_f32(unsafeAddr x[col + 4]))
    acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(h16))),  vld1q_f32(unsafeAddr x[col + 8]))
    acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(h16))), vld1q_f32(unsafeAddr x[col + 12]))

  # widen 16 int8 (base i) -> 4×float32x4, multiply by scalar s, store to dst.
  template blockS8(q, dst, i, s: untyped) =
    let v = vld1q_s8(cast[ptr int8](unsafeAddr q[i]))
    let loH = vmovl_s8(vget_low_s8(v))
    let hiH = vmovl_s8(vget_high_s8(v))
    vst1q_f32(addr dst[i],      vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(loH))),  s))
    vst1q_f32(addr dst[i + 4],  vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(loH))), s))
    vst1q_f32(addr dst[i + 8],  vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hiH))),  s))
    vst1q_f32(addr dst[i + 12], vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hiH))), s))

when lpUseAvx2:
  # widen 16 int8 (columns col..col+15) -> 2×(8×f32) and FMA the activations into acc.
  template fmaBlockX86(acc, v, x, col: untyped) =
    acc = mm256_fmadd_ps(mm256_cvtepi32_ps(mm256_cvtepi8_epi32(v)),
                         mm256_loadu_ps(unsafeAddr x[col]), acc)
    acc = mm256_fmadd_ps(mm256_cvtepi32_ps(mm256_cvtepi8_epi32(mm_unpackhi_epi64(v, v))),
                         mm256_loadu_ps(unsafeAddr x[col + 8]), acc)

  # widen 16 int8 (base i) -> 2×(8×f32), multiply by scalar s, store to dst.
  template blockS8X86(q, dst, i, s: untyped) =
    let v = mm_loadu_si128(cast[ptr m128i](unsafeAddr q[i]))
    let sv = mm256_set1_ps(s)
    mm256_storeu_ps(addr dst[i],     mm256_mul_ps(mm256_cvtepi32_ps(mm256_cvtepi8_epi32(v)), sv))
    mm256_storeu_ps(addr dst[i + 8], mm256_mul_ps(mm256_cvtepi32_ps(mm256_cvtepi8_epi32(mm_unpackhi_epi64(v, v))), sv))

# ---------------- dequantizeBatch (elementwise, bit-exact vs scalar) ----------------

template defDequantBatch(T: untyped) =
  proc dequantizeBatch*(q: openArray[T]; p: QParams; dst: var openArray[float32]) =
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
            dst[i] = float32(int8(q[i])) * s; inc i
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
            dst[i] = float32(int8(q[i])) * s; inc i
          base = hi
          inc g
        return
    dequantize(q, p, dst)   # asymmetric / non-SIMD build: bit-exact scalar path

defDequantBatch(I8)
defDequantBatch(I4)

# ---------------- fused dequant-GEMV (reduction, tolerance-tested) ----------------

proc dequantGemv*(wq: openArray[I8]; scales: openArray[float32]; groupSize: int;
                  x: openArray[float32]; y: var openArray[float32]) =
  ## y[m] = Σ_k (int8(W[m,k]) * scale[m, k div groupSize]) * x[k].
  ## W is M×K row-major (`wq`), `scales` is M×(K div groupSize) row-major,
  ## x is length K, y is length M. Weights are dequantised IN-REGISTER — never
  ## written out as an fp32 matrix (that's the whole point of fused GEMV).
  let m = y.len
  let k = x.len
  let gpr = k div groupSize                         # groups per row
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
        var acc = vdupq_n_f32(0.0'f32)
        while kk + 16 <= gend:
          let wv = vld1q_s8(cast[ptr int8](unsafeAddr wq[wbase + kk]))
          fmaBlock(acc, wv, x, kk)              # widen 16 int8 + 4× FMA (shared with the I4 GEMV)
          kk += 16
        partial = vaddvq_f32(acc)
      elif lpUseAvx2:
        var acc = mm256_setzero_ps()
        while kk + 16 <= gend:
          let v = mm_loadu_si128(cast[ptr m128i](unsafeAddr wq[wbase + kk]))
          fmaBlockX86(acc, v, x, kk)
          kk += 16
        partial = hsum256(acc)
      while kk < gend:                              # scalar tail within group
        partial += float32(int8(wq[wbase + kk])) * x[kk]
        inc kk
      total += partial * s
    y[row] = total

proc dequantGemvI4*(wpacked: openArray[byte]; scales: openArray[float32]; groupSize: int;
                    x: openArray[float32]; y: var openArray[float32]) =
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
    let eight  = mm_set1_epi8(8)
    let cnt4   = mm_cvtsi32_si128(4.cint)
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
        var acc = vdupq_n_f32(0.0'f32)
        while kk + 32 <= gend:                     # 16 bytes = 32 int4 columns
          let pv  = vld1q_u8(cast[ptr uint8](unsafeAddr wpacked[wbase + (kk shr 1)]))
          let loN = vandq_u8(pv, m0f)              # low nibbles  (even columns)
          let hiN = vshlq_u8(pv, sh4neg)           # high nibbles (odd columns), >>4
          let loS = vshlq_s8(vshlq_s8(vreinterpretq_s8_u8(loN), sh4pos), sh4neg)  # sign-extend 4-bit
          let hiS = vshlq_s8(vshlq_s8(vreinterpretq_s8_u8(hiN), sh4pos), sh4neg)
          let wA = vzip1q_s8(loS, hiS)             # columns kk .. kk+15, in order
          let wB = vzip2q_s8(loS, hiS)             # columns kk+16 .. kk+31
          fmaBlock(acc, wA, x, kk)
          fmaBlock(acc, wB, x, kk + 16)
          kk += 32
        partial = vaddvq_f32(acc)
      elif lpUseAvx2:
        var acc = mm256_setzero_ps()
        while kk + 32 <= gend:                     # 16 bytes = 32 int4 columns
          let pv  = mm_loadu_si128(cast[ptr m128i](unsafeAddr wpacked[wbase + (kk shr 1)]))
          let loN = mm_and_si128(pv, mask0f)                     # low nibbles  (even columns)
          let hiN = mm_and_si128(mm_srl_epi16(pv, cnt4), mask0f) # high nibbles (odd columns), >>4
          # sign-extend a 4-bit value: (n xor 8) - 8  (per byte, shift-free)
          let loS = mm_sub_epi8(mm_xor_si128(loN, eight), eight)
          let hiS = mm_sub_epi8(mm_xor_si128(hiN, eight), eight)
          let wA = mm_unpacklo_epi8(loS, hiS)      # columns kk .. kk+15, in order
          let wB = mm_unpackhi_epi8(loS, hiS)      # columns kk+16 .. kk+31
          fmaBlockX86(acc, wA, x, kk)
          fmaBlockX86(acc, wB, x, kk + 16)
          kk += 32
        partial = hsum256(acc)
      while kk < gend:                             # scalar tail
        let b = wpacked[wbase + (kk shr 1)]
        let nib = if (kk and 1) == 0: b and 0x0f'u8 else: b shr 4
        partial += fromNibble(nib).toFloat32 * x[kk]
        inc kk
      total += partial * s
    y[row] = total

proc dequantGemvF4*(wpacked: openArray[byte]; scales: openArray[float32]; groupSize: int;
                    x: openArray[float32]; y: var openArray[float32]) =
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
      let s = scales[sbase + g] * 0.5'f32          # halve to undo the LUT's ×2
      let gstart = g * groupSize
      let gend = gstart + groupSize
      var partial = 0.0'f32
      var kk = gstart
      when lpUseNeon:
        var acc = vdupq_n_f32(0.0'f32)
        while kk + 32 <= gend:                     # 16 bytes = 32 fp4 columns
          let pv  = vld1q_u8(cast[ptr uint8](unsafeAddr wpacked[wbase + (kk shr 1)]))
          let loN = vandq_u8(pv, m0f)              # low nibbles  (even columns)
          let hiN = vshlq_u8(pv, sh4neg)           # high nibbles (odd columns), >>4
          let loV = vqtbl1q_s8(lut, loN)           # 2×value (even columns)
          let hiV = vqtbl1q_s8(lut, hiN)           # 2×value (odd columns)
          let wA = vzip1q_s8(loV, hiV)             # columns kk .. kk+15, in order
          let wB = vzip2q_s8(loV, hiV)             # columns kk+16 .. kk+31
          fmaBlock(acc, wA, x, kk)
          fmaBlock(acc, wB, x, kk + 16)
          kk += 32
        partial = vaddvq_f32(acc)
      elif lpUseAvx2:
        var acc = mm256_setzero_ps()
        while kk + 32 <= gend:
          let pv  = mm_loadu_si128(cast[ptr m128i](unsafeAddr wpacked[wbase + (kk shr 1)]))
          let loN = mm_and_si128(pv, mask0f)
          let hiN = mm_and_si128(mm_srl_epi16(pv, cnt4), mask0f)
          let loV = mm_shuffle_epi8(lut, loN)      # 2×value (even columns)
          let hiV = mm_shuffle_epi8(lut, hiN)      # 2×value (odd columns)
          let wA = mm_unpacklo_epi8(loV, hiV)      # columns kk .. kk+15
          let wB = mm_unpackhi_epi8(loV, hiV)      # columns kk+16 .. kk+31
          fmaBlockX86(acc, wA, x, kk)
          fmaBlockX86(acc, wB, x, kk + 16)
          kk += 32
        partial = hsum256(acc)
      while kk < gend:                             # scalar tail
        let b = wpacked[wbase + (kk shr 1)]
        let nib = if (kk and 1) == 0: b and 0x0f'u8 else: b shr 4
        partial += float32(mxfp4Lut2x[int(nib)]) * x[kk]
        inc kk
      total += partial * s
    y[row] = total

# ---------------- fused GEMV over ggml block formats (Q8_0, Q4_0) ----------------

proc dequantGemvQ8_0*(w: openArray[BlockQ8_0]; blocksPerRow: int;
                      x: openArray[float32]; y: var openArray[float32]) =
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
        var acc = vdupq_n_f32(0.0'f32)
        let w0 = vld1q_s8(cast[ptr int8](unsafeAddr blk.qs[0]))
        let w1 = vld1q_s8(cast[ptr int8](unsafeAddr blk.qs[16]))
        fmaBlock(acc, w0, x, base)
        fmaBlock(acc, w1, x, base + 16)
        partial = vaddvq_f32(acc)
      elif lpUseAvx2:
        var acc = mm256_setzero_ps()
        let v0 = mm_loadu_si128(cast[ptr m128i](unsafeAddr blk.qs[0]))
        let v1 = mm_loadu_si128(cast[ptr m128i](unsafeAddr blk.qs[16]))
        fmaBlockX86(acc, v0, x, base)
        fmaBlockX86(acc, v1, x, base + 16)
        partial = hsum256(acc)
      else:
        for i in 0 ..< QK: partial += float32(blk.qs[i]) * x[base + i]
      total += partial * blk.d.toFloat32
    y[row] = total

proc dequantGemvQ4_0*(w: openArray[BlockQ4_0]; blocksPerRow: int;
                      x: openArray[float32]; y: var openArray[float32]) =
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
        var acc = vdupq_n_f32(0.0'f32)
        let pv  = vld1q_u8(cast[ptr uint8](unsafeAddr blk.qs[0]))
        let loV = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pv, m0f)), eight)     # low nibbles − 8
        let hiV = vsubq_s8(vreinterpretq_s8_u8(vshlq_u8(pv, sh4neg)), eight)  # high nibbles − 8
        fmaBlock(acc, loV, x, base)          # lanes 0..15
        fmaBlock(acc, hiV, x, base + 16)     # lanes 16..31
        partial = vaddvq_f32(acc)
      elif lpUseAvx2:
        var acc = mm256_setzero_ps()
        let pv  = mm_loadu_si128(cast[ptr m128i](unsafeAddr blk.qs[0]))
        let loV = mm_sub_epi8(mm_and_si128(pv, mask0f), eight)
        let hiV = mm_sub_epi8(mm_and_si128(mm_srl_epi16(pv, cnt4), mask0f), eight)
        fmaBlockX86(acc, loV, x, base)
        fmaBlockX86(acc, hiV, x, base + 16)
        partial = hsum256(acc)
      else:
        for i in 0 ..< QK div 2:
          let b = blk.qs[i]
          partial += float32(int(b and 0x0f'u8) - 8) * x[base + i]
          partial += float32(int(b shr 4) - 8) * x[base + i + QK div 2]
      total += partial * blk.d.toFloat32
    y[row] = total
