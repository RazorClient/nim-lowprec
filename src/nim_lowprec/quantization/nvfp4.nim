## nim_lowprec/nvfp4 — the NVIDIA NVFP4 block-quantization scheme.
##
## NVFP4 is the FP4 accuracy leader (see CONFORMANCE.md): E2M1 elements in
## blocks of 16, a per-block **FP8-E4M3** decode scale, and a per-tensor fp32
## scale — two-level scaling, versus MXFP4's single power-of-two E8M0 scale per
## 32. Both element codecs (`F4E2M1`, `F8E4M3`) already existed; this module is
## just the scheme.
##
## The algorithm is TransformerEngine's `NVFP4QuantizerRef`
## (`_quantize_blockwise_reference`, 1D path, `pow_2_scales=false`), mirrored
## operation for operation in fp32 so the result is BIT-EXACT against a numpy
## transcription of that reference (tests/test_scheme_conformance.nim):
##
##   globalEncode = min(448·6 / amax(tensor), f32max)      (0 → 1)
##   globalDecode = 1 / globalEncode
##   per block:  ds  = clamp(amax(block) · (globalEncode · (1/6)), ±448) → e4m3
##               es  = min(1 / (f32(ds) · globalDecode), f32max)
##               q   = e2m1_rne(clamp(x · es, ±6))
##
## Dequantization is defined here as (f32(q) · f32(ds)) · globalDecode, in that
## association. Like the fp8 encoders, exact −0.0 encodes as code 0 (the TE
## boundary table folds −0 into +0).
##
## DYNAMIC-RANGE LIMIT (inherent to the format, faithfully reproduced): the
## per-block scale is an fp8 E4M3, so a block whose amax lies more than ~2^17
## below the tensor amax gets a scale that underflows to zero and the whole
## block dequantizes to 0 — the TransformerEngine reference does exactly the
## same. NVFP4 assumes tensor values within E4M3 scale range of each other;
## check `blockScales` for zeros if your data might not be.

import ../formats/mxfloat, ../formats/float8

# The multiply-then-subtract/multiply chains must round exactly like the numpy
# oracle: no compiler-invented FMAs (same policy as simd/dequant).
{.localPassc: "-ffp-contract=off".}

const nvfp4BlockLen* = 16 ## NVFP4 block length (vs 32 for MX)

proc quantizeNVFP4*(
    x: openArray[float32],
    codes: var openArray[F4E2M1],
    blockScales: var openArray[F8E4M3],
): float32 =
  ## Quantize `x` (length a multiple of 16) to NVFP4. Fills one E2M1 code per
  ## element and one E4M3 decode scale per 16-element block; returns the
  ## per-tensor GLOBAL DECODE SCALE (keep all three to dequantize).
  let n = x.len
  assert n mod nvfp4BlockLen == 0
  assert codes.len == n
  assert blockScales.len == n div nvfp4BlockLen
  const F32MAX = 3.4028234663852886e38'f32
  var gamax = 0.0'f32
  for v in x:
    let a = abs(v)
    if a > gamax:
      gamax = a
  var ges = min(2688.0'f32 / gamax, F32MAX) # 448 · 6
  if ges == 0.0'f32:
    ges = 1.0'f32
  let gds = 1.0'f32 / ges
  let mult = ges * (1.0'f32 / 6.0'f32)
  for b in 0 ..< blockScales.len:
    let lo = b * nvfp4BlockLen
    var vmax = 0.0'f32
    for i in lo ..< lo + nvfp4BlockLen:
      let a = abs(x[i])
      if a > vmax:
        vmax = a
    let ds = clamp(min(vmax * mult, F32MAX), -448.0'f32, 448.0'f32)
    let ds8 = toF8E4M3(ds)
    blockScales[b] = ds8
    let es = min(1.0'f32 / (ds8.toFloat32 * gds), F32MAX)
    for i in lo ..< lo + nvfp4BlockLen:
      let scaled = clamp(x[i] * es, -6.0'f32, 6.0'f32)
      # exact ±0 → code 0, matching the reference's boundary table (−0 → +0)
      codes[i] = (if scaled == 0.0'f32: F4E2M1(0) else: toF4E2M1(scaled))
  gds

proc dequantizeNVFP4*(
    codes: openArray[F4E2M1],
    blockScales: openArray[F8E4M3],
    globalDecodeScale: float32,
    dst: var openArray[float32],
) =
  ## x̂ = (value(code) · value(blockScale)) · globalDecodeScale.
  assert dst.len == codes.len
  assert blockScales.len * nvfp4BlockLen == codes.len
  for b in 0 ..< blockScales.len:
    let s = blockScales[b].toFloat32
    let lo = b * nvfp4BlockLen
    for i in lo ..< lo + nvfp4BlockLen:
      dst[i] = (codes[i].toFloat32 * s) * globalDecodeScale
