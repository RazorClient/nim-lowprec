## nim_lowprec/quant — quantization REPRESENTATION + mechanical (de)quantize.
##
## Owns the scale/zero-point container and the affine round-trip, generic over
## any LowPrec target (I8, I4, fp8, …). It does NOT own the algorithms that
## CHOOSE scales from data (GPTQ / AWQ / EXL3 / SmoothQuant) — those consume a
## calibration dataset and live one layer up. The one stateless calibration a
## substrate may offer is abs-max, provided here as a convenience.
##
## Schemes that work on a flat array without tensor shape:
##   qPerTensor  one scale for everything
##   qPerGroup   one scale per contiguous block of `groupSize` (the LLM case)
##   qMXBlock    OCP Microscaling: a power-of-two (E8M0) scale per block — pair
##               with an MX element type (MXFP4/6/8); see `calibrateMX`
## Per-CHANNEL quant needs an axis + shape → that's the tensor layer's job.

import std/math
import ./common

type
  QScheme* = enum
    qPerTensor
    qPerGroup
    qMXBlock ## OCP Microscaling: E8M0 power-of-two scale per block

  QParams* = object
    scheme*: QScheme
    groupSize*: int ## used by qPerGroup
    scale*: seq[float32] ## len 1 (per-tensor) or ⌈n/groupSize⌉ (per-group)
    zeroPoint*: seq[float32] ## empty ⇒ symmetric; else same length as `scale`

func scaleIndex(p: QParams, i: int): int {.inline.} =
  if p.scheme == qPerTensor:
    0
  else:
    i div p.groupSize

func numGroups(n, groupSize: int): int {.inline.} =
  ## Number of contiguous groups of `groupSize` needed to cover `n` (ceil-div).
  (n + groupSize - 1) div groupSize

iterator groupBounds(n, groupSize: int): (int, int, int) =
  ## Yields (g, lo, hi): each contiguous group's index and half-open [lo, hi)
  ## bounds, tiling [0, n). The last group is short when `groupSize` does not
  ## divide `n`; every yielded group is non-empty (lo < hi).
  for g in 0 ..< numGroups(n, groupSize):
    yield (g, g * groupSize, min((g + 1) * groupSize, n))

proc quantize*[T: LowPrec](x: openArray[float32], p: QParams, dst: var openArray[T]) =
  ## dst ← round(x/scale + zeroPoint), clamped by T's range (done inside encode).
  mixin encode
  assert x.len == dst.len
  let asym = p.zeroPoint.len > 0
  for i in 0 ..< x.len:
    let si = scaleIndex(p, i)
    let z =
      if asym:
        p.zeroPoint[si]
      else:
        0.0'f32
    dst[i] = encode(x[i] / p.scale[si] + z, T)

proc dequantize*[T: LowPrec](q: openArray[T], p: QParams, dst: var openArray[float32]) =
  ## dst ← (q - zeroPoint)·scale.
  mixin decode
  assert q.len == dst.len
  let asym = p.zeroPoint.len > 0
  for i in 0 ..< q.len:
    let si = scaleIndex(p, i)
    let z =
      if asym:
        p.zeroPoint[si]
      else:
        0.0'f32
    dst[i] = (decode(q[i]) - z) * p.scale[si]

func sliceAbsMax(x: openArray[float32], lo, hi: int): float32 {.inline.} =
  ## max |x[i]| over the slice x[lo..<hi]; 0.0 for an empty slice.
  result = 0.0'f32
  for i in lo ..< hi:
    let a = abs(x[i])
    if a > result:
      result = a

func absMaxScale(
    x: openArray[float32], lo, hi: int, qmax: float32
): float32 {.inline.} =
  ## Symmetric scale for the slice x[lo..<hi]: absMax/qmax, or 1.0 for an
  ## all-zero slice (so the round-trip is the identity, not a divide-by-zero).
  let amax = sliceAbsMax(x, lo, hi)
  if amax == 0.0'f32:
    1.0'f32
  else:
    amax / qmax

proc calibrateSymmetric*(
    x: openArray[float32], groupSize: int, qmax: float32
): QParams =
  ## Symmetric per-group scales from abs-max:  scale = groupAbsMax / qmax.
  ## `groupSize <= 0` ⇒ per-tensor. (abs-max is the ONLY calibration the
  ## substrate offers; anything data-driven beyond it is an algorithm, not a
  ## primitive.) `qmax` is the target type's max magnitude, e.g. 127 / 7.
  if groupSize <= 0:
    result.scheme = qPerTensor
    result.scale = @[absMaxScale(x, 0, x.len, qmax)]
  else:
    result.scheme = qPerGroup
    result.groupSize = groupSize
    result.scale = newSeq[float32](numGroups(x.len, groupSize))
    for g, lo, hi in groupBounds(x.len, groupSize):
      result.scale[g] = absMaxScale(x, lo, hi, qmax)

func mxSharedScale(amax: float32, elemEmax: int): float32 {.inline.} =
  ## OCP MX shared scale for one block: 2^(floor(log2(amax)) - elemEmax),
  ## clamped to the E8M0 exponent range [-127, 127]. An all-zero block → 1.
  if amax == 0.0'f32:
    return 1.0'f32
  let (_, exp) = frexp(amax.float64) # amax = frac·2^exp, frac ∈ [0.5,1)
  var k = (exp - 1) - elemEmax # floor(log2(amax)) = exp-1
  if k > 127:
    k = 127
  elif k < -127:
    k = -127
  float32(cast[float64](uint64(k + 1023) shl 52)) # 2^k, exact for |k| ≤ 127

proc calibrateMX*(x: openArray[float32], blockSize, elemEmax: int): QParams =
  ## OCP Microscaling calibration: one power-of-two (E8M0-representable) shared
  ## scale per `blockSize` elements (spec default 32), = 2^(floor(log2(blockAbsMax))
  ## - elemEmax). Feed the result to the generic `quantize`/`dequantize` with an
  ## MX element type. `elemEmax` is the element format's largest normal exponent:
  ##   MXFP4 e2m1 → 2   MXFP6 e2m3 → 2   MXFP6 e3m2 → 4
  ##   MXFP8 e4m3 → 8   MXFP8 e5m2 → 15
  ## The scales are exact powers of two, so `toE8M0(scale).toFloat32 == scale`.
  result.scheme = qMXBlock
  result.groupSize = blockSize
  result.scale = newSeq[float32](numGroups(x.len, blockSize))
  for b, lo, hi in groupBounds(x.len, blockSize):
    result.scale[b] = mxSharedScale(sliceAbsMax(x, lo, hi), elemEmax)

func affineParams(
    x: openArray[float32], lo, hi: int, qmin, qmax: float32
): (float32, float32) =
  ## (scale, zeroPoint) mapping x[lo..<hi]'s [min,max] onto [qmin,qmax].
  ## A flat slice (max==min) gets unit scale so the round-trip is the identity.
  var mn = x[lo]
  var mx = x[lo]
  for i in lo + 1 ..< hi:
    if x[i] < mn:
      mn = x[i]
    if x[i] > mx:
      mx = x[i]
  let span = mx - mn
  let scale = (if span > 0.0'f32: span / (qmax - qmin) else: 1.0'f32)
  (scale, qmin - mn / scale)

proc calibrateAsymmetric*(
    x: openArray[float32], groupSize: int, qmin, qmax: float32
): QParams =
  ## Affine (asymmetric) calibration: map the data's [min,max] onto the integer
  ## range [qmin,qmax]. scale = (max-min)/(qmax-qmin); zeroPoint = qmin - min/scale.
  ## Pairs with quantize/dequantize (q = round(x/scale + zp); x ≈ (q-zp)·scale).
  ## Captures skewed / one-sided ranges that symmetric quant wastes half its
  ## codes on (e.g. post-ReLU activations). `groupSize <= 0` ⇒ per-tensor.
  if groupSize <= 0:
    result.scheme = qPerTensor
    let (s, z) = affineParams(x, 0, x.len, qmin, qmax)
    result.scale = @[s]
    result.zeroPoint = @[z]
  else:
    result.scheme = qPerGroup
    result.groupSize = groupSize
    let ng = numGroups(x.len, groupSize)
    result.scale = newSeq[float32](ng)
    result.zeroPoint = newSeq[float32](ng)
    for g, lo, hi in groupBounds(x.len, groupSize):
      let (s, z) = affineParams(x, lo, hi, qmin, qmax)
      result.scale[g] = s
      result.zeroPoint[g] = z
