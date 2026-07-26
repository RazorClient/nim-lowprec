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
## Per-CHANNEL quant needs an axis + shape → that's the tensor layer's job.

import ./common

type
  QScheme* = enum
    qPerTensor
    qPerGroup

  QParams* = object
    scheme*: QScheme
    groupSize*: int             ## used by qPerGroup
    scale*: seq[float32]        ## len 1 (per-tensor) or ⌈n/groupSize⌉ (per-group)
    zeroPoint*: seq[float32]    ## empty ⇒ symmetric; else same length as `scale`

func scaleIndex(p: QParams, i: int): int {.inline.} =
  if p.scheme == qPerTensor: 0 else: i div p.groupSize

proc quantize*[T: LowPrec](x: openArray[float32]; p: QParams; dst: var openArray[T]) =
  ## dst ← round(x/scale + zeroPoint), clamped by T's range (done inside encode).
  mixin encode
  assert x.len == dst.len
  let asym = p.zeroPoint.len > 0
  for i in 0 ..< x.len:
    let si = scaleIndex(p, i)
    let z = if asym: p.zeroPoint[si] else: 0.0'f32
    dst[i] = encode(x[i] / p.scale[si] + z, T)

proc dequantize*[T: LowPrec](q: openArray[T]; p: QParams; dst: var openArray[float32]) =
  ## dst ← (q - zeroPoint)·scale.
  mixin decode
  assert q.len == dst.len
  let asym = p.zeroPoint.len > 0
  for i in 0 ..< q.len:
    let si = scaleIndex(p, i)
    let z = if asym: p.zeroPoint[si] else: 0.0'f32
    dst[i] = (decode(q[i]) - z) * p.scale[si]

proc calibrateSymmetric*(x: openArray[float32]; groupSize: int; qmax: float32): QParams =
  ## Symmetric per-group scales from abs-max:  scale = groupAbsMax / qmax.
  ## `groupSize <= 0` ⇒ per-tensor. (abs-max is the ONLY calibration the
  ## substrate offers; anything data-driven beyond it is an algorithm, not a
  ## primitive.) `qmax` is the target type's max magnitude, e.g. 127 / 7.
  if groupSize <= 0:
    var amax = 0.0'f32
    for v in x:
      if abs(v) > amax: amax = abs(v)
    result.scheme = qPerTensor
    result.scale = @[(if amax == 0.0'f32: 1.0'f32 else: amax / qmax)]
  else:
    result.scheme = qPerGroup
    result.groupSize = groupSize
    let ng = (x.len + groupSize - 1) div groupSize
    result.scale = newSeq[float32](ng)
    for g in 0 ..< ng:
      var amax = 0.0'f32
      for i in g * groupSize ..< min((g + 1) * groupSize, x.len):
        if abs(x[i]) > amax: amax = abs(x[i])
      result.scale[g] = (if amax == 0.0'f32: 1.0'f32 else: amax / qmax)
