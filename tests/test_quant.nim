## Tests for the quantization primitives: abs-max calibration + affine
## quantize/dequantize round-trip within one quantization step.

import std/[unittest, math]
import nim_lowprec/[intx, mxfloat, quant]

suite "quant round-trip":
  test "int8, per-tensor: reconstruction error ≤ step/2":
    var x = newSeq[float32](64)
    for i in 0 ..< 64:
      x[i] = sin(float32(i) * 0.2'f32) * 3.7'f32
    let p = calibrateSymmetric(x, 0, 127.0'f32)
    check p.scheme == qPerTensor
    var q = newSeq[I8](64)
    quantize(x, p, q)
    var y = newSeq[float32](64)
    dequantize(q, p, y)
    let step = p.scale[0]
    for i in 0 ..< 64:
      check abs(y[i] - x[i]) <= step * 0.5'f32 + 1e-5'f32

  test "int4, per-group (16): 4 scales + round-trip within step":
    var x = newSeq[float32](64)
    for i in 0 ..< 64:
      x[i] = (float32(i) - 32.0'f32) * 0.3'f32
    let p = calibrateSymmetric(x, 16, 7.0'f32) # int4 qmax = 7
    check p.scheme == qPerGroup
    check p.scale.len == 4
    var q = newSeq[I4](64)
    quantize(x, p, q)
    var y = newSeq[float32](64)
    dequantize(q, p, y)
    for g in 0 ..< 4:
      let step = p.scale[g]
      for i in g * 16 ..< (g + 1) * 16:
        check abs(y[i] - x[i]) <= step * 0.5'f32 + 1e-5'f32

  test "all-zero group gets a safe unit scale (no div-by-zero)":
    let x = newSeq[float32](8) # all zeros
    let p = calibrateSymmetric(x, 0, 127.0'f32)
    check p.scale[0] == 1.0'f32
    var q = newSeq[I8](8)
    quantize(x, p, q)
    var y = newSeq[float32](8)
    dequantize(q, p, y)
    for v in y:
      check v == 0.0'f32

suite "MX block quant (microscaling)":
  test "MXFP4: E8M0 power-of-two scales, and per-block beats per-tensor":
    var x = newSeq[float32](512)
    for i in 0 ..< 512: # magnitude grows across blocks
      let g = i div 64
      x[i] = sin(float32(i) * 0.05'f32) * float32(1 + g * g)
    let pmx = calibrateMX(x, 32, 2) # MXFP4 emax = 2
    check pmx.scheme == qMXBlock
    check pmx.scale.len == 16
    for s in pmx.scale: # scales are exact E8M0 powers of two
      check toE8M0(s).toFloat32 == s
    var qmx = newSeq[F4E2M1](512)
    quantize(x, pmx, qmx)
    var rmx = newSeq[float32](512)
    dequantize(qmx, pmx, rmx)
    let pt = calibrateMX(x, 512, 2) # a single per-tensor scale, for comparison
    var qt = newSeq[F4E2M1](512)
    quantize(x, pt, qt)
    var rt = newSeq[float32](512)
    dequantize(qt, pt, rt)
    var mseMx, mseT = 0.0'f64
    for i in 0 ..< 512:
      mseMx += float64((rmx[i] - x[i]) * (rmx[i] - x[i]))
      mseT += float64((rt[i] - x[i]) * (rt[i] - x[i]))
    check mseMx < mseT # microscaling reduces error

  test "all-zero block → unit scale, exact zeros":
    let x = newSeq[float32](64)
    let p = calibrateMX(x, 32, 2)
    for s in p.scale:
      check s == 1.0'f32
    var q = newSeq[F4E2M1](64)
    quantize(x, p, q)
    var y = newSeq[float32](64)
    dequantize(q, p, y)
    for v in y:
      check v == 0.0'f32

suite "asymmetric quant":
  test "int8 asymmetric: one-sided range round-trips + beats symmetric":
    var x = newSeq[float32](128) # all in [0.5, 4.0] (post-ReLU-ish)
    for i in 0 ..< 128:
      x[i] = 0.5'f32 + abs(sin(float32(i) * 0.13'f32)) * 3.5'f32
    let pa = calibrateAsymmetric(x, 0, -128.0'f32, 127.0'f32)
    check pa.scheme == qPerTensor
    check pa.zeroPoint.len == 1
    var qa = newSeq[I8](128)
    quantize(x, pa, qa)
    var ra = newSeq[float32](128)
    dequantize(qa, pa, ra)
    for i in 0 ..< 128: # round-trip within half a step
      check abs(ra[i] - x[i]) <= pa.scale[0] * 0.5'f32 + 1e-4'f32
    # asymmetric uses the full code range on one-sided data → lower error than symmetric
    let ps = calibrateSymmetric(x, 0, 127.0'f32)
    var qs = newSeq[I8](128)
    quantize(x, ps, qs)
    var rs = newSeq[float32](128)
    dequantize(qs, ps, rs)
    var eA, eS = 0.0'f64
    for i in 0 ..< 128:
      eA += float64((ra[i] - x[i]) * (ra[i] - x[i]))
      eS += float64((rs[i] - x[i]) * (rs[i] - x[i]))
    check eA < eS

  test "per-group asymmetric: 4 groups, round-trip within step":
    var x = newSeq[float32](64)
    for i in 0 ..< 64:
      x[i] = float32(i) * 0.1'f32 - 1.0'f32 # ramps −1 → 5.3
    let p = calibrateAsymmetric(x, 16, -128.0'f32, 127.0'f32)
    check p.scheme == qPerGroup
    check p.scale.len == 4 and p.zeroPoint.len == 4
    var q = newSeq[I8](64)
    quantize(x, p, q)
    var y = newSeq[float32](64)
    dequantize(q, p, y)
    for g in 0 ..< 4:
      for i in g * 16 ..< (g + 1) * 16:
        check abs(y[i] - x[i]) <= p.scale[g] * 0.5'f32 + 1e-4'f32
