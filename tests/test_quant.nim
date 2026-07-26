## Tests for the quantization primitives: abs-max calibration + affine
## quantize/dequantize round-trip within one quantization step.

import std/[unittest, math]
import nim_lowprec/[intx, quant]

suite "quant round-trip":

  test "int8, per-tensor: reconstruction error ≤ step/2":
    var x = newSeq[float32](64)
    for i in 0 ..< 64: x[i] = sin(float32(i) * 0.2'f32) * 3.7'f32
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
    for i in 0 ..< 64: x[i] = (float32(i) - 32.0'f32) * 0.3'f32
    let p = calibrateSymmetric(x, 16, 7.0'f32)              # int4 qmax = 7
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
    let x = newSeq[float32](8)                              # all zeros
    let p = calibrateSymmetric(x, 0, 127.0'f32)
    check p.scale[0] == 1.0'f32
    var q = newSeq[I8](8)
    quantize(x, p, q)
    var y = newSeq[float32](8)
    dequantize(q, p, y)
    for v in y: check v == 0.0'f32
