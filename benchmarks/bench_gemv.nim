## Fused dequant-GEMV vs a scalar reference loop, over the three weight layouts:
## int8, packed int4, packed MXFP4 + power-of-two block scales.
##
## The point of fusion: quantized weights are dequantized in-register and never
## materialized as an fp32 matrix. Both paths here are fused (the scalar reference
## decodes in its inner loop too), so the speedup is the vectorization alone —
## `bench_ggml` measures what fusion itself buys on top of that.
##
## A reduction, so scalar and SIMD summation orders differ: checksums are compared
## within a relative tolerance, not bit-for-bit.
##
## The reference loops take their operands as `openArray` params, exactly like the
## kernels do. Reading module-level `seq`s instead would block the C compiler's
## auto-vectorization (the globals may alias) and hand the SIMD row free speedup
## that has nothing to do with intrinsics.

import nim_lowprec/[intx, mxfloat, simd/dequant]
import ./harness

const
  M = 4096
  K = 4096
  gs = 128
  gpr = K div gs
  rowBytes = K div 2
  shape = "M = " & $M & ", K = " & $K & ", group = " & $gs
  flops = 2.0 * M.float * K.float
  mid = M div 2

var scales = newSeq[float32](M * gpr)
var x = newSeq[float32](K)
var y = newSeq[float32](M)
for i in 0 ..< scales.len: scales[i] = 0.02'f32
for i in 0 ..< K: x[i] = float32(i mod 7) * 0.1'f32 - 0.3'f32

header "fused dequant-GEMV"

block: # ---- int8 weights ----
  var wq = newSeq[I8](M * K)
  for i in 0 ..< wq.len: wq[i] = toI8(float32((i mod 15) - 7))

  proc gemvScalar(wq: openArray[I8]; scales, x: openArray[float32];
                  y: var openArray[float32]) =
    for row in 0 ..< M:
      var total = 0.0'f32
      for g in 0 ..< gpr:
        let s = scales[row * gpr + g]
        var p = 0.0'f32
        for kk in g * gs ..< (g + 1) * gs:
          p += float32(int8(wq[row * K + kk])) * x[kk]
        total += p * s
      y[row] = total

  var g = flopGroup("int8 GEMV", shape, flops)
  g.scalarRow y[mid]:
    gemvScalar(wq, scales, x, y)
  g.kernelRow y[mid]:
    dequantGemv(wq, scales, gs, x, y)
  g.report()

block: # ---- packed int4 weights (2 per byte, low nibble first) ----
  var w4 = newSeq[byte](M * rowBytes)
  for r in 0 ..< M:
    for c in 0 ..< K:
      let nib = nibble(toI4(float32((c mod 15) - 7)))
      let bi = r * rowBytes + (c shr 1)
      if (c and 1) == 0: w4[bi] = nib
      else:              w4[bi] = w4[bi] or (nib shl 4)

  proc gemvI4Scalar(w4: openArray[byte]; scales, x: openArray[float32];
                    y: var openArray[float32]) =
    for row in 0 ..< M:
      var total = 0.0'f32
      for g in 0 ..< gpr:
        let s = scales[row * gpr + g]
        var p = 0.0'f32
        for c in g * gs ..< (g + 1) * gs:
          let b = w4[row * rowBytes + (c shr 1)]
          let nib = if (c and 1) == 0: b and 0x0f'u8 else: b shr 4
          p += fromNibble(nib).toFloat32 * x[c]
        total += p * s
      y[row] = total

  var g = flopGroup("int4 GEMV  (packed)", shape, flops)
  g.scalarRow y[mid]:
    gemvI4Scalar(w4, scales, x, y)
  g.kernelRow y[mid]:
    dequantGemvI4(w4, scales, gs, x, y)
  g.report()

block: # ---- packed MXFP4 weights + power-of-two (E8M0) block scales ----
  var wf4 = newSeq[byte](M * rowBytes)
  for r in 0 ..< M:
    for c in 0 ..< K:
      let nib = bits(toF4E2M1(float32((c mod 9) - 4) * 0.5'f32)) and 0x0f'u8
      let bi = r * rowBytes + (c shr 1)
      if (c and 1) == 0: wf4[bi] = nib
      else:              wf4[bi] = wf4[bi] or (nib shl 4)
  # E8M0 scales are powers of two by construction, so folding them is exact.
  var mxScales = newSeq[float32](M * gpr)
  for i in 0 ..< mxScales.len: mxScales[i] = 0.03125'f32   # 2^-5

  # The reference decodes each code with `toFloat32(F4E2M1)` — the library's
  # scalar path, which goes through float64 tinyfloat math. The kernel's byte-LUT
  # skips all of that, so this row's speedup is vectorization AND a much cheaper
  # decode; it is the honest "what a caller gets today" comparison, not a pure
  # lane-width one.
  proc gemvF4Scalar(wf4: openArray[byte]; mxScales, x: openArray[float32];
                    y: var openArray[float32]) =
    for row in 0 ..< M:
      var total = 0.0'f32
      for g in 0 ..< gpr:
        let s = mxScales[row * gpr + g]
        var p = 0.0'f32
        for c in g * gs ..< (g + 1) * gs:
          let b = wf4[row * rowBytes + (c shr 1)]
          let nib = if (c and 1) == 0: b and 0x0f'u8 else: b shr 4
          p += toFloat32(F4E2M1(nib)) * x[c]
        total += p * s
      y[row] = total

  var g = flopGroup("mxfp4 GEMV  (packed + E8M0 scales)", shape, flops)
  g.scalarRow y[mid]:
    gemvF4Scalar(wf4, mxScales, x, y)
  g.kernelRow y[mid]:
    dequantGemvF4(wf4, mxScales, gs, x, y)
  g.report()
