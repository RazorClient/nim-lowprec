## Multi-core scaling of the fused dequant-GEMV (`nim_lowprec/kernels/parallel`).
##
## The baseline row is the serial kernel; each following row is the same kernel
## row-sharded across N threads, so the speedup column IS the scaling curve.
## Threading is exact here (each row's arithmetic is untouched), so the checksums
## must match the serial row bit-for-bit — hence `tol = 0` on a GEMV group, which
## no other benchmark can claim.
##
## Expect this to stop scaling before it runs out of cores: threads are created per
## call (no pool), and `countProcessors()` counts E-cores that don't carry a
## bandwidth-hungry kernel as well as a P-core.

import std/cpuinfo
import nim_lowprec/[formats/intx, formats/float16, quantization/ggml, simd/dequant, kernels/parallel]
import ./harness

const
  M = 4096
  K = 4096
  gs = 128
  gpr = K div gs
  bpr = K div QK
  shape = "M = " & $M & ", K = " & $K
  flops = 2.0 * M.float * K.float
  mid = M div 2
  threadCounts = [2, 4, 6, 8]

var x = newSeq[float32](K)
var y = newSeq[float32](M)
for i in 0 ..< K:
  x[i] = float32(i mod 7) * 0.1'f32 - 0.3'f32

header "multi-core fused GEMV  (" & $countProcessors() & " logical cores)"

block: # ---- int8 ----
  var wq = newSeq[I8](M * K)
  var scales = newSeq[float32](M * gpr)
  for i in 0 ..< wq.len:
    wq[i] = toI8(float32((i mod 15) - 7))
  for i in 0 ..< scales.len:
    scales[i] = 0.02'f32

  var g = flopGroup("int8 GEMV, row-sharded", shape, flops, tol = 0.0)
  g.measure "serial", y[mid]:
    dequantGemv(wq, scales, gs, x, y)
  for nt in threadCounts:
    g.measure $nt & " threads", y[mid]:
      parDequantGemv(wq, scales, gs, x, y, nt)
  g.report()

block: # ---- ggml Q4_0, the shape real quantized weights ship in ----
  var w = newSeq[BlockQ4_0](M * bpr)
  for b in 0 ..< w.len:
    w[b].d = toF16(0.0125'f32)
    for i in 0 ..< QK div 2:
      w[b].qs[i] = uint8((b * 7 + i * 11) and 0xff)

  var g = flopGroup("ggml Q4_0 GEMV, row-sharded", shape, flops, tol = 0.0)
  g.measure "serial", y[mid]:
    dequantGemvQ4_0(w, bpr, x, y)
  for nt in threadCounts:
    g.measure $nt & " threads", y[mid]:
      parDequantGemvQ4_0(w, bpr, x, y, nt)
  g.report()
