## int8-activation GEMV (SDOT) vs the fp32-activation GEMV, and what each step buys.
##
## Rows, in order, are the gap to llama.cpp taken apart:
##   fp32 activations       the existing kernel: widen int8 -> fp32, FP FMA
##   int8 acts (SDOT)       `vdotq_s32`, 16 MACs per instruction, int32 accumulate
##   int8 acts + 4-bit w    same kernel over PACKED int4 — half the weight bytes
##   ... + N threads        row-sharded via nim_lowprec/kernels/parallel
##
## The int8-activation rows include the cost of quantizing `x` on every call, since
## a real caller pays that per matvec — it is O(K) against the GEMV's O(M·K), and
## the numbers show it disappearing into the noise.
##
## `tol = 0` on the SDOT groups: those kernels accumulate each group in int32, which
## is exact, so every threaded row must match the serial row bit-for-bit. The fp32
## row is a different computation (it dequantizes rather than quantizing x), so it
## is reported in its own group.

import std/cpuinfo
import nim_lowprec/[formats/intx, quantization/quant, simd/dequant, simd/target, kernels/parallel]
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
  threadCounts = [4, 8]

var wq = newSeq[I8](M * K)
var wScales = newSeq[float32](M * gpr)
var xf = newSeq[float32](K)
var y = newSeq[float32](M)
for i in 0 ..< wq.len:
  wq[i] = toI8(float32((i mod 15) - 7))
for i in 0 ..< wScales.len:
  wScales[i] = 0.02'f32
for i in 0 ..< K:
  xf[i] = float32(i mod 7) * 0.1'f32 - 0.3'f32

# int4 weights holding the same values, packed two per byte (low nibble = even col).
var w4 = newSeq[byte](M * rowBytes)
for r in 0 ..< M:
  for c in 0 ..< K:
    let n = nibble(toI4(float32((c mod 15) - 7)))
    let bi = r * rowBytes + (c shr 1)
    if (c and 1) == 0:
      w4[bi] = n
    else:
      w4[bi] = w4[bi] or (n shl 4)

let xp = calibrateSymmetric(xf, gs, 127.0'f32)
var xq = newSeq[I8](K)
quantize(xf, xp, xq)

header "int8-activation GEMV  (" & $countProcessors() & " cores, dotprod = " &
  (if lpUseDotProd: "on" else: "off") & ")"

block: # ---- the existing fp32-activation kernel, for reference ----
  var g = flopGroup("int8 weights, fp32 activations", shape, flops)
  g.measure "fp32 acts, serial", y[mid]:
    dequantGemv(wq, wScales, gs, xf, y)
  g.report()

block: # ---- SDOT over int8 weights ----
  var g = flopGroup("int8 weights, int8 activations (SDOT)", shape, flops, tol = 0.0)
  g.measure "serial", y[mid]:
    quantize(xf, xp, xq)
    dequantGemvQ8(wq, wScales, gs, xq, xp.scale, y)
  for nt in threadCounts:
    g.measure $nt & " threads", y[mid]:
      quantize(xf, xp, xq)
      parDequantGemvQ8(wq, wScales, gs, xq, xp.scale, y, nt)
  g.report()

block: # ---- SDOT over PACKED int4 weights: half the bytes per weight ----
  var g =
    flopGroup("int4 weights (packed), int8 activations (SDOT)", shape, flops, tol = 0.0)
  g.measure "serial", y[mid]:
    quantize(xf, xp, xq)
    dequantGemvI4Q8(w4, wScales, gs, xq, xp.scale, y)
  for nt in threadCounts:
    g.measure $nt & " threads", y[mid]:
      quantize(xf, xp, xq)
      parDequantGemvI4Q8(w4, wScales, gs, xq, xp.scale, y, nt)
  # Same sharding, but the workers already exist: this is the per-call thread
  # creation removed, nothing else.
  for nt in threadCounts:
    var pool = initPool(nt)
    g.measure $nt & " thr, pooled", y[mid]:
      quantize(xf, xp, xq)
      pool.parDequantGemvI4Q8(w4, wScales, gs, xq, xp.scale, y)
    pool.shutdown()
  g.report()
