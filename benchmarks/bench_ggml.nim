## ggml block formats (Q8_0 / Q4_0) — the layout real quantized LLM weights ship
## in. Two questions, measured separately:
##
## 1. block dequant → fp32: scalar reference vs `dequantizeQ*Batch` (bit-exact).
## 2. GEMV: three rows, so the two wins are separable —
##      scalar ref          fused scalar loop (decode in the inner loop)
##      dequant + f32 dot   UNFUSED: expand a row to fp32, then dot it
##      <isa> kernel        `dequantGemvQ*`, decode in-register
##    The unfused row is the one a naive "dequantize then call BLAS" pipeline gets;
##    the gap to the kernel row is fusion + vectorization together.
##
## GEMV is a reduction, so checksums are compared within a relative tolerance. The
## reference loops take `openArray` params like the kernels do, so the C compiler
## has the same auto-vectorization opportunity on both sides.

import nim_lowprec/[formats/float16, quantization/ggml, simd/dequant, simd/target]
import ./harness

const
  M = 4096
  K = 4096
  bpr = K div QK ## blocks per row
  shape = "M = " & $M & ", K = " & $K & ", QK = " & $QK
  flops = 2.0 * M.float * K.float
  mid = M div 2

var x = newSeq[float32](K)
for i in 0 ..< K:
  x[i] = float32(i mod 7) * 0.1'f32 - 0.3'f32
var y = newSeq[float32](M)
var row = newSeq[float32](K) ## scratch for the unfused path

header "ggml Q8_0 / Q4_0"

# One row's worth of blocks, reused for every matrix row: the weights only have to
# be representative, and M×K of them at 4096² is already 17 MB (Q8_0).
var q8 = newSeq[BlockQ8_0](M * bpr)
for bi in 0 ..< q8.len:
  q8[bi].d = toF16(0.0125'f32)
  for i in 0 ..< QK:
    q8[bi].qs[i] = int8(((bi + i) mod 255) - 127)

var q4 = newSeq[BlockQ4_0](M * bpr)
for bi in 0 ..< q4.len:
  q4[bi].d = toF16(0.0125'f32)
  for i in 0 ..< QK div 2:
    q4[bi].qs[i] = uint8((bi * 7 + i * 11) and 0xff)

block: # ---- 1. block dequant → fp32 (scalar-only, no kernel yet) ----
  # One row of blocks at a time — the shape a row-major GEMV loop asks for.
  var dst = newSeq[float32](bpr * QK)
  var g = elemGroup("Q8_0 dequant → fp32", M * K)
  g.scalarRow dst[K div 2]:
    for r in 0 ..< M:
      dequantizeQ8_0(q8.toOpenArray(r * bpr, (r + 1) * bpr - 1), dst)
  g.kernelRow dst[K div 2]:
    for r in 0 ..< M:
      dequantizeQ8_0Batch(q8.toOpenArray(r * bpr, (r + 1) * bpr - 1), dst)
  g.report()

  var h = elemGroup("Q4_0 dequant → fp32", M * K)
  h.scalarRow dst[K div 2]:
    for r in 0 ..< M:
      dequantizeQ4_0(q4.toOpenArray(r * bpr, (r + 1) * bpr - 1), dst)
  h.kernelRow dst[K div 2]:
    for r in 0 ..< M:
      dequantizeQ4_0Batch(q4.toOpenArray(r * bpr, (r + 1) * bpr - 1), dst)
  h.report()

block: # ---- 2. Q8_0 GEMV ----
  proc scalarFused(
      q8: openArray[BlockQ8_0], x: openArray[float32], y: var openArray[float32]
  ) =
    for r in 0 ..< M:
      var total = 0.0'f32
      for bi in 0 ..< bpr:
        let blk = q8[r * bpr + bi]
        let base = bi * QK
        var p = 0.0'f32
        for i in 0 ..< QK:
          p += float32(blk.qs[i]) * x[base + i]
        total += p * blk.d.toFloat32
      y[r] = total

  proc unfused(
      q8: openArray[BlockQ8_0],
      x: openArray[float32],
      row: var openArray[float32],
      y: var openArray[float32],
  ) =
    for r in 0 ..< M:
      dequantizeQ8_0(q8.toOpenArray(r * bpr, (r + 1) * bpr - 1), row)
      var total = 0.0'f32
      for i in 0 ..< K:
        total += row[i] * x[i]
      y[r] = total

  var g = flopGroup("Q8_0 GEMV", shape, flops)
  g.scalarRow y[mid]:
    scalarFused(q8, x, y)
  g.measure "dequant + f32 dot", y[mid]:
    unfused(q8, x, row, y)
  g.kernelRow y[mid]:
    dequantGemvQ8_0(q8, bpr, x, y)
  g.report()

block: # ---- 3. Q4_0 GEMV (value = d·(nibble−8), ggml lane order) ----
  proc scalarFused(
      q4: openArray[BlockQ4_0], x: openArray[float32], y: var openArray[float32]
  ) =
    for r in 0 ..< M:
      var total = 0.0'f32
      for bi in 0 ..< bpr:
        let blk = q4[r * bpr + bi]
        let base = bi * QK
        var p = 0.0'f32
        for i in 0 ..< QK div 2:
          let b = blk.qs[i]
          p += float32(int(b and 0x0f'u8) - 8) * x[base + i]
          p += float32(int(b shr 4) - 8) * x[base + i + QK div 2]
        total += p * blk.d.toFloat32
      y[r] = total

  proc unfused(
      q4: openArray[BlockQ4_0],
      x: openArray[float32],
      row: var openArray[float32],
      y: var openArray[float32],
  ) =
    for r in 0 ..< M:
      dequantizeQ4_0(q4.toOpenArray(r * bpr, (r + 1) * bpr - 1), row)
      var total = 0.0'f32
      for i in 0 ..< K:
        total += row[i] * x[i]
      y[r] = total

  var g = flopGroup("Q4_0 GEMV", shape, flops)
  g.scalarRow y[mid]:
    scalarFused(q4, x, y)
  g.measure "dequant + f32 dot", y[mid]:
    unfused(q4, x, row, y)
  g.kernelRow y[mid]:
    dequantGemvQ4_0(q4, bpr, x, y)
  g.report()

block: # ---- 4. int8-activation (SDOT) forms over the same blocks ----
  # Format-identical to llama.cpp's vec_dot_q8_0_q8_0 / q4_0_q8_0: activations
  # quantized to Q8_0 blocks, scale inline per 32 weights. On this machine these
  # are SLOWER than the group-scale SDOT kernels in bench_sdot (the per-32-weight
  # finalize is 4x more frequent than per-128) — they exist so real GGUF weights
  # can run without repacking. Exactness is pinned in tests/test_sdot.nim.
  var xq = newSeq[BlockQ8_0](bpr)
  var g = flopGroup(
    "Q8_0 GEMV, q8 activations (SDOT, dotprod=" & (if lpUseDotProd: "on" else: "off") &
      ")",
    shape,
    flops,
    tol = 0.0,
  )
  g.measure "serial", y[mid]:
    quantizeQ8_0(x, xq)
    dequantGemvQ8_0Q8(q8, bpr, xq, y)
  g.report()

  var h = flopGroup("Q4_0 GEMV, q8 activations (SDOT)", shape, flops, tol = 0.0)
  h.measure "serial", y[mid]:
    quantizeQ8_0(x, xq)
    dequantGemvQ4_0Q8(q4, bpr, xq, y)
  h.report()
