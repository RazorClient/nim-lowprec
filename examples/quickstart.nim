## nim-lowprec quickstart — run with:  nimble example   (or: nim c -r examples/quickstart.nim)
##
## Shows the three things the substrate is for: cheap narrow storage, generic
## fp32-accumulate BLAS-1, and quantized weights driven through a fused GEMV.

import std/math
import nim_lowprec

echo "== bf16: store narrow, compute wide =="
let a = toBF16(3.14159'f32)
echo "  bf16(pi) -> ", a.toFloat32, "   (16 bits of storage)"

echo "== BLAS-1, generic over any dtype (accumulates in fp32) =="
let xs = @[toBF16(1.0'f32), toBF16(2.0'f32), toBF16(3.0'f32)]
echo "  dot(xs, xs) = ", dot(xs, xs) # 1 + 4 + 9 = 14

echo "== MXFP4 microscaling: 4-bit weights + E8M0 block scale, fused GEMV =="
# One weight row of length K, quantized to MXFP4 in 32-element blocks.
const K = 64
var w = newSeq[float32](K)
for i in 0 ..< K:
  w[i] = sin(float32(i) * 0.3'f32) * 2.5'f32

let p = calibrateMX(w, blockSize = 32, elemEmax = 2) # elemEmax 2 = MXFP4 (e2m1)
var q = newSeq[F4E2M1](K)
quantize(w, p, q) # fp32 -> MXFP4 codes

var packed = newSeq[byte](K div 2) # 2 codes per byte
packF4(q, packed)

# activations, and the fused dequant-GEMV: y = (dequantized weights) . x
var x = newSeq[float32](K)
for i in 0 ..< K:
  x[i] = 0.5'f32
var y = newSeq[float32](1) # M = 1 output row
dequantGemvF4(packed, p.scale, 32, x, y)

# reference: decode each code and dot against x, in fp64
var refy = 0.0'f64
for i in 0 ..< K:
  refy += float64(q[i].toFloat32) * float64(p.scale[i div 32]) * float64(x[i])
echo "  fused MXFP4 GEMV y = ", y[0], "   (fp64 reference ", refy, ")"

echo "== int8-activation GEMV: SDOT (arm64) / maddubs (AVX2), exact int32 sums =="
# The fast path: weights AND activations as int8, one fp multiply per group.
const M = 4
const gs = 32
var wf = newSeq[float32](M * K)
for i in 0 ..< wf.len:
  wf[i] = float32((i mod 13) - 6) * 0.25'f32
let wp = calibrateSymmetric(wf, gs, 127.0'f32)
var wq = newSeq[I8](M * K)
quantizeBatch(wf, wp, wq) # SIMD encode, bit-exact vs scalar

var xf = newSeq[float32](K)
for i in 0 ..< K:
  xf[i] = float32(i mod 5) * 0.1'f32
let xp = calibrateSymmetric(xf, gs, 127.0'f32)
var xq = newSeq[I8](K)
quantizeBatch(xf, xp, xq)

var yQ = newSeq[float32](M)
dequantGemvQ8(wq, wp.scale, gs, xq, xp.scale, yQ)
echo "  y[0] = ", yQ[0], "   (int8 x int8, fp32 out)"

echo "== multi-column GEMM: weight stream read once, reused across n columns =="
const n = 2
var xqN = newSeq[I8](n * K) # column-major: column c at [c*K ..]
var xsN = newSeq[float32](n * (K div gs))
for c in 0 ..< n:
  for i in 0 ..< K:
    xqN[c * K + i] = xq[i]
  for g in 0 ..< K div gs:
    xsN[c * (K div gs) + g] = xp.scale[g]
var yN = newSeq[float32](M * n) # row-major: y[row*n + col]
dequantGemmQ8(wq, wp.scale, gs, xqN, xsN, n, yN)
echo "  column 0 of GEMM == GEMV bit-for-bit: ",
  cast[uint32](yN[0]) == cast[uint32](yQ[0])

when compileOption("threads"):
  echo "== caller-owned thread pool (row-sharded, bit-identical to serial) =="
  var pool = initPool(2)
  var yPool = newSeq[float32](M)
  pool.parDequantGemvQ8(wq, wp.scale, gs, xq, xp.scale, yPool)
  pool.shutdown()
  echo "  pooled == serial: ", cast[uint32](yPool[0]) == cast[uint32](yQ[0])
