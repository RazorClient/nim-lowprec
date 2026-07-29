## Multi-column GEMM scaling: total GFLOP/s at n columns vs n separate GEMVs.
##
## Single-core these are expected to TIE (one core's SDOT rate can't outrun DRAM
## even re-streaming the weights n times). The reuse pays on all cores, where
## re-streaming would need n×95 GB/s against a ~115 GB/s chip ceiling: 8-thread
## int8 GEMM measured 564 GFLOP/s at n=8 vs ~190 for the GEMV path (M1 Pro).
## Every GEMM column is bit-identical to the corresponding GEMV (pinned in
## tests/test_sdot.nim), so no checksum machinery is needed here.
import std/[times, monotimes, strutils]
import nim_lowprec/[formats/intx, simd/dequant]

const M = 4096
const K = 14336
const gs = 128
const gpr = K div gs

template bench(label: string, ops: float, body: untyped) =
  block:
    body
    var best = Inf
    for _ in 1 .. 4:
      let t0 = getMonoTime()
      body
      let s = float((getMonoTime() - t0).inNanoseconds) / 1e9
      if s < best:
        best = s
    echo "   ",
      alignLeft(label, 34), formatFloat(ops / best / 1e9, ffDecimal, 1), " GFLOP/s"

var wq = newSeq[I8](M * K)
var wp = newSeq[byte](M * (K div 2))
var wScales = newSeq[float32](M * gpr)
for i in 0 ..< wq.len:
  wq[i] = I8(int8((i mod 200) - 100))
for i in 0 ..< wp.len:
  wp[i] = uint8((i * 13 + 7) and 0xff)
for i in 0 ..< wScales.len:
  wScales[i] = 0.02'f32

for n in [1, 2, 4, 8]:
  var xq = newSeq[I8](n * K)
  var xScales = newSeq[float32](n * gpr)
  for i in 0 ..< xq.len:
    xq[i] = I8(int8((i mod 150) - 75))
  for i in 0 ..< xScales.len:
    xScales[i] = 0.03'f32
  var y = newSeq[float32](M * n)
  let ops = 2.0 * M.float * K.float * n.float
  echo "-- n = ", n, " --"
  bench("int8 GEMM", ops):
    dequantGemmQ8(wq, wScales, gs, xq, xScales, n, y)
  bench("int8 n x GEMV (no reuse)", ops):
    for c in 0 ..< n:
      dequantGemvQ8(
        wq,
        wScales,
        gs,
        xq.toOpenArray(c * K, (c + 1) * K - 1),
        xScales.toOpenArray(c * gpr, (c + 1) * gpr - 1),
        y.toOpenArray(c * M, (c + 1) * M - 1),
      )
  bench("int4 GEMM", ops):
    dequantGemmI4Q8(wp, wScales, gs, xq, xScales, n, y)
