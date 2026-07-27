## bf16 dot product: the generic scalar `reduce.dot` vs the vectorized `dotBf16`.
##
## A reduction, so the fp summation order differs between the paths — checked
## within a relative tolerance, not bit-for-bit (`tol` on the group).
##
## Two lengths, deliberately: 4096 fits in L1/L2 and shows the per-element kernel
## cost, while 16M is far past last-level cache and is bandwidth-bound — two bf16
## streams at 2 bytes/element. The gap between the two speedups is the story: this
## kernel is 4 FLOP per 4 bytes loaded, so it stops being ALU-limited very quickly.

import nim_lowprec/[bfloat16, reduce, simd/blas1]
import ./harness

header "bf16 BLAS-1"

proc fill(n: int): (seq[BF16], seq[BF16]) =
  var a = newSeq[BF16](n)
  var b = newSeq[BF16](n)
  var u = 12345'u32
  for i in 0 ..< n:
    u = u * 1664525'u32 + 1013904223'u32
    a[i] = toBF16(float32(u shr 16) / 32768.0'f32 - 1.0'f32)
    u = u * 1664525'u32 + 1013904223'u32
    b[i] = toBF16(float32(u shr 16) / 32768.0'f32 - 1.0'f32)
  (a, b)

for n in [4_096, 16_000_000]:
  let (a, b) = fill(n)
  # One dot = n multiplies + n adds. Cache-resident lengths are repeated so the
  # timed region is long enough to measure; scale the flop count to match.
  let reps = if n <= 65_536: 10_000 else: 1
  var acc = 0.0'f32
  # tol 1e-3: at N=16M a sequential fp32 sum of signed terms drifts from the
  # 4-lane one by more than rounding-at-the-end would suggest. Bounding the
  # divergence is the tests' job (`tests/test_blas1_simd.nim`); here it only has
  # to prove both paths computed the same dot product.
  var g = flopGroup("bf16 dot  (x" & $reps & ")", "N = " & $n,
                    2.0 * n.float * reps.float, tol = 1e-3)
  g.scalarRow acc:
    acc = 0.0'f32
    for _ in 1 .. reps: acc += dot(a, b)
  g.kernelRow acc:
    acc = 0.0'f32
    for _ in 1 .. reps: acc += dotBf16(a, b)
  g.report()
