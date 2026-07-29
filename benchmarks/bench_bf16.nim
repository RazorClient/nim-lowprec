## bf16 round-trip LATENCY, not throughput: each iteration's input depends on the
## previous one, so the chain cannot be vectorized or software-pipelined away. This
## is the cost of `toBF16 → toFloat32` when it sits in the middle of dependent
## scalar work — the number that matters for element-at-a-time code.
##
## The batch/vectorized throughput comparison is `bench_simd`; there is nothing to
## compare against here, so this group reports one row.

import nim_lowprec/formats/bfloat16
import ./harness

const N = 20_000_000

header "bf16 round-trip latency"

var acc = 0.0'f32
var g = elemGroup("toBF16 → toFloat32  (dependent chain)", N)
g.measure "scalar", acc:
  acc = 0.0'f32
  var x = 1.0001'f32
  for i in 0 ..< N:
    acc += toBF16(x).toFloat32
    x *= 1.0000001'f32
g.report()
