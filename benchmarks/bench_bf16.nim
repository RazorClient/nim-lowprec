import std/[times, monotimes]
import nim_lowprec/bfloat16

const N = 50_000_000

proc benchRoundTrip(): float32 =
  var acc = 0.0'f32
  var x = 1.0001'f32
  for i in 0 ..< N:
    acc += toBF16(x).toFloat32
    x *= 1.0000001'f32
  acc

let start = getMonoTime()
let acc = benchRoundTrip()
let elapsed = (getMonoTime() - start).inNanoseconds.float / 1e9

echo "round-trip conversions: ", N
echo "elapsed:                ", elapsed, " s"
echo "throughput:             ", (N.float / elapsed / 1e6), " M ops/s"
echo "checksum:               ", acc  # keep the optimizer honest
