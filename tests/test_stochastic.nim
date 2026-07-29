## Tests for toBF16Stochastic: (a) the result always brackets x, (b) the sample
## mean over many draws is ~unbiased, (c) NaN/Inf fall back to deterministic.
## The RNG state is threaded by the caller (var uint32, xorshift32); no globals.

import std/unittest
import nim_lowprec/bfloat16

# a separate xorshift32, used only to generate random fp32 *inputs* for (a)
proc nextRand(s: var uint32): uint32 =
  s = s xor (s shl 13)
  s = s xor (s shr 17)
  s = s xor (s shl 5)
  s

proc bracketCodes(x: float32): (uint16, uint16) =
  ## the two bf16 codes bracketing x: (lo = truncation, hi = lo or lo+1)
  let u = cast[uint32](x)
  let lo = uint16(u shr 16)
  let hi = (if (u and 0xffff'u32) == 0'u32: lo else: uint16((u shr 16) + 1'u32))
  (lo, hi)

suite "stochastic bf16 rounding":

  test "(a) result always lands on one of the two bracketing bf16 values":
    var gen = 0xC0FFEE11'u32          # PRNG for inputs
    var rng = 0x12345678'u32          # state under test
    var checked = 0
    var mm = 0
    for _ in 0 ..< 60000:
      let bits = nextRand(gen)
      let x = cast[float32](bits)
      if (bits and 0x7fff_ffff'u32) >= 0x7f80_0000'u32: continue   # skip NaN/Inf
      let (lo, hi) = bracketCodes(x)
      let r = toBF16Stochastic(x, rng)
      let code = r.bits
      if code != lo and code != hi: inc mm
      # the two bf16 values really do bracket x
      let loF = BF16(lo).toFloat32
      let hiF = BF16(hi).toFloat32
      if not (x >= min(loF, hiF) and x <= max(loF, hiF)): inc mm
      inc checked
    check mm == 0
    check checked > 40000            # made sure the loop actually exercised the path

  test "(a') representable values are returned exactly":
    var rng = 0xDEADBEEF'u32
    for v in [0.0'f32, 1.0, -1.0, 2.0, 0.5, -0.5, 256.0, -256.0, 3.0]:
      for _ in 0 ..< 100:
        check toBF16Stochastic(v, rng).toFloat32 == v

  test "(b) sample mean is unbiased (E[result] ~= x)":
    const n = 200_000
    let xs = [1.3'f32, 3.14159, 0.1, -2.7, 1234.5, 1e-4, 100000.0]
    var rng = 0x24681357'u32
    var worst = 0.0
    for x in xs:
      var sum = 0.0
      for _ in 0 ..< n:
        sum += toBF16Stochastic(x, rng).toFloat32.float64
      let mean = sum / n.float64
      let relErr = abs(mean - x.float64) / abs(x.float64)
      worst = max(worst, relErr)
      echo "  x=", x, "  mean=", mean, "  relErr=", relErr
      check relErr < 1e-3            # tight: theoretical SE over 200k draws is ~5e-6
    echo "  worst relErr over all x = ", worst

  test "(c) NaN in -> NaN out; Inf stays Inf":
    var rng = 0x9E3779B9'u32
    check toBF16Stochastic(NaN.float32, rng).isNaN
    check toBF16Stochastic(Inf.float32, rng).toFloat32 == Inf.float32
    check toBF16Stochastic((-Inf).float32, rng).toFloat32 == (-Inf).float32

  test "RNG state is threaded (advances, no globals)":
    var a = 0x01020304'u32
    let before = a
    discard toBF16Stochastic(1.3'f32, a)
    check a != before                # state was updated in place
    # same seed -> same sequence (pure function of the threaded state)
    var s1 = 0x55555555'u32
    var s2 = 0x55555555'u32
    for _ in 0 ..< 50:
      check toBF16Stochastic(2.7'f32, s1).bits == toBF16Stochastic(2.7'f32, s2).bits
