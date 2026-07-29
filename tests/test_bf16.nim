import std/[unittest, math]
import nim_lowprec/bfloat16

suite "bf16 conversions":
  test "round trip of exactly representable values":
    for v in [0.0'f32, 1.0, -1.0, 2.0, 0.5, 256.0]:
      check toBF16(v).toFloat32 == v

  test "round-to-nearest-even keeps error below one ulp":
    for v in [3.14159'f32, 1.2345, -678.9, 1e-30, 1e30]:
      let approx = toBF16(v).toFloat32
      let relErr = abs(approx - v) / abs(v)
      check relErr < 0.004 # 2^-8, the bf16 mantissa bound

  test "special values survive":
    check toBF16(Inf.float32).toFloat32 == Inf.float32
    check toBF16(-Inf.float32).toFloat32 == -Inf.float32
    check toBF16(NaN.float32).toFloat32.classify == fcNan

  test "arithmetic operators":
    check (toBF16(2.0) * toBF16(3.0)) == toBF16(6.0)
    check (toBF16(1.0) + toBF16(1.0)) == toBF16(2.0)
