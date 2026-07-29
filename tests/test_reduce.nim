## Tests for the generic BLAS-1 reductions — and proof the LowPrec concept
## activates: ONE set of generics runs over bf16, fp16, and int8.

import std/unittest
import nim_lowprec/[bfloat16, float16, intx, reduce]

suite "reduce (generic over LowPrec)":

  test "sum / dot / absmax / nrm2 on bf16":
    check sum([toBF16(1.0'f32), toBF16(2.0'f32), toBF16(3.0'f32)]) == 6.0'f32
    check absmax([toBF16(1.0'f32), toBF16(-5.0'f32), toBF16(3.0'f32)]) == 5.0'f32
    check nrm2([toBF16(3.0'f32), toBF16(4.0'f32)]) == 5.0'f32
    check dot([toBF16(1.0'f32), toBF16(2.0'f32)],
              [toBF16(3.0'f32), toBF16(4.0'f32)]) == 11.0'f32

  test "the SAME generics run on fp16 and int8":
    check sum([toF16(1.5'f32), toF16(2.5'f32)]) == 4.0'f32
    check sum([toI8(10.0'f32), toI8(20.0'f32), toI8(-5.0'f32)]) == 25.0'f32
    check dot([toI8(2.0'f32), toI8(3.0'f32)],
              [toI8(4.0'f32), toI8(5.0'f32)]) == 23.0'f32

  test "axpy / scal write back through fp32 (bf16)":
    var y = @[toBF16(1.0'f32), toBF16(1.0'f32)]
    axpy(2.0'f32, [toBF16(3.0'f32), toBF16(4.0'f32)], y)   # y = 2·x + y = [7, 9]
    check y[0].toFloat32 == 7.0'f32
    check y[1].toFloat32 == 9.0'f32
    var z = @[toBF16(2.0'f32), toBF16(3.0'f32)]
    scal(10.0'f32, z)                                       # z = 10·z = [20, 30]
    check z[0].toFloat32 == 20.0'f32
    check z[1].toFloat32 == 30.0'f32
