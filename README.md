# nim-lowprec

Low-precision floating-point primitives for ML inference in Nim.

One home for the low-precision family (bf16 first; fp16, fp8 to follow), sharing
one conformance-test harness, one SIMD-dispatch story, and one version.

`BF16` holds the upper 16 bits of an IEEE-754 `float32` (1 sign, 8 exponent,
7 mantissa bits): full float32 dynamic range at half the width.

## Usage

```nim
import nim_lowprec            # the whole family
# import nim_lowprec/bfloat16 # or just bf16

let a = toBF16(3.14159'f32)
echo a                  # 3.140625
echo a.toFloat32        # widen back (lossless)
echo (toBF16(2.0) * toBF16(3.0))  # 6.0
```

## Develop

```sh
nimble test    # run the test suite
nimble bench    # run microbenchmarks (release mode)
```
