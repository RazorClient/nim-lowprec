# nim-lowprec

[![CI](https://github.com/RazorClient/nim-lowprec/actions/workflows/ci.yml/badge.svg)](https://github.com/RazorClient/nim-lowprec/actions/workflows/ci.yml)

Low-precision numeric primitives for ML inference in Nim — one home for the
low-precision family, sharing one conformance-test harness and one version.

**Storage formats**

| | type(s) | notes |
|---|---|---|
| bfloat16 | `BF16` | top 16 bits of an fp32 (1/8/7) |
| IEEE half | `F16` | binary16 (1/5/10), real subnormals |
| fp8 (OCP) | `F8E4M3`, `F8E5M2` | e4m3fn / e5m2 |
| fp8 (AMD) | `F8E4M3FNUZ`, `F8E5M2FNUZ` | fnuz: bias +1, no Inf/−0, `0x80` NaN |
| integers | `I8`, `I4`, `I1` | + nibble / bit packing (ggml low-nibble-first) |

Every float↔float32 conversion is verified **bit-exact** against `ml_dtypes` /
`numpy` — exhaustive where the code space allows (all 2¹⁶ for bf16/fp16, all 2⁸
for each fp8), differential over millions of samples otherwise.

On top of the storage types: a `LowPrec` concept, generic fp32-accumulate BLAS-1
(`sum` / `dot` / `nrm2` / `axpy` / …), and affine quantization (`QParams` +
`quantize` / `dequantize` + abs-max calibration).

## Usage

```nim
import nim_lowprec

let a = toBF16(3.14159'f32)
echo a                       # 3.140625
echo a.toFloat32             # widen back (lossless)

# BLAS-1 is generic over every dtype (accumulates in fp32)
echo dot([toBF16(1.0'f32), toBF16(2.0'f32)],
         [toBF16(3.0'f32), toBF16(4.0'f32)])          # 11.0

# quantize fp32 -> int8 with per-group scales, and back
let p = calibrateSymmetric(x, groupSize = 32, qmax = 127.0'f32)
var q = newSeq[I8](x.len)
quantize(x, p, q)
```

## Develop

```sh
nimble refs    # generate golden reference vectors (needs python3 + numpy + ml_dtypes)
nimble test    # run the conformance suite (differential legs skip without the vectors)
nimble bench   # microbenchmarks (release mode)
```

CI runs the full suite on **x86_64 (Linux)** and **arm64 (macOS)**, regenerating
the reference vectors per-arch so the differential tests enforce bit-exactness on
both architectures.
