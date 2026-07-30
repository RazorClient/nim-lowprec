# nim-lowprec

Zero-dependency low-precision numeric primitives for ML inference in Nim.

The library provides storage dtypes, conversions, quantization helpers, packing
layouts, SIMD kernels, and small flat-buffer GEMV/GEMM utilities. It is not a
tensor framework.

## Install

```sh
nimble install https://github.com/RazorClient/nim-lowprec@#v1.0.0
```

## Use

```nim
import nim_lowprec

let x = toBF16(3.14159'f32)
echo x.toFloat32

let xs = @[toBF16(1.0'f32), toBF16(2.0'f32), toBF16(3.0'f32)]
echo dot(xs, xs)

let p = calibrateSymmetric(@[1.0'f32, -2.0'f32], groupSize = 2, qmax = 127.0'f32)
var q = newSeq[I8](2)
quantize(@[1.0'f32, -2.0'f32], p, q)
```

Public API:

```nim
import nim_lowprec
```

```

## Develop

nimble refs        # generate reference vectors
nimble test        # correctness tests
nimble simd        # SIMD tests
nimble example     # run examples/quickstart.nim
nimble bench       # native benchmark build
nimble benchScalar # scalar fallback benchmark build
nimble benchAll    # both benchmark modes
```

## License

MIT. See [LICENSE](LICENSE).
