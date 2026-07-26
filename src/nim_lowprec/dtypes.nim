type
  DType* = enum
    dtBF16          # bfloat16              (1/8/7)
    dtF16           # IEEE binary16         (1/5/10)
    dtF8E4M3        # OCP fp8 e4m3fn        (1/4/3, no Inf, bias 7)
    dtF8E5M2        # OCP fp8 e5m2          (1/5/2, bias 15)
    dtF8E4M3FNUZ    # AMD fp8 e4m3fnuz      (bias 8, 0x80=NaN, no Inf/-0)
    dtF8E5M2FNUZ    # AMD fp8 e5m2fnuz      (bias 16, 0x80=NaN, no Inf/-0)
    dtE8M0          # MX shared-scale exponent (8 exp bits, no sign/mantissa)
    dtF6E2M3        # MXFP6 e2m3
    dtF6E3M2        # MXFP6 e3m2
    dtF4E2M1        # MXFP4 e2m1
    dtI8            # int8
    dtI4            # int4 (packed 2 per byte)
    dtI1            # sign-mask bit

  DTypeKind* = enum
    dkFloat         # IEEE-like float (exponent + mantissa)
    dkInt           # integer
    dkScale         # scale-only exponent code (E8M0)

func bits*(d: DType): int {.inline.} =
  case d
  of dtBF16, dtF16: 16
  of dtF8E4M3, dtF8E5M2, dtF8E4M3FNUZ, dtF8E5M2FNUZ, dtE8M0, dtI8: 8
  of dtF6E2M3, dtF6E3M2: 6
  of dtF4E2M1, dtI4: 4
  of dtI1: 1

func kind*(d: DType): DTypeKind {.inline.} =
  case d
  of dtE8M0: dkScale
  of dtI8, dtI4, dtI1: dkInt
  else: dkFloat

func isSubByte*(d: DType): bool {.inline.} =
  d in {dtF6E2M3, dtF6E3M2, dtF4E2M1, dtI4, dtI1}

func hasInf*(d: DType): bool {.inline.} =
  # Whether the format encodes ±Inf. 
  # When false, overflow must SATURATE.
  d in {dtBF16, dtF16, dtF8E5M2}

func name*(d: DType): string {.inline.} =
  case d
  of dtBF16: "bf16"
  of dtF16: "f16"
  of dtF8E4M3: "f8e4m3"
  of dtF8E5M2: "f8e5m2"
  of dtF8E4M3FNUZ: "f8e4m3fnuz"
  of dtF8E5M2FNUZ: "f8e5m2fnuz"
  of dtE8M0: "e8m0"
  of dtF6E2M3: "f6e2m3"
  of dtF6E3M2: "f6e3m2"
  of dtF4E2M1: "f4e2m1"
  of dtI8: "i8"
  of dtI4: "i4"
  of dtI1: "i1"
