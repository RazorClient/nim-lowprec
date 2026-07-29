## interop.nim — map this library's `DType` <-> external ecosystem identifiers.
##
## CORRECTNESS NOTE. A wrong dtype mapping silently corrupts real model
## weights, so nothing here is guessed. Every entry is the *exact* identifier
## the foreign ecosystem uses. Where no faithful equivalent exists we make the
## gap loud, not silent:
##   * string maps  -> return "" (an empty string is never a valid tag),
##   * reverse maps -> raise ValueError (we refuse to invent a DType),
##   * DLPack       -> approximate to the correct bit width and DOCUMENT it.
##
## Three ecosystems are covered:
##   * ml_dtypes / numpy  — the dtype *name* strings, i.e. `np.dtype(x).name`.
##   * DLPack             — the (type_code, bits, lanes) DLDataType triple.
##   * safetensors        — the on-disk dtype tag strings.
##
## dtypes.nim's `name(d)` already gives this library's OWN short names
## ("bf16", "f8e4m3", ...). This module deliberately does NOT duplicate it —
## these are the *foreign* spellings, which differ (e.g. our "f8e4m3" is
## ml_dtypes' "float8_e4m3fn" and safetensors' "F8_E4M3").

import ./dtypes

# ===========================================================================
# ml_dtypes / numpy
# ===========================================================================
# ml_dtypes (github.com/jax-ml/ml_dtypes) registers these low-precision types
# as numpy dtypes; the name below is exactly `np.dtype(<scalar>).name`.
# float16 and int8 are *native* numpy — they are not ml_dtypes extensions but
# their name strings are what a caller sees, so we map them too.

func toMlDtypesName*(d: DType): string =
  ## The exact ml_dtypes / numpy dtype name for `d`.
  ##
  ## Returns "" only for dtI1: ml_dtypes ships int2/int4/uint2/uint4 but NO
  ## 1-bit integer, and this library's I1 is a ±1 *sign* bit (not a
  ## two's-complement int1) anyway, so there is no faithful equivalent.
  case d
  of dtBF16:
    "bfloat16"
  of dtF16:
    "float16"
  # native numpy (not an ml_dtypes ext)
  of dtF8E4M3:
    "float8_e4m3fn"
  # OCP e4m3fn (finite-only, one NaN).
  # NOT plain "float8_e4m3" — that is a
  # different variant this lib lacks.
  of dtF8E5M2:
    "float8_e5m2"
  of dtF8E4M3FNUZ:
    "float8_e4m3fnuz"
  of dtF8E5M2FNUZ:
    "float8_e5m2fnuz"
  of dtE8M0:
    "float8_e8m0fnu"
  # MX shared-scale exponent code
  of dtF6E2M3:
    "float6_e2m3fn"
  of dtF6E3M2:
    "float6_e3m2fn"
  of dtF4E2M1:
    "float4_e2m1fn"
  of dtI8:
    "int8"
  # native numpy
  of dtI4:
    "int4"
  # ml_dtypes ext. NOTE: numpy *core* has
  # no int4; `int4` exists only because
  # ml_dtypes registers it. Its range is
  # two's-complement -8..7, matching dtI4.
  of dtI1:
    "" # GAP: no ml_dtypes/numpy 1-bit type

func fromMlDtypesName*(s: string): DType =
  ## Inverse of `toMlDtypesName`.
  ##
  ## Raises ValueError for any string with no DType here. That INCLUDES valid
  ## ml_dtypes names this library deliberately does not implement —
  ## float8_e3m4, plain float8_e4m3, float8_e4m3b11fnuz, int2/uint2, uint4 —
  ## because snapping them onto a near-neighbour would corrupt data.
  case s
  of "bfloat16":
    dtBF16
  of "float16":
    dtF16
  of "float8_e4m3fn":
    dtF8E4M3
  of "float8_e5m2":
    dtF8E5M2
  of "float8_e4m3fnuz":
    dtF8E4M3FNUZ
  of "float8_e5m2fnuz":
    dtF8E5M2FNUZ
  of "float8_e8m0fnu":
    dtE8M0
  of "float6_e2m3fn":
    dtF6E2M3
  of "float6_e3m2fn":
    dtF6E3M2
  of "float4_e2m1fn":
    dtF4E2M1
  of "int8":
    dtI8
  of "int4":
    dtI4
  else:
    raise
      newException(ValueError, "no nim-lowprec DType for ml_dtypes name: '" & s & "'")

# ===========================================================================
# DLPack
# ===========================================================================
# DLPack describes an element with a DLDataType { code: u8, bits: u8,
# lanes: u16 }. The ratified DLDataTypeCode values used here:
const
  kDLInt* = 0'u8 ## signed integer
  kDLUInt* = 1'u8 ## unsigned integer (unused here; no unsigned DType)
  kDLFloat* = 2'u8 ## IEEE-754-style float
  kDLBfloat* = 4'u8 ## bfloat

type DLPackType* = tuple[code: uint8, bits: uint8, lanes: uint16]

func toDLPack*(d: DType): DLPackType =
  ## Map `d` to a DLPack DLDataType triple. `lanes` is always 1 (one scalar
  ## element; DLPack `lanes` is for packed vector lanes, which we do not use).
  ##
  ## APPROXIMATION — READ THIS. The widely deployed DLDataTypeCode enum
  ## (kDLInt=0, kDLUInt=1, kDLFloat=2, kDLBfloat=4) has NO ratified code for
  ## fp8 (e4m3/e5m2 and the fnuz variants), fp6, fp4, or the E8M0 scale code.
  ## For those we emit `kDLFloat` with the correct *bit width* so buffer-size
  ## and stride arithmetic stay correct, but the (code, bits) pair is LOSSY:
  ##   - all four fp8 formats collapse onto (kDLFloat, 8),
  ##   - both fp6 formats collapse onto (kDLFloat, 6).
  ## A consumer therefore CANNOT recover the exact format from the triple
  ## alone; the format must travel out of band. This is exactly why there is
  ## intentionally NO `fromDLPack`: the map is not injective for the
  ## approximated formats, and a guessed inverse would silently corrupt
  ## weights. (Newer DLPack drafts propose dedicated fp8/fp6/fp4 codes; they
  ## are not universally ratified/supported, so we stay conservative.)
  ##
  ## Extra caveats:
  ##   * dtE8M0 is a *scale exponent* code, not an arithmetic float — its
  ##     (kDLFloat, 8) triple is a pure storage approximation.
  ##   * dtI1 is this library's ±1 sign bit. DLPack (kDLInt, 1) denotes a
  ##     1-bit two's-complement integer with values {0, -1}, so the encoded
  ##     *values* differ; we emit (kDLInt, 1) as a storage-width approximation.
  case d
  of dtBF16:
    (kDLBfloat, 16'u8, 1'u16)
  # exact
  of dtF16:
    (kDLFloat, 16'u8, 1'u16)
  # exact
  of dtF8E4M3, dtF8E5M2, dtF8E4M3FNUZ, dtF8E5M2FNUZ, dtE8M0:
    (kDLFloat, 8'u8, 1'u16) # APPROX — see note (lossy)
  of dtF6E2M3, dtF6E3M2:
    (kDLFloat, 6'u8, 1'u16) # APPROX — see note (lossy)
  of dtF4E2M1:
    (kDLFloat, 4'u8, 1'u16)
  # APPROX — see note
  of dtI8:
    (kDLInt, 8'u8, 1'u16)
  # exact
  of dtI4:
    (kDLInt, 4'u8, 1'u16)
  # exact (DLPack allows sub-byte int bits)
  of dtI1:
    (kDLInt, 1'u8, 1'u16) # APPROX — ±1 sign vs int1 {0,-1}

# ===========================================================================
# safetensors
# ===========================================================================
# safetensors' Dtype enum (huggingface/safetensors) is a fixed set:
#   BOOL U8 I8 F8_E4M3 F8_E5M2 I16 U16 F16 BF16 I32 U32 F32 F64 I64 U64
# Only a subset of this low-precision library's DTypes has a tag there.

func toSafetensorsName*(d: DType): string =
  ## The safetensors on-disk dtype tag for `d`, or "" where safetensors has
  ## no tag for the format. Documented gaps:
  ##   * fnuz fp8 (E4M3FNUZ / E5M2FNUZ) — safetensors defines only the OCP
  ##     F8_E4M3 and F8_E5M2; there are no AMD fnuz tags.
  ##   * E8M0, F6 (e2m3/e3m2), F4 (e2m1) — no MX sub-byte float tags exist.
  ##   * I4 — no 4-bit int tag (int4 is usually packed into U8 out of band).
  ##   * I1 — safetensors BOOL is an 8-bit {0,1} boolean, NOT a ±1 sign, so we
  ##     do not (mis)map I1 onto it.
  case d
  of dtBF16:
    "BF16"
  of dtF16:
    "F16"
  of dtF8E4M3:
    "F8_E4M3"
  # safetensors F8_E4M3 == OCP e4m3fn
  of dtF8E5M2:
    "F8_E5M2"
  of dtI8:
    "I8"
  of dtF8E4M3FNUZ:
    ""
  # GAP: no fnuz variant
  of dtF8E5M2FNUZ:
    ""
  # GAP: no fnuz variant
  of dtE8M0:
    ""
  # GAP: no MX scale tag
  of dtF6E2M3:
    ""
  # GAP: no F6 tag
  of dtF6E3M2:
    ""
  # GAP: no F6 tag
  of dtF4E2M1:
    ""
  # GAP: no F4 tag
  of dtI4:
    ""
  # GAP: no I4 tag (packed into U8 elsewhere)
  of dtI1:
    "" # GAP: BOOL is {0,1}, not ±1 sign

func fromSafetensorsName*(s: string): DType =
  ## Inverse of `toSafetensorsName`.
  ##
  ## Raises ValueError for any tag with no low-precision DType here. That
  ## includes perfectly valid safetensors tags for full/other-width types
  ## (BOOL, U8, U16, U32, U64, I16, I32, I64, F32, F64): they are real
  ## safetensors dtypes but this library is low-precision-only, so refusing is
  ## the correct, non-corrupting behaviour (no silent widening/narrowing).
  case s
  of "BF16":
    dtBF16
  of "F16":
    dtF16
  of "F8_E4M3":
    dtF8E4M3
  of "F8_E5M2":
    dtF8E5M2
  of "I8":
    dtI8
  else:
    raise newException(
      ValueError, "no nim-lowprec DType for safetensors dtype: '" & s & "'"
    )
