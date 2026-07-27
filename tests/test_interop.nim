## Interop tests: DType <-> ml_dtypes/numpy, DLPack, safetensors.
##
## Round-trips must be EXACT wherever a mapping is defined:
##   fromX(toX(d)) == d  for every d that toX actually maps (non-"" result).
## We also check a few fromX() calls on real external strings, that the
## documented gaps really return "", and that unmapped/unknown strings raise
## (rather than silently returning a wrong DType).

import std/unittest
import nim_lowprec/dtypes
import nim_lowprec/interop

suite "ml_dtypes / numpy names":

  test "exact spellings (spot check the strings ml_dtypes actually uses)":
    check toMlDtypesName(dtBF16)       == "bfloat16"
    check toMlDtypesName(dtF16)        == "float16"
    check toMlDtypesName(dtF8E4M3)     == "float8_e4m3fn"   # NOT "float8_e4m3"
    check toMlDtypesName(dtF8E5M2)     == "float8_e5m2"
    check toMlDtypesName(dtF8E4M3FNUZ) == "float8_e4m3fnuz"
    check toMlDtypesName(dtF8E5M2FNUZ) == "float8_e5m2fnuz"
    check toMlDtypesName(dtE8M0)       == "float8_e8m0fnu"
    check toMlDtypesName(dtF6E2M3)     == "float6_e2m3fn"
    check toMlDtypesName(dtF6E3M2)     == "float6_e3m2fn"
    check toMlDtypesName(dtF4E2M1)     == "float4_e2m1fn"
    check toMlDtypesName(dtI8)         == "int8"
    check toMlDtypesName(dtI4)         == "int4"

  test "documented gap: I1 has no ml_dtypes/numpy name":
    check toMlDtypesName(dtI1) == ""

  test "round-trip EXACT for every mapped dtype":
    for d in DType:
      let s = toMlDtypesName(d)
      if s.len > 0:
        check fromMlDtypesName(s) == d

  test "fromMlDtypesName on real external strings":
    check fromMlDtypesName("float8_e4m3fn") == dtF8E4M3
    check fromMlDtypesName("bfloat16")      == dtBF16
    check fromMlDtypesName("int8")          == dtI8

  test "unknown / deliberately-unimplemented names raise":
    expect ValueError: discard fromMlDtypesName("")            # empty
    expect ValueError: discard fromMlDtypesName("float32")     # not low-prec
    expect ValueError: discard fromMlDtypesName("float8_e3m4") # valid ml_dtypes, not implemented here
    expect ValueError: discard fromMlDtypesName("float8_e4m3") # plain variant != e4m3fn
    expect ValueError: discard fromMlDtypesName("uint4")       # unsigned, no DType

suite "DLPack triples":

  test "exact formats (code, bits, lanes)":
    check toDLPack(dtBF16) == (kDLBfloat, 16'u8, 1'u16)
    check toDLPack(dtF16)  == (kDLFloat,  16'u8, 1'u16)
    check toDLPack(dtI8)   == (kDLInt,     8'u8, 1'u16)
    check toDLPack(dtI4)   == (kDLInt,     4'u8, 1'u16)

  test "approximated float formats carry the correct bit width":
    check toDLPack(dtF8E4M3)     == (kDLFloat, 8'u8, 1'u16)
    check toDLPack(dtF8E5M2)     == (kDLFloat, 8'u8, 1'u16)
    check toDLPack(dtF8E4M3FNUZ) == (kDLFloat, 8'u8, 1'u16)
    check toDLPack(dtF8E5M2FNUZ) == (kDLFloat, 8'u8, 1'u16)
    check toDLPack(dtE8M0)       == (kDLFloat, 8'u8, 1'u16)
    check toDLPack(dtF6E2M3)     == (kDLFloat, 6'u8, 1'u16)
    check toDLPack(dtF6E3M2)     == (kDLFloat, 6'u8, 1'u16)
    check toDLPack(dtF4E2M1)     == (kDLFloat, 4'u8, 1'u16)

  test "the fp8 approximation is intentionally NON-injective (documents why no fromDLPack)":
    # All four fp8 formats + E8M0 collapse onto the same (kDLFloat, 8) triple,
    # so a triple cannot be inverted back to a unique DType.
    check toDLPack(dtF8E4M3) == toDLPack(dtF8E5M2)
    check toDLPack(dtF8E4M3) == toDLPack(dtF8E4M3FNUZ)
    check toDLPack(dtF8E4M3) == toDLPack(dtE8M0)

  test "bits() width agrees with the DLPack triple's bit width":
    for d in DType:
      check toDLPack(d).bits.int == bits(d)

  test "I1 approximated as (kDLInt, 1) storage width (values differ: ±1 vs {0,-1})":
    check toDLPack(dtI1) == (kDLInt, 1'u8, 1'u16)

suite "safetensors tags":

  test "exact tags safetensors defines":
    check toSafetensorsName(dtBF16)   == "BF16"
    check toSafetensorsName(dtF16)    == "F16"
    check toSafetensorsName(dtF8E4M3) == "F8_E4M3"
    check toSafetensorsName(dtF8E5M2) == "F8_E5M2"
    check toSafetensorsName(dtI8)     == "I8"

  test "documented gaps return \"\"":
    check toSafetensorsName(dtF8E4M3FNUZ) == ""
    check toSafetensorsName(dtF8E5M2FNUZ) == ""
    check toSafetensorsName(dtE8M0)       == ""
    check toSafetensorsName(dtF6E2M3)     == ""
    check toSafetensorsName(dtF6E3M2)     == ""
    check toSafetensorsName(dtF4E2M1)     == ""
    check toSafetensorsName(dtI4)         == ""
    check toSafetensorsName(dtI1)         == ""

  test "round-trip EXACT for every mapped dtype":
    for d in DType:
      let s = toSafetensorsName(d)
      if s.len > 0:
        check fromSafetensorsName(s) == d

  test "fromSafetensorsName on real external strings":
    check fromSafetensorsName("BF16")    == dtBF16
    check fromSafetensorsName("F8_E5M2") == dtF8E5M2

  test "valid safetensors tags with no low-prec DType raise":
    expect ValueError: discard fromSafetensorsName("")     # empty
    expect ValueError: discard fromSafetensorsName("F32")  # real tag, not low-prec
    expect ValueError: discard fromSafetensorsName("F64")
    expect ValueError: discard fromSafetensorsName("BOOL") # not a ±1 sign
    expect ValueError: discard fromSafetensorsName("U8")   # unsigned, no DType
