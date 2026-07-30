# This files exists to serves as a codec between the libraries internal types and external 
# dtype names used by safetensors, ml_dtypes, NumPy, or DLPack.
# cases where we need to load say tensors from a model file etc 
# the question is whether to place this file here or does it belong to some other place

import ./dtypes

const
  kDLInt* = 0'u8
  kDLUInt* = 1'u8
  kDLFloat* = 2'u8
  kDLBfloat* = 4'u8

type
  DLPackType* = tuple[code: uint8, bits: uint8, lanes: uint16]

  DTypeInterop = object
    mlDtypes: string
    safetensors: string
    dlpack: DLPackType

const dtypeInterop: array[DType, DTypeInterop] = [
  dtBF16: DTypeInterop(
    mlDtypes: "bfloat16", safetensors: "BF16", dlpack: (kDLBfloat, 16'u8, 1'u16)
  ),
  dtF16: DTypeInterop(
    mlDtypes: "float16", safetensors: "F16", dlpack: (kDLFloat, 16'u8, 1'u16)
  ),
  dtF8E4M3: DTypeInterop(
    mlDtypes: "float8_e4m3fn", safetensors: "F8_E4M3", dlpack: (kDLFloat, 8'u8, 1'u16)
  ),
  dtF8E5M2: DTypeInterop(
    mlDtypes: "float8_e5m2", safetensors: "F8_E5M2", dlpack: (kDLFloat, 8'u8, 1'u16)
  ),
  dtF8E4M3FNUZ: DTypeInterop(
    mlDtypes: "float8_e4m3fnuz", safetensors: "", dlpack: (kDLFloat, 8'u8, 1'u16)
  ),
  dtF8E5M2FNUZ: DTypeInterop(
    mlDtypes: "float8_e5m2fnuz", safetensors: "", dlpack: (kDLFloat, 8'u8, 1'u16)
  ),
  dtE8M0: DTypeInterop(
    mlDtypes: "float8_e8m0fnu", safetensors: "", dlpack: (kDLFloat, 8'u8, 1'u16)
  ),
  dtF6E2M3: DTypeInterop(
    mlDtypes: "float6_e2m3fn", safetensors: "", dlpack: (kDLFloat, 6'u8, 1'u16)
  ),
  dtF6E3M2: DTypeInterop(
    mlDtypes: "float6_e3m2fn", safetensors: "", dlpack: (kDLFloat, 6'u8, 1'u16)
  ),
  dtF4E2M1: DTypeInterop(
    mlDtypes: "float4_e2m1fn", safetensors: "", dlpack: (kDLFloat, 4'u8, 1'u16)
  ),
  dtI8: DTypeInterop(mlDtypes: "int8", safetensors: "I8", dlpack: (kDLInt, 8'u8, 1'u16)),
  dtI4: DTypeInterop(mlDtypes: "int4", safetensors: "", dlpack: (kDLInt, 4'u8, 1'u16)),
  dtI1: DTypeInterop(mlDtypes: "", safetensors: "", dlpack: (kDLInt, 1'u8, 1'u16)),
]

func toMlDtypesName*(d: DType): string =
  dtypeInterop[d].mlDtypes

func fromMlDtypesName*(s: string): DType =
  for d in DType:
    if dtypeInterop[d].mlDtypes == s and s.len > 0:
      return d
  raise newException(ValueError, "no nim-lowprec DType for ml_dtypes name: '" & s & "'")

func toDLPack*(d: DType): DLPackType =
  dtypeInterop[d].dlpack

func toSafetensorsName*(d: DType): string =
  dtypeInterop[d].safetensors

func fromSafetensorsName*(s: string): DType =
  for d in DType:
    if dtypeInterop[d].safetensors == s and s.len > 0:
      return d
  raise
    newException(ValueError, "no nim-lowprec DType for safetensors dtype: '" & s & "'")
