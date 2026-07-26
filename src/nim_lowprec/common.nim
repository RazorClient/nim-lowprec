import ./dtypes
export dtypes

func f32ToBits*(f: float32): uint32 {.inline.} = cast[uint32](f)
func bitsToF32*(u: uint32): float32 {.inline.} = cast[float32](u)
func f64ToBits*(f: float64): uint64 {.inline.} = cast[uint64](f)
func bitsToF64*(u: uint64): float64 {.inline.} = cast[float64](u)

type
  LowPrec* = concept x, type T
    decode(x) is float32
    storageBits(T) is int
