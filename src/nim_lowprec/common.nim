import ./dtypes
export dtypes

func f32ToBits*(f: float32): uint32 {.inline.} =
  cast[uint32](f)
func bitsToF32*(u: uint32): float32 {.inline.} =
  cast[float32](u)
func f64ToBits*(f: float64): uint64 {.inline.} =
  cast[uint64](f)
func bitsToF64*(u: uint64): float64 {.inline.} =
  cast[float64](u)

type LowPrec* = concept x, type T
  decode(x) is float32
  storageBits(T) is int

template defFloatOps*(T, toFn, bitsN, DT: untyped) =
  mixin toFloat32
  func `$`*(x: T): string =
    $x.toFloat32
  func `==`*(a, b: T): bool {.inline.} =
    a.toFloat32 == b.toFloat32
  func `<`*(a, b: T): bool {.inline.} =
    a.toFloat32 < b.toFloat32
  func `<=`*(a, b: T): bool {.inline.} =
    a.toFloat32 <= b.toFloat32
  func `+`*(a, b: T): T {.inline.} =
    toFn(a.toFloat32 + b.toFloat32)
  func `-`*(a, b: T): T {.inline.} =
    toFn(a.toFloat32 - b.toFloat32)
  func `*`*(a, b: T): T {.inline.} =
    toFn(a.toFloat32 * b.toFloat32)
  func `/`*(a, b: T): T {.inline.} =
    toFn(a.toFloat32 / b.toFloat32)
  func decode*(x: T): float32 {.inline.} =
    x.toFloat32
  func encode*(f: float32, _: typedesc[T]): T {.inline.} =
    toFn(f)
  func storageBits*(_: typedesc[T]): int {.inline.} =
    bitsN
  func dtypeCode*(_: typedesc[T]): DType {.inline.} =
    DT
