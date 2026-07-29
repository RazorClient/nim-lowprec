## Batch conversions: scalar element-at-a-time vs the vectorized kernel.
##
## Every one of these is elementwise and bit-exact, so the checksums must match
## the scalar reference exactly — a mismatch fails the run. Pure memory streaming
## (read narrow, write wide, or the reverse), so at this N they measure how much
## of the load/store bandwidth each path keeps busy, not ALU throughput.
##
## The scalar references are `proc`s over `openArray` params, deliberately shaped
## like the kernels' own non-SIMD fallback: a loop over module-level `seq`s can't
## be auto-vectorized by the C compiler (the globals may alias), which would hand
## the SIMD row a few x of free speedup that has nothing to do with intrinsics.

import nim_lowprec/[formats/bfloat16, formats/float16, formats/float8, simd/convert]
import ./harness

const N = 16_000_000
const mid = N div 2

var f = newSeq[float32](N)
for i in 0 ..< N:
  f[i] = float32(i and 0xffff) * 0.01'f32 - 300.0'f32
var wide = newSeq[float32](N)

template defNarrowRef(name, T, encode: untyped) =
  ## A direct call, not a `proc` value: an indirect call per element would skew
  ## the comparison as badly in the other direction.
  proc name(src: openArray[float32], dst: var openArray[T]) =
    for i in 0 ..< src.len:
      dst[i] = encode(src[i])

defNarrowRef(bf16Ref, BF16, toBF16)
defNarrowRef(f16Ref, F16, toF16)

proc widenRef[T](src: openArray[T], dst: var openArray[float32]) =
  for i in 0 ..< src.len:
    dst[i] = toFloat32(src[i])

header "batch conversions"

block: # ---- bf16: shift-and-round, no hardware instruction ----
  var b = newSeq[BF16](N)
  var g = elemGroup("f32 -> bf16", N)
  g.scalarRow bits(b[mid]).float:
    bf16Ref(f, b)
  g.kernelRow bits(b[mid]).float:
    toBF16Batch(f, b)
  g.report()

  var h = elemGroup("bf16 -> f32", N)
  h.scalarRow wide[mid]:
    widenRef(b, wide)
  h.kernelRow wide[mid]:
    toFloat32Batch(b, wide)
  h.report()

block: # ---- fp16: hardware vcvt (NEON) / F16C (AVX2) ----
  var half = newSeq[F16](N)
  var g = elemGroup("f32 -> fp16", N)
  g.scalarRow bits(half[mid]).float:
    f16Ref(f, half)
  g.kernelRow bits(half[mid]).float:
    toF16Batch(f, half)
  g.report()

  var h = elemGroup("fp16 -> f32", N)
  h.scalarRow wide[mid]:
    widenRef(half, wide)
  h.kernelRow wide[mid]:
    toFloat32Batch(half, wide)
  h.report()

# ---- fp8 decode: one byte in, one fp32 out, three formats ----
# These speedups are the largest here for a reason that is only half about SIMD:
# the scalar `decode8` computes the magnitude in float64 (shared with the fp6/fp4
# tinyfloat core), while the kernel repacks the byte into an fp16 field and lets
# the hardware convert. Read them as "vectorized decode vs the scalar decode a
# caller actually gets today", not as a pure lane-width win.
template f8Bench(TT: untyped, nm: string) =
  block:
    var src = newSeq[TT](N)
    for i in 0 ..< N:
      # Codes cycle 0..0xfe. 0xff is NaN in e5m2/e4m3fnuz, and a NaN checksum
      # compares unequal to itself — NaN handling is the conformance tests' job.
      src[i] = TT(uint8(i mod 0xff))
    var g = elemGroup(nm & " -> f32", N)
    g.scalarRow wide[mid]:
      widenRef(src, wide)
    g.kernelRow wide[mid]:
      toFloat32Batch(src, wide)
    g.report()

f8Bench(F8E5M2, "fp8 e5m2")
f8Bench(F8E4M3, "fp8 e4m3")
f8Bench(F8E4M3FNUZ, "fp8 e4m3fnuz")
f8Bench(F8E5M2FNUZ, "fp8 e5m2fnuz")

# ---- fp8 ENCODE: round-to-odd f16 + 64 KB LUT vs the scalar encoder ----
template f8EncBench(TT, toFn, batchFn: untyped, nm: string) =
  block:
    var dst8 = newSeq[TT](N)
    var g = elemGroup("f32 -> " & nm, N)
    g.scalarRow float(uint8(dst8[mid])):
      for i in 0 ..< N:
        dst8[i] = toFn(f[i])
    g.kernelRow float(uint8(dst8[mid])):
      batchFn(f, dst8)
    g.report()

f8EncBench(F8E4M3, toF8E4M3, toF8E4M3Batch, "fp8 e4m3 (encode)")
f8EncBench(F8E5M2, toF8E5M2, toF8E5M2Batch, "fp8 e5m2 (encode)")
