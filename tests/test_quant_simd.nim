## SIMD dequant must be bit-identical to scalar `dequantize` (elementwise).
## The fused GEMV is a reduction, so it's checked vs an fp64 reference within a
## tolerance, not bit-for-bit. Local only.

import std/[unittest, math]
import nim_lowprec/[intx, mxfloat, quant, ggml, float16, simd/dequant, simd/convert]

proc scalarDeq[T](q: openArray[T]; p: QParams): seq[float32] =
  result = newSeq[float32](q.len)
  dequantize(q, p, result)

proc bitEq(a, b: openArray[float32]): int =
  for i in 0 ..< a.len:
    if cast[uint32](a[i]) != cast[uint32](b[i]): inc result

# xorshift32 RNG in [-1, 1) — no closure, threads its own state
proc nextRnd(u: var uint32): float32 =
  u = u xor (u shl 13); u = u xor (u shr 17); u = u xor (u shl 5)
  float32(u shr 8) / float32(1'u32 shl 24) * 2.0'f32 - 1.0'f32

suite "SIMD dequant == scalar":

  test "we are exercising the intended SIMD path":
    when lpSimd == "auto":
      when defined(arm64) or defined(aarch64): check simdBackend == "neon"
      else:                                    check simdBackend == "scalar"
    else:
      check simdBackend == lpSimd

  test "int8, per-tensor":
    var x = newSeq[float32](1000)
    for i in 0 ..< 1000: x[i] = sin(float32(i) * 0.1'f32) * 5.0'f32
    let p = calibrateSymmetric(x, 0, 127.0'f32)
    var q = newSeq[I8](1000); quantize(x, p, q)
    var a = newSeq[float32](1000); dequantizeBatch(q, p, a)
    check bitEq(a, scalarDeq(q, p)) == 0

  test "int8, per-group (64, ragged length)":
    var x = newSeq[float32](1000)
    for i in 0 ..< 1000: x[i] = (float32(i) - 500.0'f32) * 0.02'f32
    let p = calibrateSymmetric(x, 64, 127.0'f32)
    var q = newSeq[I8](1000); quantize(x, p, q)
    var a = newSeq[float32](1000); dequantizeBatch(q, p, a)
    check bitEq(a, scalarDeq(q, p)) == 0

  test "int4, per-group (32)":
    var x = newSeq[float32](512)
    for i in 0 ..< 512: x[i] = cos(float32(i) * 0.05'f32) * 3.0'f32
    let p = calibrateSymmetric(x, 32, 7.0'f32)
    var q = newSeq[I4](512); quantize(x, p, q)
    var a = newSeq[float32](512); dequantizeBatch(q, p, a)
    check bitEq(a, scalarDeq(q, p)) == 0

proc gemvRef(wq: openArray[I8]; scales: openArray[float32]; groupSize, M, K: int;
             x: openArray[float32]): seq[float64] =
  result = newSeq[float64](M)
  let gpr = K div groupSize
  for row in 0 ..< M:
    var acc = 0.0'f64
    for kk in 0 ..< K:
      let g = kk div groupSize
      acc += float64(int8(wq[row * K + kk])) * float64(scales[row * gpr + g]) * float64(x[kk])
    result[row] = acc

suite "SIMD fused dequant-GEMV":

  test "matches fp64 reference within tolerance (group tail exercised)":
    const M = 37
    const K = 200
    const gs = 40                        # 5 groups/row; each = 16+16+8 (SIMD + scalar tail)
    const gpr = K div gs
    var wq = newSeq[I8](M * K)
    var scales = newSeq[float32](M * gpr)
    var x = newSeq[float32](K)
    var u = 2463534242'u32
    for i in 0 ..< wq.len: wq[i] = toI8(nextRnd(u) * 110.0'f32)
    for i in 0 ..< scales.len: scales[i] = 0.005'f32 + abs(nextRnd(u)) * 0.02'f32
    for i in 0 ..< K: x[i] = nextRnd(u) * 3.0'f32
    var y = newSeq[float32](M)
    dequantGemv(wq, scales, gs, x, y)
    let refy = gemvRef(wq, scales, gs, M, K, x)
    var maxRel = 0.0
    for row in 0 ..< M:
      maxRel = max(maxRel, abs(float64(y[row]) - refy[row]) / (abs(refy[row]) + 1e-4))
    check maxRel < 1e-3

suite "SIMD fused int4 GEMV (packed)":

  test "packed int4 GEMV matches fp64 reference within tolerance (tail exercised)":
    const M = 17
    const K = 192
    const gs = 48                        # 4 groups/row; each 48 = 32 SIMD + 16 scalar tail
    const gpr = K div gs
    const rowBytes = K div 2
    var vals = newSeq[int](M * K)        # logical -8..7, kept for the fp64 reference
    var wpacked = newSeq[byte](M * rowBytes)
    var scales = newSeq[float32](M * gpr)
    var x = newSeq[float32](K)
    var u = 88172645'u32
    for r in 0 ..< M:
      for c in 0 ..< K:
        let v = int(nextRnd(u) * 7.9'f32)          # -7..7
        vals[r * K + c] = v
        let nib = nibble(toI4(float32(v)))
        let bi = r * rowBytes + (c shr 1)
        if (c and 1) == 0: wpacked[bi] = nib
        else:              wpacked[bi] = wpacked[bi] or (nib shl 4)
    for i in 0 ..< scales.len: scales[i] = 0.01'f32 + abs(nextRnd(u)) * 0.03'f32
    for i in 0 ..< K: x[i] = nextRnd(u) * 2.0'f32
    var y = newSeq[float32](M)
    dequantGemvI4(wpacked, scales, gs, x, y)
    var maxRel = 0.0
    for r in 0 ..< M:
      var acc = 0.0'f64
      for c in 0 ..< K:
        acc += float64(vals[r * K + c]) * float64(scales[r * gpr + c div gs]) * float64(x[c])
      maxRel = max(maxRel, abs(float64(y[r]) - acc) / (abs(acc) + 1e-4))
    check maxRel < 1e-3

suite "SIMD MXFP4 fused GEMV (packed, E8M0 block scale)":

  test "packed MXFP4 GEMV matches fp64 reference within tolerance (tail exercised)":
    const M = 17
    const K = 192
    const gs = 48                        # 4 blocks/row; each 48 = 32 SIMD + 16 scalar tail
    const gpr = K div gs
    const rowBytes = K div 2
    var wpacked = newSeq[byte](M * rowBytes)
    var scales = newSeq[float32](M * gpr)
    var x = newSeq[float32](K)
    var refv = newSeq[float64](M)
    var u = 20260728'u32
    for i in 0 ..< K: x[i] = nextRnd(u) * 2.0'f32
    for r in 0 ..< M:
      var rowvals = newSeq[float32](K)
      for c in 0 ..< K: rowvals[c] = nextRnd(u) * 4.5'f32       # spread across the fp4 range
      let p = calibrateMX(rowvals, gs, 2)                       # MXFP4 emax = 2
      var qf4 = newSeq[F4E2M1](K)
      quantize(rowvals, p, qf4)
      for g in 0 ..< gpr: scales[r * gpr + g] = p.scale[g]
      var packed = newSeq[byte](rowBytes)
      packF4(qf4, packed)
      for bi in 0 ..< rowBytes: wpacked[r * rowBytes + bi] = packed[bi]
      var acc = 0.0'f64
      for c in 0 ..< K:
        acc += float64(qf4[c].toFloat32) * float64(p.scale[c div gs]) * float64(x[c])
      refv[r] = acc
    var y = newSeq[float32](M)
    dequantGemvF4(wpacked, scales, gs, x, y)
    var maxRel = 0.0
    for r in 0 ..< M:
      maxRel = max(maxRel, abs(float64(y[r]) - refv[r]) / (abs(refv[r]) + 1e-4))
    check maxRel < 1e-3

suite "SIMD ggml Q8_0 / Q4_0 fused GEMV":

  test "Q8_0 GEMV matches fp64 reference":
    const M = 11
    const BPR = 3
    const K = BPR * QK
    var wq = newSeq[BlockQ8_0](M * BPR)
    var x = newSeq[float32](K)
    var refv = newSeq[float64](M)
    var u = 55555'u32
    for i in 0 ..< K: x[i] = nextRnd(u) * 2.0'f32
    for row in 0 ..< M:
      var rowf = newSeq[float32](K)
      for c in 0 ..< K: rowf[c] = nextRnd(u) * 4.0'f32
      var blocks = newSeq[BlockQ8_0](BPR)
      quantizeQ8_0(rowf, blocks)
      for bi in 0 ..< BPR: wq[row * BPR + bi] = blocks[bi]
      var deq = newSeq[float32](K)
      dequantizeQ8_0(blocks, deq)
      var acc = 0.0'f64
      for c in 0 ..< K: acc += float64(deq[c]) * float64(x[c])
      refv[row] = acc
    var y = newSeq[float32](M)
    dequantGemvQ8_0(wq, BPR, x, y)
    var maxRel = 0.0
    for row in 0 ..< M:
      maxRel = max(maxRel, abs(float64(y[row]) - refv[row]) / (abs(refv[row]) + 1e-4))
    check maxRel < 1e-3

  test "Q4_0 GEMV matches fp64 reference":
    const M = 9
    const BPR = 4
    const K = BPR * QK
    var wq = newSeq[BlockQ4_0](M * BPR)
    var x = newSeq[float32](K)
    var refv = newSeq[float64](M)
    var u = 24680'u32
    for i in 0 ..< K: x[i] = nextRnd(u) * 2.0'f32
    for row in 0 ..< M:
      var rowf = newSeq[float32](K)
      for c in 0 ..< K: rowf[c] = nextRnd(u) * 3.0'f32
      var blocks = newSeq[BlockQ4_0](BPR)
      quantizeQ4_0(rowf, blocks)
      for bi in 0 ..< BPR: wq[row * BPR + bi] = blocks[bi]
      var deq = newSeq[float32](K)
      dequantizeQ4_0(blocks, deq)
      var acc = 0.0'f64
      for c in 0 ..< K: acc += float64(deq[c]) * float64(x[c])
      refv[row] = acc
    var y = newSeq[float32](M)
    dequantGemvQ4_0(wq, BPR, x, y)
    var maxRel = 0.0
    for row in 0 ..< M:
      maxRel = max(maxRel, abs(float64(y[row]) - refv[row]) / (abs(refv[row]) + 1e-4))
    check maxRel < 1e-3


suite "ggml block dequant SIMD == scalar":
  # Elementwise (value × block scale), so bit-exact — same discipline as
  # `dequantizeBatch`. Blocks carry varied d (positive, negative, tiny) and the
  # full int8 / nibble code range.
  test "Q8_0 batch dequant bit-identical":
    const NB = 129                              # odd count: no accidental alignment
    var blocks = newSeq[BlockQ8_0](NB)
    for b in 0 ..< NB:
      blocks[b].d = toF16(float32(b - 64) * 0.0117'f32 + 0.001'f32)
      for i in 0 ..< QK: blocks[b].qs[i] = int8(((b * 31 + i * 7) mod 255) - 127)
    var want = newSeq[float32](NB * QK)
    var got = newSeq[float32](NB * QK)
    dequantizeQ8_0(blocks, want)
    dequantizeQ8_0Batch(blocks, got)
    var mism = 0
    for i in 0 ..< want.len:
      if cast[uint32](got[i]) != cast[uint32](want[i]): inc mism
    check mism == 0

  test "Q4_0 batch dequant bit-identical":
    const NB = 129
    var blocks = newSeq[BlockQ4_0](NB)
    for b in 0 ..< NB:
      blocks[b].d = toF16(float32(64 - b) * 0.0093'f32 + 0.002'f32)
      for i in 0 ..< QK div 2: blocks[b].qs[i] = uint8((b * 13 + i * 17) and 0xff)
    var want = newSeq[float32](NB * QK)
    var got = newSeq[float32](NB * QK)
    dequantizeQ4_0(blocks, want)
    dequantizeQ4_0Batch(blocks, got)
    var mism = 0
    for i in 0 ..< want.len:
      if cast[uint32](got[i]) != cast[uint32](want[i]): inc mism
    check mism == 0

suite "ggml block quantize SIMD == scalar":
  # The encode direction: same amax, same d, same multiply-by-inverse, same
  # ties-away rounding, same clamp — so bit-exact, including the packed d.
  test "quantizeQ8_0Batch bit-identical":
    const NB = 257
    var x = newSeq[float32](NB * QK)
    var u = 123456789'u32
    for i in 0 ..< x.len:
      u = u * 1664525'u32 + 1013904223'u32
      x[i] = (float32(u shr 8) / float32(1'u32 shl 24) - 0.5'f32) * 20.0'f32
    x[37] = 0.0'f32                                # a block with a zero
    for i in 5 * QK ..< 6 * QK: x[i] = 0.0'f32     # an all-zero block (d = 0 path)
    var a = newSeq[BlockQ8_0](NB)
    var b = newSeq[BlockQ8_0](NB)
    quantizeQ8_0(x, a)
    quantizeQ8_0Batch(x, b)
    var mism = 0
    for i in 0 ..< NB:
      if bits(a[i].d) != bits(b[i].d): inc mism
      for j in 0 ..< QK:
        if a[i].qs[j] != b[i].qs[j]: inc mism
    check mism == 0

  test "quantizeQ4_0Batch bit-identical":
    const NB = 257
    var x = newSeq[float32](NB * QK)
    var u = 987654321'u32
    for i in 0 ..< x.len:
      u = u * 1664525'u32 + 1013904223'u32
      x[i] = (float32(u shr 8) / float32(1'u32 shl 24) - 0.5'f32) * 16.0'f32
    for i in 9 * QK ..< 10 * QK: x[i] = 0.0'f32
    var a = newSeq[BlockQ4_0](NB)
    var b = newSeq[BlockQ4_0](NB)
    quantizeQ4_0(x, a)
    quantizeQ4_0Batch(x, b)
    var mism = 0
    for i in 0 ..< NB:
      if bits(a[i].d) != bits(b[i].d): inc mism
      for j in 0 ..< QK div 2:
        if a[i].qs[j] != b[i].qs[j]: inc mism
    check mism == 0

suite "per-group I8 quantize SIMD == scalar":
  test "quantizeBatch bit-identical (per-group, per-tensor, ragged tail)":
    for (n, gs, scheme) in [(4096, 128, qPerGroup), (4099, 64, qPerGroup), (1000, 0, qPerTensor)]:
      var x = newSeq[float32](n)
      var u = 55555'u32
      for i in 0 ..< n:
        u = u * 1664525'u32 + 1013904223'u32
        x[i] = (float32(u shr 8) / float32(1'u32 shl 24) - 0.5'f32) * 30.0'f32
      let p = (if scheme == qPerTensor: calibrateSymmetric(x, n, 127.0'f32)
               else: calibrateSymmetric(x, gs, 127.0'f32))
      var a = newSeq[I8](n)
      var b = newSeq[I8](n)
      quantize(x, p, a)
      quantizeBatch(x, p, b)
      var mism = 0
      for i in 0 ..< n:
        if int8(a[i]) != int8(b[i]): inc mism
      check mism == 0
