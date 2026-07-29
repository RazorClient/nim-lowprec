## int8-activation GEMV (`dequantGemvQ8` / `dequantGemvI4Q8`) against an EXACT
## integer reference.
##
## These kernels accumulate each group in int32, which is exact — so unlike the
## fp32-activation GEMVs (checked against fp64 within a tolerance) this one is
## pinned bit-for-bit. The reference sums in int64 and then applies the scales with
## the same two fp32 multiplies in the same order, so any difference is a real bug,
## not rounding.
##
## Runs in the ordinary suite: on Apple Silicon it exercises the SDOT path, and
## elsewhere the integer fallback, which must produce identical bits.

import std/unittest
import nim_lowprec/[intx, simd/dequant, simd/target]

# The references below must round exactly like the kernels: mul then add, two
# roundings. Apple clang's default -ffp-contract=fast would fuse them into an fma
# in THIS file too, so contraction is pinned off here just like in simd/dequant.
{.localPassc: "-ffp-contract=off".}

proc intRef(
    wq: openArray[I8],
    wScales: openArray[float32],
    groupSize: int,
    xq: openArray[I8],
    xScales: openArray[float32],
    y: var openArray[float32],
) =
  let m = y.len
  let k = xq.len
  let gpr = k div groupSize
  for row in 0 ..< m:
    var total = 0.0'f32
    for g in 0 ..< gpr:
      var isum = 0'i64 # int64: independent of the kernel's width
      for kk in g * groupSize ..< (g + 1) * groupSize:
        isum += int64(int8(wq[row * k + kk])) * int64(int8(xq[kk]))
      total += float32(int32(isum)) * wScales[row * gpr + g] * xScales[g]
    y[row] = total

proc packI4(vals: openArray[I4]): seq[byte] =
  result = newSeq[byte](vals.len div 2)
  for c in 0 ..< vals.len:
    let n = nibble(vals[c])
    if (c and 1) == 0:
      result[c shr 1] = n
    else:
      result[c shr 1] = result[c shr 1] or (n shl 4)

template checkBits(got, want: seq[float32]) =
  var mism = 0
  for i in 0 ..< want.len:
    if cast[uint32](got[i]) != cast[uint32](want[i]):
      inc mism
  check mism == 0

suite "int8-activation GEMV == exact integer reference":
  echo "  dot-product path: ",
    (if lpUseDotProd: "SDOT (FEAT_DotProd)" else: "integer fallback")

  # M = 7 and 4 exercise both the 4-row tile and its remainder; K = 96 with
  # groupSize 32 gives whole groups, K = 40 / group 20 leaves a group tail that is
  # not a multiple of 16 (int8) or 32 (packed int4).
  test "int8 weights, tiles + tile remainder + group tail":
    for (M, K, gs) in [(7, 96, 32), (4, 96, 96), (1, 96, 32), (9, 40, 20)]:
      var wq = newSeq[I8](M * K)
      var wScales = newSeq[float32](M * (K div gs))
      var xq = newSeq[I8](K)
      var xScales = newSeq[float32](K div gs)
      var u = 2463534242'u32
      template nextByte(): int =
        u = u xor (u shl 13)
        u = u xor (u shr 17)
        u = u xor (u shl 5)
        int(u and 0xff'u32) - 128

      for i in 0 ..< wq.len:
        wq[i] = I8(int8(max(nextByte(), -127)))
      for i in 0 ..< K:
        xq[i] = I8(int8(max(nextByte(), -127)))
      for i in 0 ..< wScales.len:
        wScales[i] = 0.011'f32 + float32(i mod 5) * 0.0013'f32
      for i in 0 ..< xScales.len:
        xScales[i] = 0.023'f32 + float32(i mod 3) * 0.0007'f32

      var want = newSeq[float32](M)
      intRef(wq, wScales, gs, xq, xScales, want)
      var got = newSeq[float32](M)
      dequantGemvQ8(wq, wScales, gs, xq, xScales, got)
      checkBits(got, want)

  test "packed int4 weights, tiles + tile remainder + group tail":
    for (M, K, gs) in [(7, 128, 64), (4, 128, 128), (1, 128, 64), (5, 96, 48)]:
      var vals = newSeq[I4](M * K)
      var wScales = newSeq[float32](M * (K div gs))
      var xq = newSeq[I8](K)
      var xScales = newSeq[float32](K div gs)
      var u = 88172645463325252'u64
      template nextNib(): int =
        u = u xor (u shl 13)
        u = u xor (u shr 7)
        u = u xor (u shl 17)
        int(u and 0x0f'u64) - 8

      for i in 0 ..< vals.len:
        vals[i] = toI4(float32(nextNib()))
      for i in 0 ..< K:
        u = u xor (u shl 13)
        u = u xor (u shr 7)
        u = u xor (u shl 17)
        xq[i] = I8(int8(max(int(u and 0xff'u64) - 128, -127))) # keep it inside int8
      for i in 0 ..< wScales.len:
        wScales[i] = 0.017'f32 + float32(i mod 4) * 0.0011'f32
      for i in 0 ..< xScales.len:
        xScales[i] = 0.031'f32 + float32(i mod 3) * 0.0005'f32

      # The reference takes the same weights unpacked, one per byte.
      var asI8 = newSeq[I8](vals.len)
      for i in 0 ..< vals.len:
        asI8[i] = I8(int8(vals[i]))
      var want = newSeq[float32](M)
      intRef(asI8, wScales, gs, xq, xScales, want)

      let packed = packI4(vals)
      var got = newSeq[float32](M)
      dequantGemvI4Q8(packed, wScales, gs, xq, xScales, got)
      checkBits(got, want)

  test "packed and unpacked kernels agree on the same weights":
    const M = 8
    const K = 128
    const gs = 64
    var vals = newSeq[I4](M * K)
    for i in 0 ..< vals.len:
      vals[i] = toI4(float32((i mod 15) - 7))
    var asI8 = newSeq[I8](vals.len)
    for i in 0 ..< vals.len:
      asI8[i] = I8(int8(vals[i]))
    var xq = newSeq[I8](K)
    for i in 0 ..< K:
      xq[i] = I8(int8((i mod 200) - 100))
    var wScales = newSeq[float32](M * (K div gs))
    var xScales = newSeq[float32](K div gs)
    for i in 0 ..< wScales.len:
      wScales[i] = 0.02'f32
    for i in 0 ..< xScales.len:
      xScales[i] = 0.03'f32

    var a = newSeq[float32](M)
    var b = newSeq[float32](M)
    dequantGemvQ8(asI8, wScales, gs, xq, xScales, a)
    dequantGemvI4Q8(packI4(vals), wScales, gs, xq, xScales, b)
    checkBits(b, a)

import nim_lowprec/[ggml, float16]

suite "ggml-layout SDOT GEMV == exact reference":
  ## Weights AND activations in ggml block format, scale inline per block.
  ## Per-block sums are exact int32; the fp accumulation order is the documented
  ## lane discipline (block index mod 4, then (l0+l1)+(l2+l3)) — the reference
  ## reproduces it, so the check is bit-for-bit. Runs the SDOT path on Apple
  ## Silicon and the integer fallback elsewhere; both must produce the same bits.

  proc refQ8(
      w: openArray[BlockQ8_0],
      bpr: int,
      xq: openArray[BlockQ8_0],
      y: var openArray[float32],
  ) =
    for row in 0 ..< y.len:
      var lanes: array[4, float32]
      for bi in 0 ..< bpr:
        var isum = 0'i64
        for i in 0 ..< QK:
          isum += int64(w[row * bpr + bi].qs[i]) * int64(xq[bi].qs[i])
        lanes[bi and 3] =
          lanes[bi and 3] +
          float32(int32(isum)) * (w[row * bpr + bi].d.toFloat32 * xq[bi].d.toFloat32)
      y[row] = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])

  proc refQ4(
      w: openArray[BlockQ4_0],
      bpr: int,
      xq: openArray[BlockQ8_0],
      y: var openArray[float32],
  ) =
    for row in 0 ..< y.len:
      var lanes: array[4, float32]
      for bi in 0 ..< bpr:
        var isum = 0'i64
        for i in 0 ..< QK div 2:
          let b = w[row * bpr + bi].qs[i]
          isum += (int64(b and 0x0f'u8) - 8) * int64(xq[bi].qs[i])
          isum += (int64(b shr 4) - 8) * int64(xq[bi].qs[i + QK div 2])
        lanes[bi and 3] =
          lanes[bi and 3] +
          float32(int32(isum)) * (w[row * bpr + bi].d.toFloat32 * xq[bi].d.toFloat32)
      y[row] = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])

  proc mkX(bpr: int): seq[BlockQ8_0] =
    result = newSeq[BlockQ8_0](bpr)
    for b in 0 ..< bpr:
      result[b].d = toF16(0.031'f32 + float32(b mod 5) * 0.0017'f32)
      for i in 0 ..< QK:
        result[b].qs[i] = int8(((b * 29 + i * 13) mod 255) - 127)

  test "Q8_0 x q8 activations (4-chunks + block tail)":
    for (M, bpr) in [(5, 12), (4, 7), (1, 4), (3, 3)]: # tails: bpr mod 4 = 0,3,0,3
      var w = newSeq[BlockQ8_0](M * bpr)
      for b in 0 ..< w.len:
        w[b].d = toF16(float32((b mod 9) - 4) * 0.011'f32 + 0.003'f32)
        for i in 0 ..< QK:
          w[b].qs[i] = int8(((b * 17 + i * 11) mod 255) - 127)
      let xq = mkX(bpr)
      var want = newSeq[float32](M)
      var got = newSeq[float32](M)
      refQ8(w, bpr, xq, want)
      dequantGemvQ8_0Q8(w, bpr, xq, got)
      var mism = 0
      for i in 0 ..< M:
        if cast[uint32](got[i]) != cast[uint32](want[i]):
          inc mism
      check mism == 0

  test "Q4_0 x q8 activations (4-chunks + block tail)":
    for (M, bpr) in [(5, 12), (4, 7), (1, 4), (3, 3)]:
      var w = newSeq[BlockQ4_0](M * bpr)
      for b in 0 ..< w.len:
        w[b].d = toF16(float32((b mod 7) - 3) * 0.013'f32 + 0.002'f32)
        for i in 0 ..< QK div 2:
          w[b].qs[i] = uint8((b * 23 + i * 19) and 0xff)
      let xq = mkX(bpr)
      var want = newSeq[float32](M)
      var got = newSeq[float32](M)
      refQ4(w, bpr, xq, want)
      dequantGemvQ4_0Q8(w, bpr, xq, got)
      var mism = 0
      for i in 0 ..< M:
        if cast[uint32](got[i]) != cast[uint32](want[i]):
          inc mism
      check mism == 0

suite "multi-column GEMM == per-column GEMV, bit-for-bit":
  ## Each GEMM output column must be exactly what the single-column kernel
  ## produces for that column: same 4-chain group sums (exact int32), same
  ## finalize expression, so the guarantee is bits, not tolerance. Covers n=1,
  ## even/odd n, and a group tail.
  test "int8 GEMM":
    for (M, K, gs, n) in [
      (9, 128, 64, 3), (4, 256, 128, 2), (5, 96, 48, 1), (6, 80, 40, 4)
    ]:
      let gpr = K div gs
      var wq = newSeq[I8](M * K)
      var xq = newSeq[I8](n * K)
      var wScales = newSeq[float32](M * gpr)
      var xScales = newSeq[float32](n * gpr)
      for i in 0 ..< wq.len:
        wq[i] = I8(int8((i * 37 mod 251) - 125))
      for i in 0 ..< xq.len:
        xq[i] = I8(int8((i * 41 mod 241) - 120))
      for i in 0 ..< wScales.len:
        wScales[i] = 0.012'f32 + float32(i mod 6) * 0.0009'f32
      for i in 0 ..< xScales.len:
        xScales[i] = 0.021'f32 + float32(i mod 4) * 0.0013'f32

      var yG = newSeq[float32](M * n)
      dequantGemmQ8(wq, wScales, gs, xq, xScales, n, yG)
      var mism = 0
      for c in 0 ..< n:
        var yc = newSeq[float32](M)
        dequantGemvQ8(
          wq,
          wScales,
          gs,
          xq.toOpenArray(c * K, (c + 1) * K - 1),
          xScales.toOpenArray(c * gpr, (c + 1) * gpr - 1),
          yc,
        )
        for r in 0 ..< M:
          if cast[uint32](yG[r * n + c]) != cast[uint32](yc[r]):
            inc mism
      check mism == 0

  test "packed int4 GEMM":
    for (M, K, gs, n) in [
      (9, 128, 64, 3), (4, 256, 128, 2), (5, 192, 96, 1), (6, 160, 80, 4)
    ]:
      let gpr = K div gs
      var wp = newSeq[byte](M * (K div 2))
      var xq = newSeq[I8](n * K)
      var wScales = newSeq[float32](M * gpr)
      var xScales = newSeq[float32](n * gpr)
      for i in 0 ..< wp.len:
        wp[i] = uint8((i * 29 + 5) and 0xff)
      for i in 0 ..< xq.len:
        xq[i] = I8(int8((i * 43 mod 239) - 119))
      for i in 0 ..< wScales.len:
        wScales[i] = 0.014'f32 + float32(i mod 5) * 0.0011'f32
      for i in 0 ..< xScales.len:
        xScales[i] = 0.019'f32 + float32(i mod 3) * 0.0017'f32

      var yG = newSeq[float32](M * n)
      dequantGemmI4Q8(wp, wScales, gs, xq, xScales, n, yG)
      var mism = 0
      for c in 0 ..< n:
        var yc = newSeq[float32](M)
        dequantGemvI4Q8(
          wp,
          wScales,
          gs,
          xq.toOpenArray(c * K, (c + 1) * K - 1),
          xScales.toOpenArray(c * gpr, (c + 1) * gpr - 1),
          yc,
        )
        for r in 0 ..< M:
          if cast[uint32](yG[r * n + c]) != cast[uint32](yc[r]):
            inc mism
      check mism == 0
