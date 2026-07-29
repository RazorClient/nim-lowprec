## Row-sharded GEMV must be BIT-IDENTICAL to the serial kernel.
##
## Sharding only changes which thread visits a row, never the arithmetic within a
## row — so unlike the SIMD-vs-scalar reduction tests (tolerance), this one demands
## exact equality. Thread counts are chosen to cover: no sharding, a count that
## divides the row count, one that leaves a remainder for the last shard, and one
## far larger than the work (must degrade to serial, not spawn 64 threads).

import std/unittest
import nim_lowprec/[intx, mxfloat, ggml, float16, simd/dequant, parallel]

const
  K = 512
  gs = 64
  gpr = K div gs
  threadCounts = [0, 1, 2, 3, 64] # 0 = library default, 3 = ragged, 64 = over-subscribed

# 300 rows: not a multiple of 2 or 3, so the last shard always takes a remainder.
const rowCounts = [300, 64, 1]

proc mkX(): seq[float32] =
  result = newSeq[float32](K)
  var u = 7919'u32
  for i in 0 ..< K:
    u = u * 1664525'u32 + 1013904223'u32
    result[i] = float32(u shr 16) / 32768.0'f32 - 1.0'f32

let x = mkX()

template checkExact(serialCall, parCall: untyped) =
  ## Both write into `y`; compare the two results bit-for-bit via their raw bits
  ## (so a NaN would also have to match, and -0.0 wouldn't pass as 0.0).
  # {.inject.}: the caller's `serialCall`/`parCall` expressions name `y` and `nt`,
  # and template-declared symbols are gensym'd unless injected.
  var y {.inject.} = newSeq[float32](M)
  serialCall
  let want = y
  for nt {.inject.} in threadCounts:
    for i in 0 ..< M:
      y[i] = 0.0'f32
    parCall
    var mism = 0
    for i in 0 ..< M:
      if cast[uint32](y[i]) != cast[uint32](want[i]):
        inc mism
    check mism == 0

suite "parallel GEMV == serial GEMV, bit-for-bit":
  test "int8 weights":
    for M in rowCounts:
      var wq = newSeq[I8](M * K)
      var scales = newSeq[float32](M * gpr)
      for i in 0 ..< wq.len:
        wq[i] = toI8(float32((i mod 15) - 7))
      for i in 0 ..< scales.len:
        scales[i] = 0.02'f32 + float32(i mod 3) * 0.001'f32
      checkExact(
        dequantGemv(wq, scales, gs, x, y), parDequantGemv(wq, scales, gs, x, y, nt)
      )

  test "packed int4 weights":
    for M in rowCounts:
      var w4 = newSeq[byte](M * (K div 2))
      var scales = newSeq[float32](M * gpr)
      for i in 0 ..< w4.len:
        w4[i] = uint8((i * 7 + 3) and 0xff)
      for i in 0 ..< scales.len:
        scales[i] = 0.02'f32 + float32(i mod 3) * 0.001'f32
      checkExact(
        dequantGemvI4(w4, scales, gs, x, y), parDequantGemvI4(w4, scales, gs, x, y, nt)
      )

  test "packed MXFP4 weights + E8M0 scales":
    for M in rowCounts:
      var wf4 = newSeq[byte](M * (K div 2))
      var scales = newSeq[float32](M * gpr)
      for i in 0 ..< wf4.len:
        wf4[i] = uint8((i * 11 + 5) and 0xff)
      for i in 0 ..< scales.len:
        scales[i] = toFloat32(toE8M0(0.03125'f32))
      checkExact(
        dequantGemvF4(wf4, scales, gs, x, y),
        parDequantGemvF4(wf4, scales, gs, x, y, nt),
      )

  test "ggml Q8_0 blocks":
    const bpr = K div QK
    for M in rowCounts:
      var w = newSeq[BlockQ8_0](M * bpr)
      for b in 0 ..< w.len:
        w[b].d = toF16(0.0125'f32 + float32(b mod 3) * 0.001'f32)
        for i in 0 ..< QK:
          w[b].qs[i] = int8(((b + i) mod 255) - 127)
      checkExact(dequantGemvQ8_0(w, bpr, x, y), parDequantGemvQ8_0(w, bpr, x, y, nt))

  test "ggml Q4_0 blocks":
    const bpr = K div QK
    for M in rowCounts:
      var w = newSeq[BlockQ4_0](M * bpr)
      for b in 0 ..< w.len:
        w[b].d = toF16(0.0125'f32 + float32(b mod 3) * 0.001'f32)
        for i in 0 ..< QK div 2:
          w[b].qs[i] = uint8((b * 7 + i * 11) and 0xff)
      checkExact(dequantGemvQ4_0(w, bpr, x, y), parDequantGemvQ4_0(w, bpr, x, y, nt))

when compileOption("threads"):
  # The pool exists only in threaded builds; the par* GEMV/GEMM names still
  # exist without --threads:on (as serial aliases) and are covered above.
  suite "pooled GEMV == serial GEMV, bit-for-bit":
    # A pool reused across calls must give the same answer as the serial kernel every
    # time, including when the row count doesn't divide evenly and when a later call
    # follows an earlier one on the same workers (the generation counter has to make
    # each job run exactly once).
    test "pooled int8-activation kernels, reused across calls":
      const gs = 64
      var pool = initPool(4)
      try:
        for M in [300, 129, 64]:
          var wq = newSeq[I8](M * K)
          var w4 = newSeq[byte](M * (K div 2))
          var wScales = newSeq[float32](M * gpr)
          var xq = newSeq[I8](K)
          var xScales = newSeq[float32](gpr)
          for i in 0 ..< wq.len:
            wq[i] = I8(int8((i mod 200) - 100))
          for i in 0 ..< w4.len:
            w4[i] = uint8((i * 13 + 7) and 0xff)
          for i in 0 ..< wScales.len:
            wScales[i] = 0.019'f32 + float32(i mod 4) * 0.0007'f32
          for i in 0 ..< K:
            xq[i] = I8(int8((i mod 150) - 75))
          for i in 0 ..< gpr:
            xScales[i] = 0.027'f32 + float32(i mod 3) * 0.0011'f32

          var want = newSeq[float32](M)
          var got = newSeq[float32](M)

          dequantGemvQ8(wq, wScales, gs, xq, xScales, want)
          for _ in 1 .. 3: # same pool, repeated dispatches
            for i in 0 ..< M:
              got[i] = 0.0'f32
            pool.parDequantGemvQ8(wq, wScales, gs, xq, xScales, got)
            var mism = 0
            for i in 0 ..< M:
              if cast[uint32](got[i]) != cast[uint32](want[i]):
                inc mism
            check mism == 0

          dequantGemvI4Q8(w4, wScales, gs, xq, xScales, want)
          for _ in 1 .. 3:
            for i in 0 ..< M:
              got[i] = 0.0'f32
            pool.parDequantGemvI4Q8(w4, wScales, gs, xq, xScales, got)
            var mism = 0
            for i in 0 ..< M:
              if cast[uint32](got[i]) != cast[uint32](want[i]):
                inc mism
            check mism == 0
      finally:
        pool.shutdown()

    test "shutdown is idempotent and a fresh pool still works":
      var p = initPool(2)
      p.shutdown()
      p.shutdown()
      check p.threadCount == 0
      var q = initPool(2)
      check q.threadCount == 2
      q.shutdown()

  suite "pooled GEMM == serial GEMM, bit-for-bit":
    test "int8 and packed int4 multi-column, reused pool":
      const gs = 64
      const n = 3
      var pool = initPool(4)
      try:
        for M in [300, 128]:
          let gpr = K div gs
          var wq = newSeq[I8](M * K)
          var w4 = newSeq[byte](M * (K div 2))
          var wScales = newSeq[float32](M * gpr)
          var xq = newSeq[I8](n * K)
          var xScales = newSeq[float32](n * gpr)
          for i in 0 ..< wq.len:
            wq[i] = I8(int8((i mod 200) - 100))
          for i in 0 ..< w4.len:
            w4[i] = uint8((i * 13 + 7) and 0xff)
          for i in 0 ..< wScales.len:
            wScales[i] = 0.019'f32 + float32(i mod 4) * 0.0007'f32
          for i in 0 ..< xq.len:
            xq[i] = I8(int8((i mod 150) - 75))
          for i in 0 ..< xScales.len:
            xScales[i] = 0.027'f32 + float32(i mod 3) * 0.0011'f32

          var want = newSeq[float32](M * n)
          var got = newSeq[float32](M * n)

          dequantGemmQ8(wq, wScales, gs, xq, xScales, n, want)
          for _ in 1 .. 2:
            for i in 0 ..< got.len:
              got[i] = 0.0'f32
            pool.parDequantGemmQ8(wq, wScales, gs, xq, xScales, n, got)
            var mism = 0
            for i in 0 ..< want.len:
              if cast[uint32](got[i]) != cast[uint32](want[i]):
                inc mism
            check mism == 0

          dequantGemmI4Q8(w4, wScales, gs, xq, xScales, n, want)
          for _ in 1 .. 2:
            for i in 0 ..< got.len:
              got[i] = 0.0'f32
            pool.parDequantGemmI4Q8(w4, wScales, gs, xq, xScales, n, got)
            var mism = 0
            for i in 0 ..< want.len:
              if cast[uint32](got[i]) != cast[uint32](want[i]):
                inc mism
            check mism == 0
      finally:
        pool.shutdown()
