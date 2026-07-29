## nim_lowprec/parallel — row-sharded, multi-core wrappers over the fused GEMV kernels.
##
## WHY THESE ARE WRAPPERS, NOT THREADS INSIDE THE KERNELS
##
## The kernels stay single-threaded, exactly like ggml's `vec_dot_*`: llama.cpp
## shards rows one layer up, in `ggml_graph_compute`. A substrate that spawns its
## own threads cannot see that its caller is already running layers on a pool, and
## the two pools oversubscribe. So threading is opt-in and lives here, in separate
## procs, and the serial kernels remain the primitive.
##
## Every kernel in `simd/dequant` computes each output row independently, so a
## shard is just a row range: disjoint `w`/`scales`/`y` slices, `x` shared
## read-only. Nothing is synchronized because nothing is shared for writing, and
## the results are BIT-IDENTICAL to the serial call — the per-row arithmetic is
## untouched, only the order rows are visited changes. `tests/test_parallel.nim`
## enforces that bit-exactness.
##
## COST MODEL — threads are created per call (stdlib `createThread`, no pool, no
## dependency). That costs tens of microseconds per thread, so these are worth it
## for a 4096² weight matrix and pointless for a 64-row one; below
## `minRowsPerThread` rows per shard the call degrades to the serial kernel.
## Measured on an M1 Pro (8 P-cores + 2 E-cores), int8 4096² (`bench_parallel`):
##
##   threads   1     2     4     6     8
##   GFLOP/s   8.6  18.7  35.8  49.9  61.5
##
## `countProcessors()` reports 10 here and 10 is SLOWER than 8 — the E-cores don't
## carry this kernel as well as a P-core. Pass a `threads` count you measured; the
## default is only a starting point.
##
## The spawn tax matters more the faster the kernel is. For the int8-activation
## SDOT kernels it is no longer small change: `dequantGemvI4Q8` at 4096² runs in
## 0.62 ms serially, so a perfect 8-way split would be 0.08 ms, and the measured
## 0.21 ms is mostly ~8 thread creations. A caller in that regime should keep a
## persistent pool and call the serial kernels on row ranges itself — this module
## deliberately owns no long-lived threads.

import std/[cpuinfo, locks]
import ./intx, ./ggml
import ./simd/dequant

const minRowsPerThread* = 64
  ## Below this many rows per shard, spawning costs more than it saves.

proc shardCount(rows, threads: int): int =
  ## How many shards to actually use: never more than one per `minRowsPerThread`
  ## rows, never more than requested, never fewer than 1.
  let want = if threads > 0: threads else: countProcessors()
  result = max(1, min(want, rows div minRowsPerThread))

when compileOption("threads"):

  # ---------------- persistent worker pool ----------------
  #
  # The `par*` procs above create threads per call. That is fine at ~4 ms of work per
  # matvec but not at 0.5 ms: for `dequantGemvI4Q8` at m=4096, k=14336 the serial
  # kernel takes 2.64 ms, a perfect 8-way split would be 0.33 ms, and the measured
  # 0.52 ms is ~0.19 ms of thread creation — 36% of the runtime, spent on nothing.
  #
  # `LpPool` starts its workers once and hands them row ranges. The CALLER owns it, so
  # the substrate still has no ambient threads and no threading policy of its own: you
  # create one, reuse it across matvecs, and `shutdown` it. Results are unchanged and
  # still bit-identical to the serial kernels — sharding never alters a row's
  # arithmetic.
  #
  #   var pool = initPool(8)
  #   defer: pool.shutdown()
  #   pool.parDequantGemvI4Q8(w4, wScales, gs, xq, xScales, y)
  #
  # Measured, `dequantGemvI4Q8`, 8 workers (`bench_sdot`):
  #
  #                       4096²      m=4096, k=14336
  #   per-call threads   153.9        238.8   GFLOP/s
  #   pooled             225.9        273.4   GFLOP/s   (1.47x / 1.14x)
  #
  # Only the two int8-activation kernels have pooled forms so far — they are the
  # ones fast enough for spawn cost to dominate.

  type
    ShardFn = proc (payload: pointer; row0, rows: int) {.nimcall, gcsafe.}

    PoolCtx = object
      lock: Lock
      hasWork, allDone: Cond
      fn: ShardFn
      payload: pointer
      rows, nworkers, generation, pending: int
      stopping: bool

    LpPool* = object
      ctx: ptr PoolCtx
      threads: seq[Thread[(ptr PoolCtx, int)]]

  proc workerLoop(arg: (ptr PoolCtx, int)) {.thread.} =
    let ctx = arg[0]
    let id = arg[1]
    var seen = 0
    while true:
      acquire(ctx.lock)
      # `generation` rather than a flag: a worker that was still finishing the last
      # job when the next one was posted must not miss it, and a spurious wakeup
      # must not run a job twice.
      while ctx.generation == seen and not ctx.stopping:
        wait(ctx.hasWork, ctx.lock)
      if ctx.stopping:
        release(ctx.lock)
        break
      seen = ctx.generation
      let fn = ctx.fn
      let payload = ctx.payload
      let rows = ctx.rows
      let nw = ctx.nworkers
      release(ctx.lock)

      let per = rows div nw
      let row0 = per * id
      let myRows = if id == nw - 1: rows - per * id else: per
      if myRows > 0: fn(payload, row0, myRows)

      acquire(ctx.lock)
      dec ctx.pending
      if ctx.pending == 0: signal(ctx.allDone)
      release(ctx.lock)

  proc initPool*(threads = 0): LpPool =
    ## Start `threads` workers (default `countProcessors()`, which counts E-cores —
    ## measure before trusting it). Call `shutdown` when done.
    let nw = max(1, if threads > 0: threads else: countProcessors())
    result.ctx = cast[ptr PoolCtx](allocShared0(sizeof(PoolCtx)))
    initLock(result.ctx.lock)
    initCond(result.ctx.hasWork)
    initCond(result.ctx.allDone)
    result.ctx.nworkers = nw
    result.threads = newSeq[Thread[(ptr PoolCtx, int)]](nw)
    for i in 0 ..< nw:
      createThread(result.threads[i], workerLoop, (result.ctx, i))

  proc shutdown*(p: var LpPool) =
    ## Stop and join the workers. Idempotent.
    if p.ctx == nil: return
    acquire(p.ctx.lock)
    p.ctx.stopping = true
    broadcast(p.ctx.hasWork)
    release(p.ctx.lock)
    joinThreads(p.threads)
    deinitCond(p.ctx.allDone)
    deinitCond(p.ctx.hasWork)
    deinitLock(p.ctx.lock)
    deallocShared(p.ctx)
    p.ctx = nil
    p.threads = @[]

  proc threadCount*(p: LpPool): int =
    ## Workers in this pool, 0 if it was never initialized or is shut down.
    if p.ctx == nil: 0 else: p.ctx.nworkers

  proc dispatch(p: var LpPool; fn: ShardFn; payload: pointer; rows: int) =
    ## Post one job and block until every worker has finished its slice.
    acquire(p.ctx.lock)
    p.ctx.fn = fn
    p.ctx.payload = payload
    p.ctx.rows = rows
    p.ctx.pending = p.ctx.nworkers
    inc p.ctx.generation
    broadcast(p.ctx.hasWork)
    while p.ctx.pending > 0:
      wait(p.ctx.allDone, p.ctx.lock)
    release(p.ctx.lock)

  template defParGemv(parName, workerName, WT, ST, kern, wPerElem: untyped) =
    ## Generates `parName`, the row-sharded form of `kern`. `wPerElem` is how many
    ## weights share one storage element — 1 for one-per-byte int8, 2 for packed
    ## nibbles — so a row occupies `k div wPerElem` elements. (A plain divisor, not
    ## an expression referencing the generated proc's locals: symbols declared in a
    ## template are gensym'd, so an injected `k div 2` would not bind to them.)
    ##
    ## `workerName` must be unique per instantiation: types declared in a template
    ## are gensym'd but PROCS are not, so three `worker`s would collide into an
    ## ambiguous overload set that `createThread` cannot resolve.
    type Shard = object
      w: ptr UncheckedArray[WT]
      sc: ptr UncheckedArray[ST]
      x: ptr UncheckedArray[float32]
      y: ptr UncheckedArray[float32]
      row0, rows, k, shape, wStride, sStride: int

    proc workerName(s: Shard) {.thread.} =
      # The public serial kernel, over this shard's rows. Shapes come from the
      # slice lengths, which is why no library change was needed to make this work.
      kern(s.w.toOpenArray(s.row0 * s.wStride, (s.row0 + s.rows) * s.wStride - 1),
           s.sc.toOpenArray(s.row0 * s.sStride, (s.row0 + s.rows) * s.sStride - 1),
           s.shape,
           s.x.toOpenArray(0, s.k - 1),
           s.y.toOpenArray(s.row0, s.row0 + s.rows - 1))

    proc parName*(w: openArray[WT]; sc: openArray[ST]; shape: int;
                  x: openArray[float32]; y: var openArray[float32];
                  threads = 0) =
      let rows = y.len
      let k = x.len
      let nt = shardCount(rows, threads)
      if nt <= 1:
        kern(w, sc, shape, x, y)          # not worth threading — stay serial
        return
      let wStride = k div wPerElem
      let sStride = k div shape
      var thr = newSeq[Thread[Shard]](nt)
      let per = rows div nt
      for t in 0 ..< nt:
        createThread(thr[t], workerName, Shard(
          w: cast[ptr UncheckedArray[WT]](unsafeAddr w[0]),
          sc: cast[ptr UncheckedArray[ST]](unsafeAddr sc[0]),
          x: cast[ptr UncheckedArray[float32]](unsafeAddr x[0]),
          y: cast[ptr UncheckedArray[float32]](addr y[0]),
          row0: per * t,
          rows: (if t == nt - 1: rows - per * t else: per),   # last shard takes the remainder
          k: k, shape: shape, wStride: wStride, sStride: sStride))
      joinThreads(thr)

  # group-scale kernels: `shape` is groupSize, scales are (k div groupSize) per row
  defParGemv(parDequantGemv,   shardWorkerI8, I8,   float32, dequantGemv,   1)  # one int8 per byte
  defParGemv(parDequantGemvI4, shardWorkerI4, byte, float32, dequantGemvI4, 2)  # two nibbles per byte
  defParGemv(parDequantGemvF4, shardWorkerF4, byte, float32, dequantGemvF4, 2)  # two fp4 per byte

  template defParBlockGemv(parName, workerName, BT, kern: untyped) =
    ## The ggml block kernels: `shape` is blocksPerRow and the scale lives inside
    ## each block, so there is no separate scales array to shard.
    type Shard = object
      w: ptr UncheckedArray[BT]
      x: ptr UncheckedArray[float32]
      y: ptr UncheckedArray[float32]
      row0, rows, k, bpr: int

    proc workerName(s: Shard) {.thread.} =
      kern(s.w.toOpenArray(s.row0 * s.bpr, (s.row0 + s.rows) * s.bpr - 1),
           s.bpr,
           s.x.toOpenArray(0, s.k - 1),
           s.y.toOpenArray(s.row0, s.row0 + s.rows - 1))

    proc parName*(w: openArray[BT]; blocksPerRow: int;
                  x: openArray[float32]; y: var openArray[float32];
                  threads = 0) =
      let rows = y.len
      let nt = shardCount(rows, threads)
      if nt <= 1:
        kern(w, blocksPerRow, x, y)
        return
      var thr = newSeq[Thread[Shard]](nt)
      let per = rows div nt
      for t in 0 ..< nt:
        createThread(thr[t], workerName, Shard(
          w: cast[ptr UncheckedArray[BT]](unsafeAddr w[0]),
          x: cast[ptr UncheckedArray[float32]](unsafeAddr x[0]),
          y: cast[ptr UncheckedArray[float32]](addr y[0]),
          row0: per * t,
          rows: (if t == nt - 1: rows - per * t else: per),
          k: x.len, bpr: blocksPerRow))
      joinThreads(thr)

  defParBlockGemv(parDequantGemvQ8_0, shardWorkerQ8_0, BlockQ8_0, dequantGemvQ8_0)
  defParBlockGemv(parDequantGemvQ4_0, shardWorkerQ4_0, BlockQ4_0, dequantGemvQ4_0)

  template defParQ8Gemv(parName, workerName, trampolineName, WT, kern, wPerElem: untyped) =
    ## The int8-ACTIVATION kernels: the activation vector and its scales are shared
    ## read-only by every shard (they are per-column, not per-row), so only the
    ## weights, weight scales and `y` get sliced.
    type Shard = object
      w: ptr UncheckedArray[WT]
      ws, xs: ptr UncheckedArray[float32]
      xq: ptr UncheckedArray[I8]
      y: ptr UncheckedArray[float32]
      row0, rows, k, gs, wStride, sStride: int

    proc workerName(s: Shard) {.thread.} =
      kern(s.w.toOpenArray(s.row0 * s.wStride, (s.row0 + s.rows) * s.wStride - 1),
           s.ws.toOpenArray(s.row0 * s.sStride, (s.row0 + s.rows) * s.sStride - 1),
           s.gs,
           s.xq.toOpenArray(0, s.k - 1),
           s.xs.toOpenArray(0, s.sStride - 1),
           s.y.toOpenArray(s.row0, s.row0 + s.rows - 1))

    # Same slicing, but the row range arrives from the pool instead of being baked
    # into the shard, so one descriptor serves every worker.
    proc trampolineName(payload: pointer; row0, rows: int) {.nimcall, gcsafe.} =
      let s = cast[ptr Shard](payload)
      kern(s.w.toOpenArray(row0 * s.wStride, (row0 + rows) * s.wStride - 1),
           s.ws.toOpenArray(row0 * s.sStride, (row0 + rows) * s.sStride - 1),
           s.gs,
           s.xq.toOpenArray(0, s.k - 1),
           s.xs.toOpenArray(0, s.sStride - 1),
           s.y.toOpenArray(row0, row0 + rows - 1))

    proc parName*(pool: var LpPool; w: openArray[WT]; wScales: openArray[float32];
                  groupSize: int; xq: openArray[I8]; xScales: openArray[float32];
                  y: var openArray[float32]) =
      ## Pooled form: no thread creation per call. Bit-identical to the serial kernel.
      let rows = y.len
      let k = xq.len
      if pool.threadCount <= 1 or rows < 2 * minRowsPerThread:
        kern(w, wScales, groupSize, xq, xScales, y)
        return
      var shard = Shard(
        w: cast[ptr UncheckedArray[WT]](unsafeAddr w[0]),
        ws: cast[ptr UncheckedArray[float32]](unsafeAddr wScales[0]),
        xs: cast[ptr UncheckedArray[float32]](unsafeAddr xScales[0]),
        xq: cast[ptr UncheckedArray[I8]](unsafeAddr xq[0]),
        y: cast[ptr UncheckedArray[float32]](addr y[0]),
        row0: 0, rows: rows, k: k, gs: groupSize,
        wStride: k div wPerElem, sStride: k div groupSize)
      pool.dispatch(trampolineName, addr shard, rows)

    proc parName*(w: openArray[WT]; wScales: openArray[float32]; groupSize: int;
                  xq: openArray[I8]; xScales: openArray[float32];
                  y: var openArray[float32]; threads = 0) =
      let rows = y.len
      let k = xq.len
      let nt = shardCount(rows, threads)
      if nt <= 1:
        kern(w, wScales, groupSize, xq, xScales, y)
        return
      let wStride = k div wPerElem
      let sStride = k div groupSize
      var thr = newSeq[Thread[Shard]](nt)
      let per = rows div nt
      for t in 0 ..< nt:
        createThread(thr[t], workerName, Shard(
          w: cast[ptr UncheckedArray[WT]](unsafeAddr w[0]),
          ws: cast[ptr UncheckedArray[float32]](unsafeAddr wScales[0]),
          xs: cast[ptr UncheckedArray[float32]](unsafeAddr xScales[0]),
          xq: cast[ptr UncheckedArray[I8]](unsafeAddr xq[0]),
          y: cast[ptr UncheckedArray[float32]](addr y[0]),
          row0: per * t,
          rows: (if t == nt - 1: rows - per * t else: per),
          k: k, gs: groupSize, wStride: wStride, sStride: sStride))
      joinThreads(thr)

  defParQ8Gemv(parDequantGemvQ8,   shardWorkerQ8i8, poolTrampQ8i8, I8,   dequantGemvQ8,   1)
  defParQ8Gemv(parDequantGemvI4Q8, shardWorkerQ8i4, poolTrampQ8i4, byte, dequantGemvI4Q8, 2)

  template defParGemm(parName, trampName, WT, kern, wPerElem: untyped) =
    ## Row-sharded multi-column GEMM. `y` is M×n ROW-major, so a row range is one
    ## contiguous block; `xq`/`xScales` (all n columns) are shared read-only.
    type Shard = object
      w: ptr UncheckedArray[WT]
      ws, xs: ptr UncheckedArray[float32]
      xq: ptr UncheckedArray[I8]
      y: ptr UncheckedArray[float32]
      k, gs, n, wStride, sStride: int

    proc trampName(payload: pointer; row0, rows: int) {.nimcall, gcsafe.} =
      let s = cast[ptr Shard](payload)
      kern(s.w.toOpenArray(row0 * s.wStride, (row0 + rows) * s.wStride - 1),
           s.ws.toOpenArray(row0 * s.sStride, (row0 + rows) * s.sStride - 1),
           s.gs,
           s.xq.toOpenArray(0, s.n * s.k - 1),
           s.xs.toOpenArray(0, s.n * s.sStride - 1), s.n,
           s.y.toOpenArray(row0 * s.n, (row0 + rows) * s.n - 1))

    proc parName*(pool: var LpPool; w: openArray[WT]; wScales: openArray[float32];
                  groupSize: int; xq: openArray[I8]; xScales: openArray[float32];
                  n: int; y: var openArray[float32]) =
      ## Pooled multi-column GEMM — the 8-thread n=8 int8 configuration measured
      ## 564 GFLOP/s on an M1 Pro. Bit-identical per column to the serial kernel.
      let rows = y.len div n
      let k = xq.len div n
      if pool.threadCount <= 1 or rows < 2 * minRowsPerThread:
        kern(w, wScales, groupSize, xq, xScales, n, y)
        return
      var shard = Shard(
        w: cast[ptr UncheckedArray[WT]](unsafeAddr w[0]),
        ws: cast[ptr UncheckedArray[float32]](unsafeAddr wScales[0]),
        xs: cast[ptr UncheckedArray[float32]](unsafeAddr xScales[0]),
        xq: cast[ptr UncheckedArray[I8]](unsafeAddr xq[0]),
        y: cast[ptr UncheckedArray[float32]](addr y[0]),
        k: k, gs: groupSize, n: n,
        wStride: k div wPerElem, sStride: k div groupSize)
      pool.dispatch(trampName, addr shard, rows)

  defParGemm(parDequantGemmQ8,   poolTrampGemmQ8, I8,   dequantGemmQ8,   1)
  defParGemm(parDequantGemmI4Q8, poolTrampGemmI4, byte, dequantGemmI4Q8, 2)

else:
  # Built without --threads:on: the parallel names still exist and still compute
  # the right answer, so callers don't need their own `when` guards.
  proc parDequantGemv*(wq: openArray[I8]; scales: openArray[float32]; groupSize: int;
                       x: openArray[float32]; y: var openArray[float32]; threads = 0) =
    dequantGemv(wq, scales, groupSize, x, y)

  proc parDequantGemvI4*(w: openArray[byte]; scales: openArray[float32]; groupSize: int;
                         x: openArray[float32]; y: var openArray[float32]; threads = 0) =
    dequantGemvI4(w, scales, groupSize, x, y)

  proc parDequantGemvF4*(w: openArray[byte]; scales: openArray[float32]; groupSize: int;
                         x: openArray[float32]; y: var openArray[float32]; threads = 0) =
    dequantGemvF4(w, scales, groupSize, x, y)

  proc parDequantGemvQ8_0*(w: openArray[BlockQ8_0]; blocksPerRow: int;
                           x: openArray[float32]; y: var openArray[float32]; threads = 0) =
    dequantGemvQ8_0(w, blocksPerRow, x, y)

  proc parDequantGemvQ4_0*(w: openArray[BlockQ4_0]; blocksPerRow: int;
                           x: openArray[float32]; y: var openArray[float32]; threads = 0) =
    dequantGemvQ4_0(w, blocksPerRow, x, y)

  proc parDequantGemvQ8*(w: openArray[I8]; wScales: openArray[float32]; groupSize: int;
                         xq: openArray[I8]; xScales: openArray[float32];
                         y: var openArray[float32]; threads = 0) =
    dequantGemvQ8(w, wScales, groupSize, xq, xScales, y)

  proc parDequantGemvI4Q8*(w: openArray[byte]; wScales: openArray[float32]; groupSize: int;
                           xq: openArray[I8]; xScales: openArray[float32];
                           y: var openArray[float32]; threads = 0) =
    dequantGemvI4Q8(w, wScales, groupSize, xq, xScales, y)

  # --threads:off builds still get the GEMM names (serial), pool type absent.
  proc parDequantGemmQ8*(w: openArray[I8]; wScales: openArray[float32]; groupSize: int;
                         xq: openArray[I8]; xScales: openArray[float32]; n: int;
                         y: var openArray[float32]) =
    dequantGemmQ8(w, wScales, groupSize, xq, xScales, n, y)

  proc parDequantGemmI4Q8*(w: openArray[byte]; wScales: openArray[float32]; groupSize: int;
                           xq: openArray[I8]; xScales: openArray[float32]; n: int;
                           y: var openArray[float32]) =
    dequantGemmI4Q8(w, wScales, groupSize, xq, xScales, n, y)
