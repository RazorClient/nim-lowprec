## benchmarks/harness — shared timing + reporting for the scalar-vs-SIMD comparison.
##
## Every microbenchmark declares ONE workload, then times two or more paths over
## it: a scalar reference loop and the `-d:lpSimd=` kernel. Each path is warmed
## once (page in `dst`, prime the i-cache), then run `-d:benchReps=N` times and
## reported at its BEST time — for a fixed amount of work the minimum is the
## least noisy estimator, since every source of noise only ever adds time.
##
## Paths are also cross-checked: each records one checksum, compared against the
## first path's. Elementwise kernels must agree EXACTLY (`tol = 0`), so a bench
## that silently stopped computing the right thing can't post a speedup. GEMV and
## dot are reductions — the vector summation order differs from a scalar loop —
## so those groups pass a relative `tol` instead.
##
## Build the same benchmark both ways to compare the two library paths directly:
##   nim c -r -d:release benchmarks/bench_gemv.nim                  # native SIMD
##   nim c -r -d:release -d:lpSimd=scalar benchmarks/bench_gemv.nim # fallback
## (`nimble bench`, `nimble benchScalar`, `nimble benchAll` do this for all of them.)

import std/[times, monotimes, strformat, strutils, math]
import nim_lowprec/simd/target

export simdBackend

const benchReps* {.intdefine.}: int = 3
  ## Timed runs per path, after one untimed warm-up. Best time wins.

type
  BenchRow = object
    label: string
    sec: float
    chk: float

  BenchGroup* = object
    title, unit, workLabel: string
    work: float     ## elements or flops per run — the throughput numerator
    scale: float    ## 1e6 for Melem/s, 1e9 for GFLOP/s
    tol: float      ## allowed relative checksum drift between paths
    rows: seq[BenchRow]

func withSep(n: int): string = ($n).insertSep(',')

func elemGroup*(title: string; n: int; tol = 0.0): BenchGroup =
  ## Throughput in elements/s. `tol = 0` demands bit-identical checksums.
  BenchGroup(title: title, unit: "Melem/s", workLabel: "N = " & withSep(n),
             work: n.float, scale: 1e6, tol: tol)

func flopGroup*(title, shape: string; flops: float; tol = 1e-3): BenchGroup =
  ## Throughput in GFLOP/s, for the GEMV/dot reductions. Tolerance, not exactness.
  BenchGroup(title: title, unit: "GFLOP/s", workLabel: shape,
             work: flops, scale: 1e9, tol: tol)

template measure*(g: var BenchGroup; name: string; checksum, body: untyped) =
  ## Time `body` (warm-up + `benchReps` runs, best kept) and record `checksum` —
  ## an expression reading the result, evaluated after the last run. It both keeps
  ## the optimizer from deleting the work and proves the paths agree.
  block:
    body
    var best = Inf
    for _ in 1 .. benchReps:
      let t0 = getMonoTime()
      body
      let s = float((getMonoTime() - t0).inNanoseconds) / 1e9
      if s < best: best = s
    g.rows.add BenchRow(label: name, sec: best, chk: float(checksum))

template scalarRow*(g: var BenchGroup; checksum, body: untyped) =
  ## The reference path — always first, so it is the speedup baseline.
  measure(g, "scalar ref", checksum, body)

template kernelRow*(g: var BenchGroup; checksum, body: untyped) =
  ## The library kernel, labelled by the ISA it was actually compiled for.
  measure(g, simdBackend & " kernel", checksum, body)

proc report*(g: BenchGroup) =
  ## One group: a header, a row per path with ms / throughput / speedup, and the
  ## checksum verdict. Speedups are relative to the first row.
  echo ""
  echo &"── {g.title}  ·  {g.workLabel}  ·  best of {benchReps}"
  echo "   " & alignLeft("path", 16) & align("ms", 10) &
       align(g.unit, 12) & align("speedup", 10)
  let base = g.rows[0].sec
  for r in g.rows:
    echo "   " & alignLeft(r.label, 16) & align(&"{r.sec * 1e3:.2f}", 10) &
         align(&"{g.work / r.sec / g.scale:.1f}", 12) &
         align(&"{base / r.sec:.2f}x", 10)

  let want = g.rows[0].chk
  var bad: seq[string]
  for r in g.rows[1 .. ^1]:
    let drift = abs(r.chk - want) / max(abs(want), 1e-30)
    if r.chk != want and (g.tol <= 0.0 or drift > g.tol or isNaN(drift)):
      bad.add &"{r.label} = {r.chk:.9g}"
  if bad.len == 0:
    let how = if g.tol <= 0.0: "bit-identical" else: &"within {g.tol:.0e}"
    echo &"   checksum {want:.9g} — all paths agree ({how})"
  else:
    echo &"   CHECKSUM MISMATCH: baseline {want:.9g} vs " & bad.join(", ")
    quit 1

proc warmCore() =
  ## Spin ~50 ms of dependent fp work before the first measurement.
  ##
  ## On Apple Silicon a P-core starts at a low clock and ramps: the same 17 MB
  ## streaming loop measures 10.5 GB/s as the first thing a process does and
  ## 17.1 GB/s once warm. Each group's own warm-up run usually covers this, but the
  ## FIRST group in a binary would otherwise be timed against a cold clock and read
  ## ~40% slow — which is exactly the row a reader compares everything else to.
  var acc = 1.000001'f32
  let t0 = getMonoTime()
  while (getMonoTime() - t0).inMilliseconds < 50:
    for _ in 1 .. 100_000: acc *= 1.0000001'f32
  if acc == 0.0'f32: echo ""       # keep the loop from being optimized away

proc header*(what: string) =
  warmCore()
  echo &"{what}   backend = {simdBackend}   (-d:lpSimd={lpSimd})"
