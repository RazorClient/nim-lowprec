## Elementwise dequant: scalar `dequantize` vs `dequantizeBatch` (value × per-group
## scale → fp32). Both are bit-exact, so the checksums must match exactly.
##
## The asymmetric group is the interesting control: `dequantizeBatch` has no
## zero-point fast path, so it falls through to the exact scalar code — that row
## is EXPECTED to sit at ~1x, and a suspiciously fast one would mean the fallback
## stopped being taken.

import nim_lowprec/[intx, quant, simd/dequant]
import ./harness

const N = 16_000_000
const gs = 128
const mid = N div 2

var x = newSeq[float32](N)
for i in 0 ..< N:
  x[i] = float32(i mod 1000) * 0.01'f32 - 5.0'f32
var dst = newSeq[float32](N)

header "elementwise dequant"

block: # ---- int8, symmetric per-group: the vectorized path ----
  let p = calibrateSymmetric(x, gs, 127.0'f32)
  var q = newSeq[I8](N)
  quantize(x, p, q)
  var g = elemGroup("int8 dequant  (sym, group " & $gs & ")", N)
  g.scalarRow dst[mid]:
    dequantize(q, p, dst)
  g.kernelRow dst[mid]:
    dequantizeBatch(q, p, dst)
  g.report()

block: # ---- int4 stored one-per-byte (unpacked) ----
  let p = calibrateSymmetric(x, gs, 7.0'f32)
  var q = newSeq[I4](N)
  quantize(x, p, q)
  var g = elemGroup("int4 dequant  (sym, group " & $gs & ")", N)
  g.scalarRow dst[mid]:
    dequantize(q, p, dst)
  g.kernelRow dst[mid]:
    dequantizeBatch(q, p, dst)
  g.report()

block: # ---- asymmetric: no SIMD path, must fall back exactly ----
  let p = calibrateAsymmetric(x, gs, -128.0'f32, 127.0'f32)
  var q = newSeq[I8](N)
  quantize(x, p, q)
  var g = elemGroup("int8 dequant  (asym → scalar fallback)", N)
  g.scalarRow dst[mid]:
    dequantize(q, p, dst)
  g.kernelRow dst[mid]:
    dequantizeBatch(q, p, dst)
  g.report()
