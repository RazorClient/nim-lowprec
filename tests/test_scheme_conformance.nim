## BLOCK-SCHEME conformance — external oracles for the schemes, not the elements
## (CONFORMANCE.md P0/P1). Golden vectors come from `nimble refs`:
##
##   ggml Q8_0/Q4_0     quantize + dequantize vs gguf-py (llama.cpp's own numpy
##                      implementation, documented bit-exact vs ggml-quants.c)
##   ggml Q4_K/Q6_K     DEQUANTIZE (the ingestion direction) of synthetic valid
##                      super-blocks vs gguf-py
##   MX (fp4/fp6)       calibrateMX + quantize + dequantize end-to-end vs an
##                      independent numpy implementation of OCP MX v1.0 with
##                      ml_dtypes element rounding
##   NVFP4              quantizeNVFP4 vs a transcription of TransformerEngine's
##                      NVFP4QuantizerRef — codes, e4m3 block scales, and the
##                      fp32 global scale all compared bit-for-bit
##
## Every comparison here is BIT-exact. Suites skip when vectors are absent.

import std/[unittest, os, math]
import nim_lowprec

# The references round mul-then-sub in two steps (numpy); no compiler FMAs here.
{.localPassc: "-ffp-contract=off".}

const refDir = currentSourcePath().parentDir / "refs"

proc readBytes(name: string): seq[byte] =
  let path = refDir / name
  if not fileExists(path):
    return @[]
  result = newSeq[byte](getFileSize(path))
  let f = open(path)
  defer:
    f.close()
  discard f.readBuffer(addr result[0], result.len)

proc readF32(name: string): seq[float32] =
  let raw = readBytes(name)
  result = newSeq[float32](raw.len div 4)
  if raw.len > 0:
    copyMem(addr result[0], unsafeAddr raw[0], raw.len)

template bitEq(got, want: openArray[float32], mism: var int) =
  for i in 0 ..< want.len:
    if cast[uint32](got[i]) != cast[uint32](want[i]):
      inc mism

suite "ggml Q8_0/Q4_0 vs gguf-py (external oracle)":
  let x = readF32("ref_scheme_input.bin")

  test "Q8_0 quantize bytes + dequant, bit-exact":
    if x.len == 0:
      skip()
    else:
      let wantBytes = readBytes("ref_ggml_q8_0_bytes.bin")
      let wantDq = readF32("ref_ggml_q8_0_dq.bin")
      var blocks = newSeq[BlockQ8_0](x.len div QK)
      quantizeQ8_0(x, blocks)
      var rawGot = newSeq[byte](blocks.len * sizeof(BlockQ8_0))
      copyMem(addr rawGot[0], addr blocks[0], rawGot.len)
      check rawGot == wantBytes
      # dequantize THEIR bytes (independent of our quantizer) vs THEIR floats
      var theirs = newSeq[BlockQ8_0](wantBytes.len div sizeof(BlockQ8_0))
      copyMem(addr theirs[0], unsafeAddr wantBytes[0], wantBytes.len)
      var dq = newSeq[float32](theirs.len * QK)
      dequantizeQ8_0(theirs, dq)
      var mism = 0
      bitEq(dq, wantDq, mism)
      check mism == 0

  test "Q4_0 quantize bytes + dequant, bit-exact":
    if x.len == 0:
      skip()
    else:
      let wantBytes = readBytes("ref_ggml_q4_0_bytes.bin")
      let wantDq = readF32("ref_ggml_q4_0_dq.bin")
      var blocks = newSeq[BlockQ4_0](x.len div QK)
      quantizeQ4_0(x, blocks)
      var rawGot = newSeq[byte](blocks.len * sizeof(BlockQ4_0))
      copyMem(addr rawGot[0], addr blocks[0], rawGot.len)
      check rawGot == wantBytes
      var theirs = newSeq[BlockQ4_0](wantBytes.len div sizeof(BlockQ4_0))
      copyMem(addr theirs[0], unsafeAddr wantBytes[0], wantBytes.len)
      var dq = newSeq[float32](theirs.len * QK)
      dequantizeQ4_0(theirs, dq)
      var mism = 0
      bitEq(dq, wantDq, mism)
      check mism == 0

suite "ggml k-quants (Q4_K/Q6_K) dequant vs gguf-py":
  test "Q4_K super-block dequant, bit-exact":
    let raw = readBytes("ref_ggml_q4_k_bytes.bin")
    if raw.len == 0:
      skip()
    else:
      let want = readF32("ref_ggml_q4_k_dq.bin")
      var blocks = newSeq[BlockQ4_K](raw.len div sizeof(BlockQ4_K))
      copyMem(addr blocks[0], unsafeAddr raw[0], raw.len)
      var dq = newSeq[float32](blocks.len * QK_K)
      dequantizeQ4_K(blocks, dq)
      var mism = 0
      bitEq(dq, want, mism)
      check mism == 0

  test "Q6_K super-block dequant, bit-exact":
    let raw = readBytes("ref_ggml_q6_k_bytes.bin")
    if raw.len == 0:
      skip()
    else:
      let want = readF32("ref_ggml_q6_k_dq.bin")
      var blocks = newSeq[BlockQ6_K](raw.len div sizeof(BlockQ6_K))
      copyMem(addr blocks[0], unsafeAddr raw[0], raw.len)
      var dq = newSeq[float32](blocks.len * QK_K)
      dequantizeQ6_K(blocks, dq)
      var mism = 0
      bitEq(dq, want, mism)
      check mism == 0

template mxSchemeTest(
    TT: untyped, refFile: static string, emax: int, nm: static string
) =
  test nm:
    let x = readF32("ref_mx_input.bin")
    let want = readF32(refFile)
    if x.len == 0 or want.len == 0:
      skip()
    else:
      let p = calibrateMX(x, 32, emax)
      var q = newSeq[TT](x.len)
      quantize(x, p, q)
      var dq = newSeq[float32](x.len)
      dequantize(q, p, dq)
      var mism = 0
      bitEq(dq, want, mism)
      check mism == 0

suite "MX block scheme end-to-end vs OCP v1.0 + ml_dtypes":
  ## Proves calibrateMX (shared-scale selection incl. the E8M0 clamp and the
  ## zero-block rule) + generic quantize/dequantize, not just the element codecs.
  mxSchemeTest(F4E2M1, "ref_mx_f4e2m1_dq.bin", 2, "MXFP4 e2m1, block 32")
  mxSchemeTest(F6E2M3, "ref_mx_f6e2m3_dq.bin", 2, "MXFP6 e2m3, block 32")
  mxSchemeTest(F6E3M2, "ref_mx_f6e3m2_dq.bin", 4, "MXFP6 e3m2, block 32")

suite "NVFP4 vs TransformerEngine NVFP4QuantizerRef":
  test "codes + e4m3 block scales + global scale, bit-exact":
    let x = readF32("ref_nvfp4_input.bin")
    let golden = readBytes("ref_nvfp4_out.bin")
    if x.len == 0 or golden.len == 0:
      skip()
    else:
      let nb = x.len div nvfp4BlockLen
      var wantGds: float32
      copyMem(addr wantGds, unsafeAddr golden[0], 4)
      var codes = newSeq[F4E2M1](x.len)
      var scales = newSeq[F8E4M3](nb)
      let gds = quantizeNVFP4(x, codes, scales)
      check cast[uint32](gds) == cast[uint32](wantGds)
      var mism = 0
      for b in 0 ..< nb:
        if uint8(scales[b]) != uint8(golden[4 + b]):
          inc mism
      for i in 0 ..< x.len:
        if (bits(codes[i]) and 0x0f'u8) != uint8(golden[4 + nb + i]):
          inc mism
      check mism == 0

  test "round-trip through the documented dequant order":
    let x = readF32("ref_nvfp4_input.bin")
    if x.len == 0:
      skip()
    else:
      var codes = newSeq[F4E2M1](x.len)
      var scales = newSeq[F8E4M3](x.len div nvfp4BlockLen)
      let gds = quantizeNVFP4(x, codes, scales)
      var dq = newSeq[float32](x.len)
      dequantizeNVFP4(codes, scales, gds, dq)
      # sanity: fp4 error is relative to the BLOCK amax, not the element — an
      # element much smaller than its block's amax rounds to zero by design.
      # Worst-case e2m1 half-spacing is vmax/6, plus ~6% e4m3 scale rounding.
      var worst = 0.0
      for b in 0 ..< scales.len:
        # Blocks whose e4m3 scale underflowed to 0 are lost BY DESIGN, and
        # blocks with SUBNORMAL e4m3 scales carry up to ~50% scale-rounding
        # (the encode compensates, but the ±6 clamp then bites) — the golden
        # input spans 2^40 to exercise exactly these paths, and the oracle
        # test above proves we reproduce TE bit-for-bit on them. The tight
        # error bound only applies where the block scale is a NORMAL e4m3.
        if (uint8(scales[b]) and 0x78'u8) == 0'u8:
          continue
        var vmax = 0.0'f32
        for i in b * nvfp4BlockLen ..< (b + 1) * nvfp4BlockLen:
          vmax = max(vmax, abs(x[i]))
        if vmax == 0.0'f32:
          continue
        for i in b * nvfp4BlockLen ..< (b + 1) * nvfp4BlockLen:
          worst = max(worst, abs(dq[i].float64 - x[i].float64) / vmax.float64)
      check worst < 0.20

suite "MXINT8 (named recipe: calibrateMX + I8)":
  ## MXINT8 = int8 elements + one E8M0 power-of-two scale per 32-element block —
  ## per the 2025 INT-vs-FP study, the 8-bit accuracy/efficiency winner. It is
  ## expressible entirely with existing pieces: `calibrateMX(x, 32, elemEmax = 6)`
  ## (so a block's amax lands in the int8 grid's top octave, |q| in [64, 127])
  ## + generic quantize/dequantize over I8. One documented deviation from OCP
  ## spec-pure MXINT8: our `toI8` rounds ties AWAY (like ggml), not to even.
  test "scale placement and round-trip error bound":
    var x = newSeq[float32](32 * 64)
    var u = 31337'u32
    for i in 0 ..< x.len:
      u = u * 1664525'u32 + 1013904223'u32
      x[i] =
        (float32(u shr 8) / float32(1'u32 shl 24) - 0.5'f32) *
        pow(2.0'f32, float32(int(u shr 28) and 15) - 8.0'f32)
    let p = calibrateMX(x, 32, elemEmax = 6)
    var q = newSeq[I8](x.len)
    quantize(x, p, q)
    var dq = newSeq[float32](x.len)
    dequantize(q, p, dq)
    var ok = true
    for b in 0 ..< p.scale.len:
      var amax = 0.0'f32
      var qmax = 0
      for i in b * 32 ..< (b + 1) * 32:
        amax = max(amax, abs(x[i]))
        qmax = max(qmax, abs(int(int8(q[i]))))
        # per-element error ≤ half a step, except at the very top: amax/scale
        # lands in [64, 128), and ties-away can round 127.5+ up to 128, which
        # clamps to 127 — error up to (but below) one full step there.
        ok = ok and abs(dq[i] - x[i]) < p.scale[b]
      if amax > 0:
        ok = ok and qmax >= 64 and qmax <= 127 # amax in the top octave
    check ok
