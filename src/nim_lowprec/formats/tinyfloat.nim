## nim_lowprec/tinyfloat — shared numeric core for the sub-16-bit float formats.
##
## The bit-layout math common to float8 (fp8×4) and mxfloat (MXFP4/MXFP6): the
## exact power-of-two, the magnitude of a finite code, and the round-to-nearest-
## even search for the code nearest a target magnitude. This is the SIGN-FREE,
## SPECIAL-FREE core — each format module layers its own sign / zero / NaN / Inf /
## saturation edges on top. Bit-exactness (vs ml_dtypes) is enforced by the
## per-format conformance suites.

func pow2*(e: int): float64 {.inline.} =
  ## Exact 2^e for e in the double normal range (format exponents are tiny).
  cast[float64](uint64(e + 1023) shl 52)

func tinyMag*(c, ebits, mbits, bias: int): float64 =
  ## Positive magnitude of finite code `c` (monotone increasing in c) for a
  ## sign-magnitude float of `ebits` exponent + `mbits` mantissa bits.
  let ef = (c shr mbits) and ((1 shl ebits) - 1)
  let mf = c and ((1 shl mbits) - 1)
  if ef == 0:
    float64(mf) * pow2(1 - bias - mbits) # subnormal (0 if mf==0)
  else:
    float64((1 shl mbits) + mf) * pow2(ef - bias - mbits) # normal

func nearestCode*(a: float64, ebits, mbits, bias, cmax: int): int =
  ## Code in [0, cmax] whose magnitude is nearest `a` (with 0 ≤ a < mag(cmax)),
  ## round-to-nearest-even. The caller applies the sign and handles the zero
  ## code plus the overflow / NaN / Inf / saturation edges.
  var lo = 0
  var hi = cmax
  while lo < hi:
    let m = (lo + hi) div 2
    if tinyMag(m, ebits, mbits, bias) >= a:
      hi = m
    else:
      lo = m + 1
  let cHi = hi
  if cHi == 0:
    return 0
  let cLo = cHi - 1
  let midv = (tinyMag(cLo, ebits, mbits, bias) + tinyMag(cHi, ebits, mbits, bias)) * 0.5
  if a < midv:
    cLo
  elif a > midv:
    cHi
  else:
    (
      if (cLo and 1) == 0: cLo else: cHi # tie → even
    )
