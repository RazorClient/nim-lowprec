func pow2*(e: int): float64 {.inline.} =
  # Exact 2^e for e in the double normal range.
  cast[float64](uint64(e + 1023) shl 52)

func tinyMag*(c, ebits, mbits, bias: int): float64 =
  let ef = (c shr mbits) and ((1 shl ebits) - 1)
  let mf = c and ((1 shl mbits) - 1)
  if ef == 0:
    float64(mf) * pow2(1 - bias - mbits) # subnormal (0 if mf==0)
  else:
    float64((1 shl mbits) + mf) * pow2(ef - bias - mbits) # normal

func nearestCode*(a: float64, ebits, mbits, bias, cmax: int): int =
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
    (if (cLo and 1) == 0: cLo else: cHi)
