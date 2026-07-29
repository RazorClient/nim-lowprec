## nim_lowprec/simd/target — compile-time SIMD ISA selection.
##
## The active kernel path is chosen ONCE, at compile time, from the `-d:lpSimd=`
## define (default "auto" → NEON on arm64, scalar elsewhere). There is NO runtime
## dispatch yet — a build compiles exactly one path; a cpuid layer (single fat
## binary, pick at runtime) arrives later, as do the SSE2 / AVX-512 tiers.
##
##   auto    NEON on arm64/aarch64, else scalar          (the default)
##   scalar  portable fallback, no intrinsics
##   neon    ARM NEON               (arm_neon.h)
##   avx2    x86 AVX2 + FMA + F16C  (immintrin.h)

const lpSimd* {.strdefine.}: string = "auto"

const
  lpUseNeon* =
    (lpSimd == "auto" and (defined(arm64) or defined(aarch64))) or lpSimd == "neon"
  lpUseAvx2* = lpSimd == "avx2"
  lpUseScalar* = not (lpUseNeon or lpUseAvx2)

const simdBackend* =
  when lpUseNeon:
    "neon"
  elif lpUseAvx2:
    "avx2"
  else:
    "scalar"

# ---- ARM dot-product (SDOT / FEAT_DotProd, ARMv8.2) ----
#
# `vdotq_s32` does 16 int8 multiply-accumulates into int32 in ONE instruction —
# the int8×int8 GEMV path lives or dies on it. It is NOT in baseline ARMv8.0, so
# it needs its own gate, separate from `lpSimd`:
#
#   auto  on when the target is Apple Silicon (M1 and later all have FEAT_DotProd
#         and Apple clang enables it by default) — otherwise off
#   on    force it: for Linux/arm64 builds that pass a -mcpu with dotprod, e.g.
#         `--passC:-march=armv8.2-a+dotprod`. Building this without the feature is
#         a C compile error, not a runtime surprise.
#   off   never; the int8×int8 kernels use their exact integer fallback instead
#         (same result, bit-for-bit — the fallback accumulates in int32 too).
const lpDotProd* {.strdefine.}: string = "auto"

const lpUseDotProd* =
  lpUseNeon and ((lpDotProd == "auto" and defined(macosx)) or lpDotProd == "on")

# An unknown value is a build error, not a silent fall-through to scalar.
when lpSimd notin ["auto", "scalar", "neon", "avx2"]:
  {.error: "unknown -d:lpSimd=" & lpSimd & "  (valid: auto | scalar | neon | avx2)".}
when lpDotProd notin ["auto", "on", "off"]:
  {.error: "unknown -d:lpDotProd=" & lpDotProd & "  (valid: auto | on | off)".}
when lpDotProd == "on" and not lpUseNeon:
  {.error: "-d:lpDotProd=on requires the NEON path (-d:lpSimd=neon or auto on arm64)".}

# ...and so is asking for an ISA the target CPU can't host.
when lpUseNeon and not (defined(arm64) or defined(aarch64)):
  {.error: "-d:lpSimd=neon requires an arm64/aarch64 target".}
when lpUseAvx2 and not (defined(amd64) or defined(i386)):
  {.error: "-d:lpSimd=avx2 requires an x86 target".}
