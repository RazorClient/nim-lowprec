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
  lpUseNeon*   = (lpSimd == "auto" and (defined(arm64) or defined(aarch64))) or
                 lpSimd == "neon"
  lpUseAvx2*   = lpSimd == "avx2"
  lpUseScalar* = not (lpUseNeon or lpUseAvx2)

const simdBackend* =
  when lpUseNeon: "neon"
  elif lpUseAvx2: "avx2"
  else:           "scalar"

# An unknown value is a build error, not a silent fall-through to scalar.
when lpSimd notin ["auto", "scalar", "neon", "avx2"]:
  {.error: "unknown -d:lpSimd=" & lpSimd & "  (valid: auto | scalar | neon | avx2)".}

# ...and so is asking for an ISA the target CPU can't host.
when lpUseNeon and not (defined(arm64) or defined(aarch64)):
  {.error: "-d:lpSimd=neon requires an arm64/aarch64 target".}
when lpUseAvx2 and not (defined(amd64) or defined(i386)):
  {.error: "-d:lpSimd=avx2 requires an x86 target".}
