## nim_lowprec/simd/x86 — shared immintrin.h intrinsic bindings (SSE2..AVX2/FMA/F16C).
##
## The x86 counterpart of simd/neon.nim: thin `importc` wrappers over the Intel
## intrinsics used by the SIMD kernels, guarded by `when defined(amd64) or
## defined(i386)` so this module is empty on any non-x86 target. Vector types are
## `{.bycopy.}` objects mirroring the __m128/__m128i/__m256/__m256i registers.
##
## Shifts use the variable-count forms (`_mm_srl_epi32`, `_mm256_sll_epi32`, …)
## — the count travels in the low 64 bits of an __m128i — rather than the
## immediate `_mm_slli_*`, mirroring neon.nim's deliberate variable-count `vshlq_*`:
## it keeps the amount a runtime value and sidesteps the compile-time-constant
## requirement. The genuinely immediate args (`_mm256_cvtps_ph` rounding mode,
## `_mm256_extractf128_ps` lane) are always passed as literals.

when defined(amd64) or defined(i386):
  {.push header: "immintrin.h".}

  type
    m128i* {.importc: "__m128i", bycopy.} = object
    m256i* {.importc: "__m256i", bycopy.} = object
    m256*  {.importc: "__m256",  bycopy.} = object
    m128*  {.importc: "__m128",  bycopy.} = object

  # ---- loads / stores ----
  proc mm_loadu_si128*(p: ptr m128i): m128i {.importc: "_mm_loadu_si128".}
  proc mm_storeu_si128*(p: ptr m128i; a: m128i) {.importc: "_mm_storeu_si128".}
  proc mm_storel_epi64*(p: ptr m128i; a: m128i) {.importc: "_mm_storel_epi64".}
  proc mm_loadl_epi64*(p: ptr m128i): m128i {.importc: "_mm_loadl_epi64".}   # 8 bytes → low 64
  proc mm_loadu_ps*(p: ptr float32): m128 {.importc: "_mm_loadu_ps".}
  proc mm256_loadu_ps*(p: ptr float32): m256 {.importc: "_mm256_loadu_ps".}
  proc mm256_storeu_ps*(p: ptr float32; a: m256) {.importc: "_mm256_storeu_ps".}

  # ---- set / broadcast ----
  proc mm_cvtsi32_si128*(a: cint): m128i {.importc: "_mm_cvtsi32_si128".}
  proc mm_set1_epi32*(a: cint): m128i {.importc: "_mm_set1_epi32".}
  proc mm_set1_epi8*(a: cint): m128i {.importc: "_mm_set1_epi8".}
  proc mm_set1_epi16*(a: cint): m128i {.importc: "_mm_set1_epi16".}
  proc mm256_set1_ps*(a: cfloat): m256 {.importc: "_mm256_set1_ps".}
  proc mm256_setzero_ps*(): m256 {.importc: "_mm256_setzero_ps".}

  # ---- F16C convert (fp16 <-> fp32) ----
  proc mm256_cvtph_ps*(a: m128i): m256 {.importc: "_mm256_cvtph_ps".}         # 8×fp16 -> 8×fp32
  proc mm256_cvtps_ph*(a: m256; rounding: cint): m128i {.importc: "_mm256_cvtps_ph".}  # 8×fp32 -> 8×fp16

  # ---- widen / float convert ----
  proc mm256_cvtepu16_epi32*(a: m128i): m256i {.importc: "_mm256_cvtepu16_epi32".}  # 8×u16 -> 8×u32 (zero-ext)
  proc mm256_cvtepi8_epi32*(a: m128i): m256i {.importc: "_mm256_cvtepi8_epi32".}    # 8×i8  -> 8×i32 (sign-ext)
  proc mm256_cvtepi32_ps*(a: m256i): m256 {.importc: "_mm256_cvtepi32_ps".}
  proc mm_cvtepu8_epi16*(a: m128i): m128i {.importc: "_mm_cvtepu8_epi16".}   # 8×u8 -> 8×u16 (zero-ext)

  # ---- arithmetic ----
  proc mm256_mul_ps*(a, b: m256): m256 {.importc: "_mm256_mul_ps".}
  proc mm256_fmadd_ps*(a, b, c: m256): m256 {.importc: "_mm256_fmadd_ps".}     # a*b + c  (FMA)
  proc mm_add_epi32*(a, b: m128i): m128i {.importc: "_mm_add_epi32".}
  proc mm_sub_epi8*(a, b: m128i): m128i {.importc: "_mm_sub_epi8".}
  proc mm_add_ps*(a, b: m128): m128 {.importc: "_mm_add_ps".}
  proc mm_hadd_ps*(a, b: m128): m128 {.importc: "_mm_hadd_ps".}

  # ---- bitwise / variable-count shift / compare / select ----
  proc mm_and_si128*(a, b: m128i): m128i {.importc: "_mm_and_si128".}
  proc mm_or_si128*(a, b: m128i): m128i {.importc: "_mm_or_si128".}
  proc mm_xor_si128*(a, b: m128i): m128i {.importc: "_mm_xor_si128".}
  proc mm_cmpgt_epi32*(a, b: m128i): m128i {.importc: "_mm_cmpgt_epi32".}
  proc mm_cmpeq_epi16*(a, b: m128i): m128i {.importc: "_mm_cmpeq_epi16".}
  proc mm_blendv_epi8*(a, b, mask: m128i): m128i {.importc: "_mm_blendv_epi8".}
  proc mm_srl_epi32*(a, count: m128i): m128i {.importc: "_mm_srl_epi32".}
  proc mm_srl_epi16*(a, count: m128i): m128i {.importc: "_mm_srl_epi16".}
  proc mm_sll_epi16*(a, count: m128i): m128i {.importc: "_mm_sll_epi16".}
  proc mm256_sll_epi32*(a: m256i; count: m128i): m256i {.importc: "_mm256_sll_epi32".}

  # ---- shuffle / interleave / cast / extract ----
  proc mm_shuffle_epi8*(a, b: m128i): m128i {.importc: "_mm_shuffle_epi8".}
  proc mm_unpacklo_epi8*(a, b: m128i): m128i {.importc: "_mm_unpacklo_epi8".}
  proc mm_unpackhi_epi8*(a, b: m128i): m128i {.importc: "_mm_unpackhi_epi8".}
  proc mm_unpackhi_epi64*(a, b: m128i): m128i {.importc: "_mm_unpackhi_epi64".}
  proc mm_castps_si128*(a: m128): m128i {.importc: "_mm_castps_si128".}
  proc mm256_castsi256_ps*(a: m256i): m256 {.importc: "_mm256_castsi256_ps".}
  proc mm256_castps256_ps128*(a: m256): m128 {.importc: "_mm256_castps256_ps128".}
  proc mm256_extractf128_ps*(a: m256; imm: cint): m128 {.importc: "_mm256_extractf128_ps".}
  proc mm_cvtss_f32*(a: m128): cfloat {.importc: "_mm_cvtss_f32".}

  {.pop.}

  # Horizontal sum of an 8-lane accumulator → scalar (used by the fused GEMVs).
  proc hsum256*(v: m256): float32 {.inline.} =
    let lo = mm256_castps256_ps128(v)
    let hi = mm256_extractf128_ps(v, 1)
    var s = mm_add_ps(lo, hi)          # 4 partial sums
    s = mm_hadd_ps(s, s)               # 2
    s = mm_hadd_ps(s, s)               # 1
    float32(mm_cvtss_f32(s))
