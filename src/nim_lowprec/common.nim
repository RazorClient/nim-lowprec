## nim_lowprec/common — shared scaffolding for the low-precision dtype family.
##
## Every dtype (bf16, fp16, fp8, …) pivots through `float32` for arithmetic and
## shares one SIMD-dispatch story, so the pieces they have in common live here
## rather than being duplicated per module.
##
## TODO (later): LowPrec concept, fp32-pivot helpers, nimsimd cpuid/feature
## detection + scalar-reference-diffed-against-vector dispatch scaffold.
