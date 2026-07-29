#!/usr/bin/env python3
"""Generate reference vectors for nim-lowprec conformance tests.
Requires: numpy, ml_dtypes   ->   pip install numpy ml_dtypes
ml_dtypes is the reference JAX/TF/OpenXLA treats as ground truth for bf16.
"""
import os
import numpy as np
import ml_dtypes

HERE = os.path.dirname(os.path.abspath(__file__))


def emit_bf16():
    # 1) bf16 -> f32 : all 65536 bf16 bit patterns, exhaustive.
    codes = np.arange(65536, dtype=np.uint16)
    f32 = codes.view(ml_dtypes.bfloat16).astype(np.float32)
    f32.view("<u4").tofile(os.path.join(HERE, "ref_bf16_to_f32.bin"))

    # 2) f32 -> bf16 : structured edges + a large random sample.
    edges = np.array([
        0x00000000, 0x80000000,              # +0, -0
        0x3F800000, 0xBF800000,              # +1, -1
        0x7F800000, 0xFF800000,              # +Inf, -Inf
        0x7F800001, 0x7FC00000, 0xFFC00000,  # NaNs, incl. the truncation trap
        0x00000001, 0x007FFFFF,              # smallest / largest fp32 denormals
        0x3F808000, 0x3F818000,              # exact RNE ties (even both ways)
        0x7F7FFFFF,                          # max finite fp32 (rounds toward Inf)
    ], dtype=np.uint32)
    rng = np.random.default_rng(0)
    rand = rng.integers(0, 1 << 32, size=4_000_000, dtype=np.uint64).astype(np.uint32)
    ins = np.unique(np.concatenate([edges, rand])).astype(np.uint32)
    outs = ins.view(np.float32).astype(ml_dtypes.bfloat16).view(np.uint16)
    with open(os.path.join(HERE, "ref_f32_to_bf16.bin"), "wb") as fh:
        fh.write(np.uint32(len(ins)).tobytes())
        fh.write(ins.astype("<u4").tobytes())
        fh.write(outs.astype("<u2").tobytes())


def emit_f16():
    # 1) f16 -> f32 : all 65536 IEEE half patterns, exhaustive.
    codes = np.arange(65536, dtype=np.uint16)
    f32 = codes.view(np.float16).astype(np.float32)
    f32.view("<u4").tofile(os.path.join(HERE, "ref_f16_to_f32.bin"))

    # 2) f32 -> f16 : fp16-stressing edges + a large random sample.
    edges = np.array([
        0x00000000, 0x80000000,              # +0, -0
        0x3F800000, 0xBF800000,              # +1, -1
        0x7F800000, 0xFF800000,              # +Inf, -Inf
        0x7F800001, 0x7FC00000,              # NaNs
        0x477FE000,                          # 65504 = max fp16 normal
        0x47800000,                          # 65536 -> overflow to Inf
        0x38800000,                          # 2^-14 = min fp16 normal
        0x33800000,                          # 2^-24 = min fp16 subnormal
        0x33000000,                          # 2^-25 = round-to-zero threshold
        0x38000000,                          # 2^-15
    ], dtype=np.uint32)
    rng = np.random.default_rng(1)
    rand = rng.integers(0, 1 << 32, size=4_000_000, dtype=np.uint64).astype(np.uint32)
    ins = np.unique(np.concatenate([edges, rand])).astype(np.uint32)
    with np.errstate(over="ignore", invalid="ignore"):
        outs = ins.view(np.float32).astype(np.float16).view(np.uint16)
    with open(os.path.join(HERE, "ref_f32_to_f16.bin"), "wb") as fh:
        fh.write(np.uint32(len(ins)).tobytes())
        fh.write(ins.astype("<u4").tobytes())
        fh.write(outs.astype("<u2").tobytes())


def emit_fp8(name, dt):
    # 1) fp8 -> f32 : all 256 codes, exhaustive.
    codes = np.arange(256, dtype=np.uint8)
    with np.errstate(over="ignore", invalid="ignore"):
        f32 = codes.view(dt).astype(np.float32)
    f32.view("<u4").tofile(os.path.join(HERE, f"ref_{name}_to_f32.bin"))

    # 2) f32 -> fp8 : edges (near each format's max) + random sample.
    edges = np.array([
        0x00000000, 0x80000000, 0x3F800000, 0xBF800000,   # ±0, ±1
        0x7F800000, 0xFF800000, 0x7FC00000,               # ±Inf, NaN
        0x43600000, 0x43E00000,                           # 224, 448 (e4m3fn max)
        0x43700000, 0x47600000,                           # 240 (fnuz), 57344 (e5m2 max)
        0x38800000, 0x33800000, 0x3B800000,               # small / subnormal-ish
    ], dtype=np.uint32)
    rng = np.random.default_rng(7)
    rand = rng.integers(0, 1 << 32, size=2_000_000, dtype=np.uint64).astype(np.uint32)
    ins = np.unique(np.concatenate([edges, rand])).astype(np.uint32)
    with np.errstate(over="ignore", invalid="ignore"):
        outs = ins.view(np.float32).astype(dt).view(np.uint8)
    with open(os.path.join(HERE, f"ref_f32_to_{name}.bin"), "wb") as fh:
        fh.write(np.uint32(len(ins)).tobytes())
        fh.write(ins.astype("<u4").tobytes())
        fh.write(outs.astype("<u1").tobytes())


def emit_mx_float(name, dt, ncodes, seed):
    # 1) decode : all codes, exhaustive (tiny code spaces).
    codes = np.arange(ncodes, dtype=np.uint8)
    with np.errstate(over="ignore", invalid="ignore"):
        f32 = codes.view(dt).astype(np.float32)
    f32.view("<u4").tofile(os.path.join(HERE, f"ref_{name}_to_f32.bin"))

    # 2) encode : finite edges + random. These formats have NO NaN, so NaN
    # *inputs* are undefined — the Nim test skips them; inf saturates to max.
    edge_vals = np.array([0.0, -0.0, 0.25, 0.5, 1.0, -1.0, 1.5, -1.5, 2.0, 3.0,
                          4.0, 6.0, 7.0, 7.5, 8.0, 28.0, 100.0, -100.0,
                          np.inf, -np.inf], dtype=np.float32)
    edges = edge_vals.view(np.uint32)
    rng = np.random.default_rng(seed)
    rand = rng.integers(0, 1 << 32, size=2_000_000, dtype=np.uint64).astype(np.uint32)
    ins = np.unique(np.concatenate([edges, rand])).astype(np.uint32)
    with np.errstate(over="ignore", invalid="ignore"):
        outs = ins.view(np.float32).astype(dt).view(np.uint8)
    with open(os.path.join(HERE, f"ref_f32_to_{name}.bin"), "wb") as fh:
        fh.write(np.uint32(len(ins)).tobytes())
        fh.write(ins.astype("<u4").tobytes())
        fh.write(outs.astype("<u1").tobytes())


def emit_e8m0():
    # 1) decode : all 256 codes (0x00 = 2^-127, 0xFF = NaN).
    codes = np.arange(256, dtype=np.uint8)
    with np.errstate(over="ignore", invalid="ignore"):
        f32 = codes.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
    f32.view("<u4").tofile(os.path.join(HERE, "ref_e8m0_to_f32.bin"))

    # 2) encode : NORMAL positive floats only (+ specials). E8M0 is a scale, so
    # its domain is normal positive floats; ml_dtypes rounds float32-SUBNORMAL
    # inputs (~1e-39) UP in a way our true-round-to-nearest encode deliberately
    # doesn't — an out-of-domain quirk we don't replicate. 0/inf/nan → 0xFF in
    # both, so a direct bit-compare works.
    edge_vals = np.array([1.0, 1.4, 1.5, 1.9, 2.0, 2.9, 3.0, 0.4, 0.75, 6.0,
                          2.0**-126, 2.0**126, 2.0**127, 0.0, np.inf, np.nan],
                         dtype=np.float32)
    edges = edge_vals.view(np.uint32)
    rng = np.random.default_rng(23)
    rand = rng.integers(0x00800000, 0x7f800000, size=2_000_000, dtype=np.uint64).astype(np.uint32)  # normal +
    ins = np.unique(np.concatenate([edges, rand])).astype(np.uint32)
    with np.errstate(over="ignore", invalid="ignore"):
        outs = ins.view(np.float32).astype(ml_dtypes.float8_e8m0fnu).view(np.uint8)
    with open(os.path.join(HERE, "ref_f32_to_e8m0.bin"), "wb") as fh:
        fh.write(np.uint32(len(ins)).tobytes())
        fh.write(ins.astype("<u4").tobytes())
        fh.write(outs.astype("<u1").tobytes())


# ======================= BLOCK-SCHEME golden vectors =======================
# Everything above validates ELEMENT codecs. The emitters below validate the
# block SCHEMES end-to-end (CONFORMANCE.md P0/P1): shared-scale selection,
# layout, and the exact fp op order of (de)quantization.

def emit_ggml_schemes():
    """External oracle: gguf-py (`pip install gguf`) — llama.cpp's own numpy
    implementation, documented bit-exact vs ggml-quants.c. Replaces the old
    round-trip self-test with a real second implementation. Skipped with a
    warning when gguf is not installed; the Nim tests skip when files are absent."""
    try:
        import gguf
        from gguf import GGMLQuantizationType as T, quants
    except ImportError:
        print("  !! gguf not installed -> skipping ggml scheme goldens (pip install gguf)")
        return
    rng = np.random.default_rng(1234)
    n = 256 * 64                                # 64 super-blocks = 512 Q*_0 blocks
    x = (rng.standard_normal(n) * 4.0).astype(np.float32)
    x[100:132] = 0.0                            # an all-zero Q*_0 block
    x[4096:4104] = np.float32(1e-30)            # a denormal-scale block
    x.view("<u4").tofile(os.path.join(HERE, "ref_scheme_input.bin"))
    # Q8_0 / Q4_0: gguf-py implements the QUANTIZE direction (bit-exact vs
    # ggml-quants.c) — golden covers quantize bytes AND their dequant.
    for name, t in [("q8_0", T.Q8_0), ("q4_0", T.Q4_0)]:
        b = quants.quantize(x, t)
        d = quants.dequantize(b, t).astype(np.float32)
        b.tofile(os.path.join(HERE, f"ref_ggml_{name}_bytes.bin"))
        d.view("<u4").tofile(os.path.join(HERE, f"ref_ggml_{name}_dq.bin"))
    # Q4_K / Q6_K: gguf-py only implements DEQUANTIZE for the k-quants — which
    # is the ingestion direction. Golden = synthetic (valid) random blocks +
    # gguf-py's dequantization of them. d/dmin are kept finite so the fp32
    # comparison can be bit-exact (NaN payloads would differ meaninglessly).
    rng2 = np.random.default_rng(5678)
    def finite_f16(count):
        v = (rng2.standard_normal(count) * 0.1).astype(np.float16)
        return v.view(np.uint8).reshape(count, 2)
    NB = 128
    q4k = np.zeros((NB, 144), dtype=np.uint8)
    q4k[:, 0:2]   = finite_f16(NB)                                   # d
    q4k[:, 2:4]   = finite_f16(NB)                                   # dmin
    q4k[:, 4:144] = rng2.integers(0, 256, size=(NB, 140), dtype=np.uint16).astype(np.uint8)
    q6k = np.zeros((NB, 210), dtype=np.uint8)
    q6k[:, 0:208]   = rng2.integers(0, 256, size=(NB, 208), dtype=np.uint16).astype(np.uint8)
    q6k[:, 208:210] = finite_f16(NB)                                 # d (at the END in q6_K)
    for name, t, blocks in [("q4_k", T.Q4_K, q4k), ("q6_k", T.Q6_K, q6k)]:
        raw = blocks.reshape(-1)
        d = quants.dequantize(raw, t).astype(np.float32)
        raw.tofile(os.path.join(HERE, f"ref_ggml_{name}_bytes.bin"))
        d.view("<u4").tofile(os.path.join(HERE, f"ref_ggml_{name}_dq.bin"))
    print("  ggml scheme goldens written (gguf-py oracle)")


def emit_mx_schemes():
    """OCP MX v1.0 block scheme, implemented independently here in numpy with
    ml_dtypes element rounding: shared scale = 2^(clamp(floor(log2(blockAmax))
    - elemEmax, -127, 127)) (amax==0 -> scale 1), elements = (x/scale) rounded
    by ml_dtypes, fake-quant out = element * scale. This is what validates
    `calibrateMX` + generic quantize/dequantize END TO END — the element goldens
    above cannot see the scale selection. (microxcaling implements the same
    spec; it is torch-based and not on PyPI, hence this transcription — see
    CONFORMANCE.md.)"""
    formats = [("f4e2m1", ml_dtypes.float4_e2m1fn, 2),
               ("f6e2m3", ml_dtypes.float6_e2m3fn, 2),
               ("f6e3m2", ml_dtypes.float6_e3m2fn, 4)]
    BS = 32
    rng = np.random.default_rng(777)
    nblocks = 512
    n = BS * nblocks
    x = (rng.standard_normal(n) * 2.0).astype(np.float32)
    # magnitude sweep across blocks, including E8M0-clamp extremes and denormals
    mags = np.float32(2.0) ** rng.integers(-140, 120, size=nblocks).astype(np.float32)
    x = (x.reshape(nblocks, BS) * mags[:, None]).astype(np.float32)
    x[3] = 0.0                                  # all-zero block
    x[4, :16] = 0.0                             # half-zero block
    x = x.reshape(-1)
    x.view("<u4").tofile(os.path.join(HERE, "ref_mx_input.bin"))
    xb = x.reshape(nblocks, BS)
    amax = np.abs(xb).max(axis=1)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(amax.astype(np.float64)))   # exact for powers of two
    for name, dt, emax in formats:
        k = np.clip(e - emax, -127, 127)
        scale = np.where(amax == 0, np.float32(1.0),
                         np.exp2(k).astype(np.float32))[:, None].astype(np.float32)
        q = (xb / scale).astype(dt)             # ml_dtypes: RNE, saturating
        dq = (q.astype(np.float32) * scale).astype(np.float32)
        dq.reshape(-1).view("<u4").tofile(os.path.join(HERE, f"ref_mx_{name}_dq.bin"))
    print("  MX scheme goldens written (OCP v1.0 + ml_dtypes elements)")


def emit_nvfp4():
    """NVFP4 golden — numpy transcription of TransformerEngine's
    NVFP4QuantizerRef._quantize_blockwise_reference (1D path, pow_2_scales=False,
    no 4over6), transformer_engine/pytorch/custom_recipes/quantization_ref_nvfp4.py.
    Block 16, per-block FP8-E4M3 decode scale, per-tensor fp32 scale. The e4m3
    cast uses ml_dtypes (RNE, same as torch's cast after the clamp)."""
    F32MAX = np.float32(np.finfo(np.float32).max)
    rng = np.random.default_rng(4242)
    nblocks = 1024
    n = 16 * nblocks
    x = (rng.standard_normal(n) * 3.0).astype(np.float32)
    mags = np.float32(2.0) ** rng.integers(-20, 20, size=nblocks).astype(np.float32)
    x = (x.reshape(nblocks, 16) * mags[:, None]).astype(np.float32)
    x[7] = 0.0                                  # zero block inside a nonzero tensor
    xf = x.reshape(-1)
    xf.view("<u4").tofile(os.path.join(HERE, "ref_nvfp4_input.bin"))

    gamax = np.float32(np.abs(xf).max())
    ges = np.minimum(np.float32(2688.0) / gamax, F32MAX)      # 448 * 6
    if ges == np.float32(0.0):
        ges = np.float32(1.0)
    gds = np.float32(1.0) / ges
    mult = ges * np.float32(np.float32(1.0) / np.float32(6.0))  # reciprocal(6) then mul

    vmax = np.abs(x).max(axis=1).astype(np.float32)
    ds = np.minimum(vmax * mult, F32MAX)
    ds = np.clip(ds, -np.float32(448.0), np.float32(448.0))
    ds8 = ds.astype(ml_dtypes.float8_e4m3fn)                  # per-block decode scale
    with np.errstate(divide="ignore"):
        es = np.minimum(np.float32(1.0) / (ds8.astype(np.float32) * gds), F32MAX)
    scaled = (x * es[:, None].astype(np.float32)).astype(np.float32)
    clipped = np.clip(scaled, -6.0, 6.0).astype(np.float32)

    # TE's cast_to_fp4x2 boundary table (== RNE on the e2m1 grid, -0 -> code 0)
    c = clipped
    codes = np.zeros(c.shape, dtype=np.uint8)
    codes[(c >= 0.0) & (c <= 0.25)] = 0
    codes[(c > 0.25) & (c < 0.75)] = 1
    codes[(c >= 0.75) & (c <= 1.25)] = 2
    codes[(c > 1.25) & (c < 1.75)] = 3
    codes[(c >= 1.75) & (c <= 2.5)] = 4
    codes[(c > 2.5) & (c < 3.5)] = 5
    codes[(c >= 3.5) & (c <= 5.0)] = 6
    codes[c > 5.0] = 7
    codes[(c >= -0.25) & (c < -0.0)] = 8
    codes[(c < -0.25) & (c > -0.75)] = 9
    codes[(c <= -0.75) & (c >= -1.25)] = 10
    codes[(c < -1.25) & (c > -1.75)] = 11
    codes[(c <= -1.75) & (c >= -2.5)] = 12
    codes[(c < -2.5) & (c > -3.5)] = 13
    codes[(c <= -3.5) & (c >= -5.0)] = 14
    codes[c < -5.0] = 15

    with open(os.path.join(HERE, "ref_nvfp4_out.bin"), "wb") as fh:
        fh.write(np.float32(gds).tobytes())                   # 4 B global decode scale
        fh.write(ds8.view(np.uint8).tobytes())                # nblocks e4m3 bytes
        fh.write(codes.reshape(-1).astype("<u1").tobytes())   # n fp4 codes (unpacked)
    print("  NVFP4 goldens written (TE NVFP4QuantizerRef transcription)")


if __name__ == "__main__":
    emit_bf16()
    emit_f16()
    emit_fp8("e4m3", ml_dtypes.float8_e4m3fn)
    emit_fp8("e5m2", ml_dtypes.float8_e5m2)
    emit_fp8("e4m3fnuz", ml_dtypes.float8_e4m3fnuz)
    emit_fp8("e5m2fnuz", ml_dtypes.float8_e5m2fnuz)
    emit_mx_float("f4e2m1", ml_dtypes.float4_e2m1fn, 16, 41)
    emit_mx_float("f6e2m3", ml_dtypes.float6_e2m3fn, 64, 42)
    emit_mx_float("f6e3m2", ml_dtypes.float6_e3m2fn, 64, 43)
    emit_e8m0()
    emit_ggml_schemes()
    emit_mx_schemes()
    emit_nvfp4()
    print(f"wrote element + block-scheme reference vectors to {HERE}")
