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
    print(f"wrote bf16 + f16 + fp8x4 + mxfp4/6 + e8m0 reference vectors to {HERE}")
