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


if __name__ == "__main__":
    emit_bf16()
    emit_f16()
    emit_fp8("e4m3", ml_dtypes.float8_e4m3fn)
    emit_fp8("e5m2", ml_dtypes.float8_e5m2)
    emit_fp8("e4m3fnuz", ml_dtypes.float8_e4m3fnuz)
    emit_fp8("e5m2fnuz", ml_dtypes.float8_e5m2fnuz)
    print(f"wrote bf16 + f16 + fp8x4 reference vectors to {HERE}")
