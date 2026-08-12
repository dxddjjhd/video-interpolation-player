"""RIFE 适配层测试：用合成 ONNX 模型验证加载/IO 适配/预处理。

合成模型输出"两帧线性混合"，可精确验证：
- 模型加载与输入结构自动识别（concat / pair_ts）
- 32 对齐 pad/crop 正确性
- [0,1] 归一化与反归一化正确性
- timestep 语义正确性
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from player.interpolate.rife import RifeInterpolator  # noqa: E402

SYN_CONCAT = ROOT / "tests/data/synthetic_concat.onnx"
SYN_PAIR = ROOT / "tests/data/synthetic_pair_ts.onnx"


@pytest.fixture(scope="module")
def concat_eng():
    eng = RifeInterpolator(str(SYN_CONCAT), use_cuda=False)
    yield eng
    eng.close()


@pytest.fixture(scope="module")
def pair_eng():
    eng = RifeInterpolator(str(SYN_PAIR), use_cuda=False)
    yield eng
    eng.close()


def _frames(h=50, w=66):
    """尺寸故意不齐 32（50=16+18余, 66 非 32 倍数），验证 pad/crop。"""
    f0 = np.zeros((h, w, 3), dtype=np.uint8)
    f1 = np.full((h, w, 3), 200, dtype=np.uint8)
    return f0, f1


def test_load_and_shape(concat_eng, pair_eng):
    assert concat_eng._io["mode"] == "concat"
    assert concat_eng._io["has_ts"] is False
    assert pair_eng._io["mode"] == "pair_ts"
    assert pair_eng._io["has_ts"] is True


def test_concat_mean_value(concat_eng):
    """合成模型: 输出应精确等于两帧平均。"""
    f0, f1 = _frames()
    mids = concat_eng.interpolate(f0, f1, [0.5])
    assert len(mids) == 1
    assert mids[0].shape == f0.shape
    assert mids[0].dtype == np.uint8
    expected = ((f0.astype(np.int16) + f1.astype(np.int16)) / 2).astype(np.uint8)
    assert np.array_equal(mids[0], expected), "均值插值结果不符"


def test_pair_ts_blend_value(pair_eng):
    """三输入模型: 输出应等于 (1-t)*f0 + t*f1。"""
    f0, f1 = _frames()
    mids = pair_eng.interpolate(f0, f1, [0.25, 0.75])
    assert len(mids) == 2
    for t, m in zip([0.25, 0.75], mids):
        expected = ((1 - t) * f0 + t * f1).astype(np.uint8)
        assert np.abs(m.astype(int) - expected.astype(int)).max() <= 1


def test_roundtrip_concat_ts_unused(pair_eng):
    """验证 has_ts 路径: timestep 参与线性混合。"""
    f0, f1 = _frames()
    mids = pair_eng.interpolate(f0, f1, [0.0 + 1e-9])  # 几乎等于 f0
    assert np.abs(mids[0].astype(int) - f0.astype(int)).max() <= 1
    mids = pair_eng.interpolate(f0, f1, [1.0 - 1e-9])  # 几乎等于 f1
    assert np.abs(mids[0].astype(int) - f1.astype(int)).max() <= 1
