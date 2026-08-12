"""插帧引擎单元测试：输出帧数、尺寸、内容合理性。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from player.interpolate.optical_flow import OpticalFlowInterpolator  # noqa: E402


def _moving_pair(h=120, w=160):
    """两帧：白色方块从左侧移到右侧。"""
    f0 = np.zeros((h, w, 3), dtype=np.uint8)
    f1 = np.zeros((h, w, 3), dtype=np.uint8)
    f0[:, 20:40, :] = 255
    f1[:, 100:120, :] = 255
    return f0, f1


def test_optical_flow_output_shape():
    eng = OpticalFlowInterpolator()
    f0, f1 = _moving_pair()
    mids = eng.interpolate(f0, f1, [0.5])
    assert len(mids) == 1
    assert mids[0].shape == f0.shape
    assert mids[0].dtype == np.uint8
    # 中间帧应包含运动物体的过渡位置（方块两侧都有部分白色）
    row = mids[0][60]
    white = np.where(row.mean(axis=1) > 100)[0]
    assert len(white) > 0, "中间帧没有过渡内容"


def test_optical_flow_multiple_times():
    eng = OpticalFlowInterpolator()
    f0, f1 = _moving_pair()
    mids = eng.interpolate(f0, f1, [0.25, 0.5, 0.75])
    assert len(mids) == 3
    # t 递增时，白色区域质心应单调移动
    coms = []
    for m in mids:
        row = m[60].mean(axis=1)
        white = np.where(row > 100)[0]
        coms.append(white.mean() if len(white) else -1)
    assert coms[0] < coms[1] < coms[2], f"质心未单调: {coms}"


def test_optical_flow_empty_times():
    eng = OpticalFlowInterpolator()
    f0, f1 = _moving_pair()
    assert eng.interpolate(f0, f1, []) == []


def test_optical_flow_identity():
    """相同帧插帧应输出原帧。"""
    eng = OpticalFlowInterpolator()
    f = np.full((60, 80, 3), 128, dtype=np.uint8)
    mids = eng.interpolate(f, f, [0.5])
    assert np.abs(mids[0].astype(int) - f.astype(int)).mean() < 2


@pytest.mark.skipif(not (ROOT / "player/interpolate/models").exists(),
                    reason="模型目录不存在")
def test_rife_if_model_available():
    from player.interpolate.download_model import find_model
    from player.interpolate.rife import RifeInterpolator
    model = find_model()
    if model is None:
        pytest.skip("RIFE ONNX 模型未下载")
    eng = RifeInterpolator(str(model), use_cuda=False)
    f0, f1 = _moving_pair()
    mids = eng.interpolate(f0, f1, [0.5])
    assert len(mids) == 1
    assert mids[0].shape == f0.shape
    assert mids[0].dtype == np.uint8
