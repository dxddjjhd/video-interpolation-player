"""播放引擎集成测试（无 GUI）：帧流节奏、pts 单调、暂停恢复。

通过 QCoreApplication.processEvents 泵送 Qt 队列连接，
在真实时钟节奏下验证插帧播放管线。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, Qt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from player.core.player_engine import PlayerEngine  # noqa: E402
from player.utils.hardware import detect  # noqa: E402

TEST_VIDEO = ROOT / "tests/data/test_60fps.mp4"

# 用光流引擎保证测试不依赖模型下载
HW = detect()
HW.suggested_engine = "optical"


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance() or QCoreApplication([])
    return app


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="测试视频未生成")
def test_engine_streams_frames_at_factor2(qapp):
    engine = PlayerEngine(HW)
    frames: list[tuple[float, np.ndarray]] = []
    engine.frame_ready.connect(
        lambda pts, f: frames.append((pts, f)), Qt.DirectConnection)

    engine.open(str(TEST_VIDEO))
    engine.set_factor(2)

    deadline = time.monotonic() + 20.0
    try:
        # 收集前 3 秒内容
        while time.monotonic() < deadline:
            qapp.processEvents()
            if frames and frames[-1][0] >= 3.0:
                break
            time.sleep(0.02)
        assert frames, "没有收到任何帧"

        pts_list = [p for p, _ in frames]
        # 2x: 3 秒内应有约 360 帧
        n_expected = int(3.0 * 60 * 2)
        assert len(frames) >= n_expected * 0.7, \
            f"帧数不足: {len(frames)} < {n_expected * 0.7}"
        # pts 单调递增（允许微小抖动）
        assert all(b > a for a, b in zip(pts_list, pts_list[1:]))
        # 帧尺寸正确
        assert frames[0][1].shape == (720, 1280, 3)
        # 平均帧间隔 ≈ 1/120s（共享云主机性能波动，容差放宽到 95-130fps）
        gaps = [b - a for a, b in zip(pts_list, pts_list[1:])]
        avg = sum(gaps) / len(gaps)
        assert 1 / 130 < avg < 1 / 95, f"帧间隔异常: {avg:.5f}s"
    finally:
        engine.close()


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="测试视频未生成")
def test_engine_pause_resume(qapp):
    engine = PlayerEngine(HW)
    frames: list[float] = []
    engine.frame_ready.connect(
        lambda pts, f: frames.append(pts), Qt.DirectConnection)

    engine.open(str(TEST_VIDEO))
    engine.set_factor(2)
    try:
        # 等前 0.8s 开始播放
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not frames:
            qapp.processEvents()
            time.sleep(0.02)
        assert frames

        engine.pause()
        assert engine.state == "paused"
        n_at_pause = len(frames)
        time.sleep(0.4)
        qapp.processEvents()
        assert len(frames) == n_at_pause, "暂停期间仍在出帧"

        engine.play()
        assert engine.state == "playing"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(frames) < n_at_pause + 20:
            qapp.processEvents()
            time.sleep(0.02)
        assert len(frames) >= n_at_pause + 20, "恢复后未继续出帧"
    finally:
        engine.close()
        qapp.processEvents()
