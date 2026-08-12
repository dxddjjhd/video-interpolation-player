"""变速/循环/音量功能测试。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, Qt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from player.core.player_engine import PlayerEngine  # noqa: E402
from player.utils.hardware import detect  # noqa: E402

TEST_VIDEO = ROOT / "tests/data/test_60fps.mp4"
HW = detect()
HW.suggested_engine = "optical"


@pytest.fixture(scope="module")
def qapp():
    return QCoreApplication.instance() or QCoreApplication([])


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="测试视频未生成")
def test_speed_change_preserves_playback(qapp):
    """变速后管线重建，帧流继续正常（pts 单调推进）。"""
    engine = PlayerEngine(HW)
    frames: list[float] = []
    engine.frame_ready.connect(
        lambda pts, f: frames.append(pts), Qt.DirectConnection)
    engine.open(str(TEST_VIDEO))
    engine.set_factor(1)     # 无插帧，帧率=源帧率
    try:
        # 收集基准 pts
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and (not frames or frames[-1] < 0.8):
            qapp.processEvents()
            time.sleep(0.02)
        assert frames, "无帧输出"
        base_count = len(frames)

        # 变速 2x（重建管线）
        engine.set_speed(2.0)
        assert engine._speed == 2.0
        time.sleep(0.3)
        qapp.processEvents()

        deadline = time.monotonic() + 8
        n_before = len(frames)
        while time.monotonic() < deadline and len(frames) < n_before + 30:
            qapp.processEvents()
            time.sleep(0.02)
        assert len(frames) > n_before + 20, "变速后帧流中断"
        new_pts = frames[n_before:]
        # pts 单调
        assert all(b > a for a, b in zip(new_pts, new_pts[1:])), \
            "变速后 pts 不单调"
        # 2x 速度：约 2 倍帧推进
        dt_wall = time.monotonic() - (deadline - 8)
        progress = new_pts[-1] - new_pts[0]
        assert progress / max(dt_wall, 1e-6) > 1.2, \
            f"变速未生效: {progress / max(dt_wall, 1e-6):.2f}x"
    finally:
        engine.close()
        qapp.processEvents()


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="测试视频未生成")
def test_volume_and_loop_api(qapp):
    """音量设置与循环模式 API 正常（不抛异常）。"""
    engine = PlayerEngine(HW)
    engine.open(str(TEST_VIDEO))
    try:
        engine.set_volume(0.0)
        engine.set_volume(1.0)
        engine.set_volume(0.5)
        engine.set_loop(True)
        assert engine._loop is True
        engine.set_loop(False)
        assert engine._loop is False
        # 非法值被钳制
        engine.set_volume(5.0)
        engine.set_speed(9.0)
        assert engine._speed == 4.0
        engine.set_speed(0.1)
        assert engine._speed == 0.25
    finally:
        engine.close()
        qapp.processEvents()


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="测试视频未生成")
def test_loop_playback_restarts(qapp):
    """循环模式：播放结束后回到开头继续。"""
    engine = PlayerEngine(HW)
    frames: list[float] = []
    engine.frame_ready.connect(
        lambda pts, f: frames.append(pts), Qt.DirectConnection)
    engine.set_loop(True)
    engine.set_factor(1)
    engine.open(str(TEST_VIDEO))
    try:
        deadline = time.monotonic() + 20
        # 等播放到接近结尾（6s 视频）
        while time.monotonic() < deadline:
            qapp.processEvents()
            if frames and frames[-1] > 5.5:
                break
            time.sleep(0.02)
        assert frames and frames[-1] > 5.5, "播放未到结尾"
        # 等回到开头（出现 < 0.2 的 pts）
        restart = False
        while time.monotonic() < deadline + 6:
            qapp.processEvents()
            if frames[-1] < 0.2 and len(frames) > 100:
                restart = True
                break
            time.sleep(0.02)
        assert restart, "循环未回到开头"
    finally:
        engine.close()
        qapp.processEvents()
