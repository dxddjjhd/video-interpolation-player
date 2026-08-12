"""管线单元测试：有界队列语义、解码器帧序。"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from player.core.pipeline import BoundedQueue  # noqa: E402


def test_queue_fifo():
    q = BoundedQueue(3)
    for i in range(3):
        assert q.put(i)
    assert q.qsize() == 3
    assert [q.get() for _ in range(3)] == [0, 1, 2]


def test_queue_drops_oldest_when_full():
    q = BoundedQueue(3)
    for i in range(6):
        q.put(i)
    assert q.qsize() == 3
    assert q.get() == 3   # 0,1,2 被丢弃


def test_queue_get_timeout():
    q = BoundedQueue(2)
    t0 = time.monotonic()
    assert q.get(timeout=0.05) is None
    assert time.monotonic() - t0 < 0.5


def test_queue_close_wakes_get():
    q = BoundedQueue(2)
    result = []

    def reader():
        result.append(q.get(timeout=2.0))

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)
    q.close()
    t.join(timeout=1.0)
    assert result == [None]


def test_queue_clear():
    q = BoundedQueue(4)
    for i in range(4):
        q.put(i)
    q.clear()
    assert q.empty()
    assert q.get(timeout=0.01) is None


# ------------------------------------------------------------------
# 解码器测试（需要测试视频）
# ------------------------------------------------------------------
TEST_VIDEO = ROOT / "tests/data/test_60fps.mp4"


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="测试视频未生成")
def test_decoder_frame_order_and_pts():
    from player.core.decoder import VideoDecoder
    dec = VideoDecoder(str(TEST_VIDEO))
    try:
        assert dec.width == 1280 and dec.height == 720
        assert abs(dec.src_fps - 60.0) < 1.0
        assert dec.duration > 5.0

        pts_list = []
        frames = 0
        while frames < 30:
            f = dec.read_video()
            if f is None:
                break
            pts_list.append(f.pts)
            frames += 1
            assert f.data.shape == (720, 1280, 3)
            assert f.data.dtype == np.uint8
        assert frames >= 20
        # pts 严格递增（展示序）
        assert all(b > a for a, b in zip(pts_list, pts_list[1:]))
        # 帧间距约 1/60s
        gaps = [b - a for a, b in zip(pts_list, pts_list[1:])]
        assert abs(sum(gaps) / len(gaps) - 1 / 60) < 0.005
    finally:
        dec.close()


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="测试视频未生成")
def test_decoder_audio():
    from player.core.decoder import VideoDecoder
    dec = VideoDecoder(str(TEST_VIDEO))
    try:
        chunks = []
        for _ in range(5):
            c = dec.read_audio()
            if c is None:
                break
            chunks.append(c)
        assert len(chunks) >= 3
        assert all(len(c.data) > 0 for c in chunks)
        # 相邻块 pts 连续
        assert all(b.pts > a.pts for a, b in zip(chunks, chunks[1:]))
    finally:
        dec.close()


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="测试视频未生成")
def test_decoder_seek():
    from player.core.decoder import VideoDecoder
    dec = VideoDecoder(str(TEST_VIDEO))
    try:
        dec.seek(3.0)
        frames = []
        for _ in range(10):
            f = dec.read_video()
            if f is None:
                break
            frames.append(f.pts)
        assert len(frames) >= 5
        # seek 后应回到目标附近（可能在关键帧稍前）
        assert frames[0] <= 3.0 + 0.05
        # 之后时间递增
        assert all(b > a for a, b in zip(frames, frames[1:]))
    finally:
        dec.close()
