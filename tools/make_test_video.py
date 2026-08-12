"""生成合成测试视频：移动图形 + 纯音音频，用于验证播放与插帧。

用法: python tools/make_test_video.py [输出路径] [时长秒] [fps]
默认输出 tests/data/test_60fps.mp4（60fps, 720p, 6 秒）
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import av
import numpy as np

W, H = 1280, 720
FPS = 60
DURATION = 6.0
AUDIO_RATE = 48000


def make_frame(t: float) -> np.ndarray:
    """画一帧：背景渐变 + 移动圆 + 移动方块 + 旋转扇形。"""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # 背景渐变
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    img[:, :, 0] = (yy * 40 + 10).astype(np.uint8)
    img[:, :, 2] = ((1 - yy) * 40 + 10).astype(np.uint8)

    # 匀速移动的白色圆（插帧时应看到圆在两帧间平滑移动）
    cx = int((t / DURATION) * (W - 200) + 100)
    cy = int(H * 0.5 + 120 * math.sin(2 * math.pi * 0.5 * t))
    cv2_circle(img, (cx, cy), 60, (255, 255, 255))

    # 斜向移动的彩色方块
    bx = int((t * 1.3) % (W + 100) - 50)
    by = int(H * 0.75 + 60 * math.sin(2 * math.pi * 1.0 * t))
    img[max(0, by - 35):by + 35, max(0, bx - 35):bx + 35, :] = (80, 200, 255)

    # 旋转扇形（顺时针旋转的弧线）
    ang = 2 * math.pi * 1.5 * t
    for i in range(100):
        a = ang + i * 0.02
        x = int(W * 0.25 + 150 * math.cos(a))
        y = int(H * 0.25 + 150 * math.sin(a))
        if 0 <= x < W and 0 <= y < H:
            img[y - 1:y + 2, x - 1:x + 2, :] = (255, 120, 0)
    return img


def cv2_circle(img, center, radius, color):
    y, x = center[1], center[0]
    for dy in range(-radius, radius):
        for dx in range(-radius, radius):
            if dx * dx + dy * dy <= radius * radius:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W:
                    img[yy, xx] = color


def make_audio(duration: float, rate: int = AUDIO_RATE) -> np.ndarray:
    """440Hz 正弦 + 1kHz 双音交替，便于听感验证。"""
    n = int(duration * rate)
    t = np.arange(n) / rate
    tone = 0.25 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 880 * t)
    stereo = np.stack([tone, tone], axis=1)
    return (stereo * 32767).astype(np.int16)


def main() -> None:
    tools_dir = Path(__file__).resolve().parent
    out = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else tools_dir.parent / "tests" / "data" / "test_60fps.mp4"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else DURATION
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else FPS
    out.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(str(out), mode="w")

    vstream = container.add_stream("libx264", rate=fps)
    vstream.width, vstream.height = W, H
    vstream.pix_fmt = "yuv420p"
    vstream.options = {"preset": "ultrafast", "crf": "23"}

    astream = container.add_stream("aac", rate=AUDIO_RATE)
    astream.layout = "stereo"

    audio = make_audio(duration)
    n_frames = int(duration * fps)
    for i in range(n_frames):
        t = i / fps
        frame = av.VideoFrame.from_ndarray(make_frame(t), format="rgb24")
        for pkt in vstream.encode(frame):
            container.mux(pkt)

    # 音频分块写入（平面 s16p: 形状 (channels, samples)）
    chunk = AUDIO_RATE // 10
    for start in range(0, len(audio), chunk):
        af = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(audio[start:start + chunk].T),
            format="s16p", layout="stereo")
        af.sample_rate = AUDIO_RATE
        af.pts = int(start / AUDIO_RATE * 90000)
        af.time_base = av.utils.Fraction(1, 90000)
        for pkt in astream.encode(af):
            container.mux(pkt)

    for pkt in vstream.encode():
        container.mux(pkt)
    for pkt in astream.encode():
        container.mux(pkt)
    container.close()
    print(f"测试视频已生成: {out} ({n_frames} 帧 @ {fps}fps, {duration}s, 含音频)")


if __name__ == "__main__":
    main()
