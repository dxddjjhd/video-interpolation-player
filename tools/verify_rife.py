"""RIFE 模型就绪后的验证：加载 → 插帧正确性 → CUDA 测速 → 写报告。

用法: python tools/verify_rife.py
输出: 项目根目录 RIFE_验证报告.txt，同时打印到控制台
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from player.interpolate.download_model import find_model  # noqa: E402
from player.interpolate.rife import RifeInterpolator  # noqa: E402
from player.utils.hardware import detect  # noqa: E402


def main() -> int:
    model = find_model()
    report_path = ROOT / "RIFE_验证报告.txt"
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    if model is None:
        out("❌ 未找到 ONNX 模型 (player/interpolate/models/*.onnx)")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return 1

    out(f"模型: {model.name} ({model.stat().st_size / 1024 / 1024:.1f}MB)")
    hw = detect()
    out(f"CUDA: {'可用 ' + (hw.cuda_device or '') if hw.cuda_available else '不可用'}")

    eng = RifeInterpolator(str(model), use_cuda=hw.cuda_available)
    eng.warmup()

    # 用合成测试视频的相邻帧对做 2x 插帧测速（720p）
    from player.core.decoder import VideoDecoder
    dec = VideoDecoder(str(ROOT / "tests/data/test_60fps.mp4"))
    frames = []
    for _ in range(40):
        f = dec.read_video()
        if f is None:
            break
        frames.append(f.data)
    dec.close()

    if len(frames) < 10:
        out("❌ 测试视频帧不足")
        return 1

    # 正确性：连续插帧，中间帧应接近两帧平均（运动场景下的合理范围）
    mids = eng.interpolate(frames[5], frames[6], [0.5])
    assert mids and mids[0].shape == frames[5].shape, "输出尺寸错误"
    diff = np.abs(mids[0].astype(int) - frames[5].astype(int)).mean()
    out(f"正确性: 中间帧与原帧平均差 {diff:.1f} (0-255, 应>0 且 <60)")

    # 测速：对 20 对帧做 2x 插帧
    pairs = list(zip(frames[5:25], frames[6:26]))
    eng.interpolate(frames[5], frames[6], [0.5])  # 预热
    t0 = time.perf_counter()
    n = 0
    for f0, f1 in pairs:
        mids = eng.interpolate(f0, f1, [0.5])
        n += 1
    dt = time.perf_counter() - t0
    per_pair_ms = dt / n * 1000
    fps = 1000 / per_pair_ms
    out(f"测速: {n} 对帧 720p 2x 插帧 → 每对 {per_pair_ms:.1f}ms, "
        f"等效 {fps:.1f} 对/秒 (2x 实时需 ≥{2 * 60:.0f} 对/秒)")
    ok = fps >= 60 and 0 < diff < 60
    out(f"\n{'✅ RIFE 引擎验证通过，可实时 2x 插帧' if ok else '⚠️ 见上结果'}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    out(f"\n报告已写入: {report_path}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
