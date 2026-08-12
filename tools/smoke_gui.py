"""GUI 冒烟测试：启动真实窗口播放测试视频，验证渲染管线。

运行: python tools/smoke_gui.py [秒数]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from player.app import detect, print_report  # noqa: E402
from player.ui.main_window import MainWindow  # noqa: E402

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0

hw = detect()
hw.suggested_engine = "optical"   # 冒烟测试固定用光流，不依赖模型
print_report(hw)

app = QApplication(sys.argv)
app.setStyle("Fusion")
win = MainWindow(hw)
win.show()

video = ROOT / "tests/data/test_60fps.mp4"
win.open_file(str(video))

rendered = []
win.video._orig_set_frame = win.video.set_frame
win.video.set_frame = lambda pts, f: (rendered.append(pts),
                                      win.video._orig_set_frame(pts, f))

t0 = time.monotonic()
last_stats = [None]
win.engine.stats_changed.connect(lambda s: last_stats.__setitem__(0, s))

while time.monotonic() - t0 < DURATION:
    app.processEvents()
    time.sleep(0.02)
    if last_stats[0] and last_stats[0].get("pts", 0) >= min(3.0, DURATION - 1):
        break

print(f"\n=== 冒烟测试结果 ===")
print(f"渲染帧数       : {len(rendered)}")
if rendered:
    print(f"首帧 pts       : {rendered[0]:.3f}s  末帧 pts: {rendered[-1]:.3f}s")
    ok = rendered[0] < 0.5 and rendered[-1] > 2.0 and len(rendered) > 100
else:
    ok = False
print(f"状态栏统计     : {last_stats[0]}")
print(f"结果           : {'✅ 通过' if ok else '❌ 失败'}")
win.engine.close()
app.processEvents()
sys.exit(0 if ok else 1)
