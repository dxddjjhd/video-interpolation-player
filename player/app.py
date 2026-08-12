"""应用入口：QApplication 初始化、硬件检测、主窗口创建。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from .utils.hardware import detect, print_report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    QCoreApplication.setOrganizationName("ZCode")
    QCoreApplication.setApplicationName("VideoInterpPlayer")

    # 硬件检测（在 QApplication 创建前完成，报告打到控制台）
    hw = detect()
    print_report(hw)

    app = QApplication(argv if argv is not None else sys.argv)
    app.setStyle("Fusion")

    # 现代化深色主题
    theme_path = Path(__file__).resolve().parent / "ui" / "theme.qss"
    if theme_path.exists():
        app.setStyleSheet(theme_path.read_text(encoding="utf-8"))

    from .ui.main_window import MainWindow
    win = MainWindow(hw)
    win.show()

    # 支持命令行直接打开视频: python main.py 视频.mp4
    if len(sys.argv) > 1:
        win.open_file(sys.argv[1])

    return app.exec()
