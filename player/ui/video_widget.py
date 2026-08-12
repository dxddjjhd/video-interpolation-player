"""视频渲染组件：QPainter 软件渲染（QImage 零拷贝 + drawImage）。

采用 QPainter 而非 QOpenGLWidget：部分虚拟化/云环境（如本机 vGPU）
的 OpenGL 桥接损坏（3.3 core 上下文创建失败、部分 GL 调用静默崩溃），
QPainter 走 Qt 光栅引擎，任何环境都稳定。720p 下每帧 CPU 开销 2-5ms，
足以支撑 120fps 输出；缩放由 drawImage 硬件无关地完成。
"""

from __future__ import annotations

import logging
import time

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)


class VideoWidget(QWidget):
    """显示 RGB 帧；帧由主线程 set_frame() 注入。

    左上角可叠加 OSD：实时显示画面显示帧率（set_frame 调用频率）。
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._frame: np.ndarray | None = None
        self._qimg: QImage | None = None
        self._frame_pts = 0.0

        # OSD 帧率统计
        self.show_fps_osd = True
        self._fps = 0.0
        self._fps_count = 0
        self._fps_t0 = time.monotonic()

    # ------------------------------------------------------------------
    def set_frame(self, pts: float, frame: np.ndarray) -> None:
        """设置待显示帧（RGB uint8），触发重绘。QImage 零拷贝引用缓冲。

        非 C 连续帧先拷贝（解码器已保证连续，这里是防御）。
        """
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        # 显示帧率统计（每秒实际更新的帧数）
        now = time.monotonic()
        self._fps_count += 1
        if now - self._fps_t0 >= 0.5:
            self._fps = self._fps_count / (now - self._fps_t0)
            self._fps_count = 0
            self._fps_t0 = now
        self._frame = frame
        self._frame_pts = pts
        h, w = frame.shape[:2]
        self._qimg = QImage(frame.data, w, h, w * 3, QImage.Format_RGB888)
        self.update()

    def clear_frame(self) -> None:
        self._frame = None
        self._qimg = None
        self.update()

    @property
    def current_pts(self) -> float:
        return self._frame_pts

    @property
    def display_fps(self) -> float:
        """当前实际显示帧率（每秒画面更新次数）。"""
        return self._fps

    # ------------------------------------------------------------------
    def paintEvent(self, e) -> None:  # noqa: N802 (Qt 命名)
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), Qt.black)
            if self._qimg is not None:
                # 宽高比适配（drawImage 目标矩形缩放，QImage 源保持原尺寸）
                iw, ih = self._qimg.width(), self._qimg.height()
                if iw > 0 and ih > 0:
                    scale = min(self.width() / iw, self.height() / ih)
                    dw, dh = iw * scale, ih * scale
                    target = self.rect()
                    x = target.x() + (target.width() - dw) / 2
                    y = target.y() + (target.height() - dh) / 2
                    p.setRenderHint(QPainter.SmoothPixmapTransform, scale < 1.0)
                    p.drawImage(QRectF(x, y, dw, dh), self._qimg)
            # OSD：左上角半透明黑条 + 显示帧率
            if self.show_fps_osd:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(0, 0, 0, 140))
                p.drawRect(8, 8, 120, 26)
                p.setPen(Qt.white)
                p.drawText(QRectF(8, 8, 120, 26), Qt.AlignCenter,
                           f"{self._fps:.0f} fps")
        finally:
            p.end()
