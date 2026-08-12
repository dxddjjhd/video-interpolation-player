"""主窗口：播放控制、倍率/引擎切换、进度条、状态栏。"""

from __future__ import annotations

import logging
import os
import threading

from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QMainWindow, QProgressDialog, QPushButton,
                               QSlider, QStatusBar, QVBoxLayout, QWidget)

from ..core.player_engine import PlayerEngine
from ..interpolate import download_model
from ..superres import download_model as sr_download_model
from ..utils.hardware import HardwareInfo
from .video_widget import VideoWidget

log = logging.getLogger(__name__)


class ModelLoader(QObject):
    """后台线程下载模型（RIFE 或超分），通过信号回报进度。"""

    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, kind: str = "rife") -> None:
        super().__init__()
        self._kind = kind
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        def cb(msg: str) -> None:
            self.progress.emit(msg)
        if self._kind == "sr":
            path = sr_download_model.ensure_model(progress_cb=cb)
        else:
            path = download_model.ensure_model(progress_cb=cb)
        ok = path is not None
        self.finished.emit(ok, str(path) if path else "")


class MainWindow(QMainWindow):

    def __init__(self, hw: HardwareInfo, parent=None) -> None:
        super().__init__(parent)
        self._hw = hw
        self.engine = PlayerEngine(hw, self)
        self._current_path: str | None = None
        self._updating_slider = False

        self.setWindowTitle("视频实时插帧播放器")
        self.resize(1024, 640)
        self._build_ui()
        self._connect_engine()
        self._maybe_download_model()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        self.video = VideoWidget(central)
        layout.addWidget(self.video, 1)

        # 控制栏
        bar = QHBoxLayout()
        self.btn_open = QPushButton("打开", self)
        self.btn_play = QPushButton("播放", self)
        self.btn_stop = QPushButton("停止", self)
        bar.addWidget(self.btn_open)
        bar.addWidget(self.btn_play)
        bar.addWidget(self.btn_stop)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 1000)
        self.slider.setEnabled(False)
        self.time_label = QLabel("00:00 / 00:00", self)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.time_label)

        bar.addWidget(QLabel("倍率", self))
        self.combo_factor = QComboBox(self)
        self.combo_factor.addItem("2x", 2)
        self.combo_factor.addItem("4x", 4)
        self.combo_factor.setCurrentIndex(0)
        bar.addWidget(self.combo_factor)

        bar.addWidget(QLabel("引擎", self))
        self.combo_engine = QComboBox(self)
        self.combo_engine.addItem("自动", "auto")
        self.combo_engine.addItem("RIFE (AI)", "rife")
        self.combo_engine.addItem("光流", "optical")
        self.combo_engine.setCurrentIndex(0)
        bar.addWidget(self.combo_engine)

        bar.addWidget(QLabel("插帧分辨率", self))
        self.combo_scale = QComboBox(self)
        self.combo_scale.addItem("原尺寸", 1.0)
        self.combo_scale.addItem("75%", 0.75)
        self.combo_scale.addItem("50%", 0.5)
        self.combo_scale.setCurrentIndex(0)
        self.combo_scale.setToolTip("RIFE 插帧计算分辨率（降低可提升速度，显示时放大回原尺寸）")
        bar.addWidget(self.combo_scale)

        bar.addWidget(QLabel("超分", self))
        self.combo_sr = QComboBox(self)
        self.combo_sr.addItem("关闭", "off")
        self.combo_sr.addItem("Lanczos 2x", "lanczos2")
        self.combo_sr.addItem("Lanczos 4x", "lanczos4")
        self.combo_sr.addItem("FSR 2x", "fsr2")
        self.combo_sr.addItem("FSR 4x", "fsr4")
        self.combo_sr.addItem("AI 2x", "ai2")
        self.combo_sr.addItem("AI 4x", "ai4")
        self.combo_sr.setCurrentIndex(0)
        self.combo_sr.setToolTip(
            "超分辨率放大（FSR=Lanczos+边缘锐化；AI=RealESRGAN，跟不上时自动降级）")
        bar.addWidget(self.combo_sr)
        layout.addLayout(bar)

        # 菜单
        m_file = self.menuBar().addMenu("文件(&F)")
        act_open = QAction("打开(&O)...", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self.open_dialog)
        m_file.addAction(act_open)
        act_close = QAction("退出(&X)", self)
        act_close.setShortcut(QKeySequence.Quit)
        act_close.triggered.connect(self.close)
        m_file.addAction(act_close)

        m_play = self.menuBar().addMenu("播放(&P)")
        act_toggle = QAction("播放/暂停(&P)", self)
        act_toggle.setShortcut(Qt.Key_Space)
        act_toggle.triggered.connect(self.engine.toggle_play)
        m_play.addAction(act_toggle)
        self.act_osd = QAction("显示帧率(&F)", self)
        self.act_osd.setCheckable(True)
        self.act_osd.setChecked(True)
        self.act_osd.toggled.connect(
            lambda on: setattr(self.video, "show_fps_osd", on))
        m_play.addAction(self.act_osd)

        m_help = self.menuBar().addMenu("帮助(&H)")
        act_about = QAction("关于(&A)", self)
        act_about.triggered.connect(self._about)
        m_help.addAction(act_about)

        # 状态栏
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self.lbl_status = QLabel("就绪", self)
        self.lbl_stats = QLabel("", self)
        sb.addWidget(self.lbl_status, 1)
        sb.addPermanentWidget(self.lbl_stats)

        # 拖放打开
        self.setAcceptDrops(True)

        # 信号
        self.btn_open.clicked.connect(self.open_dialog)
        self.btn_play.clicked.connect(self.engine.toggle_play)
        self.btn_stop.clicked.connect(self.engine.close)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.combo_factor.currentIndexChanged.connect(
            lambda _: self.engine.set_factor(self.combo_factor.currentData()))
        self.combo_engine.currentIndexChanged.connect(self._on_engine_changed)
        self.combo_scale.currentIndexChanged.connect(
            lambda _: self.engine.set_interp_scale(
                self.combo_scale.currentData()))
        self.combo_sr.currentIndexChanged.connect(self._on_sr_changed)

    def _connect_engine(self) -> None:
        self.engine.frame_ready.connect(self.video.set_frame)
        self.engine.state_changed.connect(self._on_state)
        self.engine.position_changed.connect(self._on_position)
        self.engine.duration_ready.connect(self._on_duration)
        self.engine.stats_changed.connect(self._on_stats)
        self.engine.playback_finished.connect(self._on_finished)
        self.engine.error.connect(self._on_error)

    # ------------------------------------------------------------------
    def _maybe_download_model(self) -> None:
        """CUDA 可用但模型缺失 → 后台下载。"""
        # 超分模型已就绪则直接标记（RealESRGAN 模型很小，可能已下载）
        if (sr_download_model.find_model(2) is not None
                or sr_download_model.find_model(4) is not None):
            self.engine.set_sr_model_ready(True)
        if not self._hw.cuda_available:
            self.engine.set_model_ready(False)
            return
        if download_model.find_model() is not None:
            self.engine.set_model_ready(True)
            return
        self._dl = ModelLoader()
        self._dl.progress.connect(self._on_dl_progress)
        self._dl.finished.connect(self._on_dl_finished)
        self._dl_dialog = QProgressDialog(
            "正在下载 RIFE 插帧模型 (~190MB)，首次运行需要…", None, 0, 0, self)
        self._dl_dialog.setWindowTitle("模型下载")
        self._dl_dialog.setMinimumWidth(420)
        self._dl_dialog.setCancelButton(None)
        self._dl_dialog.show()
        self._dl.start()

    def _on_dl_progress(self, msg: str) -> None:
        if hasattr(self, "_dl_dialog"):
            self._dl_dialog.setLabelText(msg)

    def _on_dl_finished(self, ok: bool, path: str) -> None:
        if hasattr(self, "_dl_dialog"):
            self._dl_dialog.close()
        if ok:
            self.lbl_status.setText(f"RIFE 模型就绪: {os.path.basename(path)}")
            self.engine.set_model_ready(True)
        else:
            self.lbl_status.setText(
                "RIFE 模型下载失败，可手动下载 rife*.onnx 放入 "
                f"{download_model.MODELS_DIR}，或使用光流引擎")

    def _on_sr_changed(self, _: int = 0) -> None:
        kind = self.combo_sr.currentData()
        if kind.startswith("ai") and sr_download_model.find_model(
                int(kind[-1])) is None:
            # AI 超分模型缺失 → 先下载（~6.5MB，很快）
            self._dl_sr = ModelLoader("sr")
            self._dl_sr.progress.connect(self._on_dl_progress)
            self._dl_sr.finished.connect(self._on_sr_dl_finished)
            self._dl_dialog = QProgressDialog(
                "正在下载 RealESRGAN 超分模型 (~6.5MB)…", None, 0, 0, self)
            self._dl_dialog.setWindowTitle("模型下载")
            self._dl_dialog.setMinimumWidth(420)
            self._dl_dialog.setCancelButton(None)
            self._dl_dialog.show()
            self._dl_sr.start()
            return
        self.engine.set_sr_model_ready(True)
        try:
            self.engine.set_superres(kind)
        except Exception as e:  # noqa: BLE001
            self.lbl_status.setText(f"超分切换失败: {e}")

    def _on_sr_dl_finished(self, ok: bool, path: str) -> None:
        if hasattr(self, "_dl_dialog"):
            self._dl_dialog.close()
        if ok:
            self.lbl_status.setText("RealESRGAN 模型就绪")
            self.engine.set_sr_model_ready(True)
            self.engine.set_superres(self.combo_sr.currentData())
        else:
            self.lbl_status.setText("RealESRGAN 模型下载失败，可使用 Lanczos 超分")
            self.combo_sr.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # 播放控制
    # ------------------------------------------------------------------
    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开视频", "",
            "视频文件 (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.ts *.m4v);;所有文件 (*)")
        if path:
            self.open_file(path)

    def open_file(self, path: str) -> None:
        self._current_path = path
        self.engine.open(path)
        self.setWindowTitle(f"{os.path.basename(path)} — 视频实时插帧播放器")
        self.btn_play.setText("暂停")

    def _on_engine_changed(self, _: int) -> None:
        kind = self.combo_engine.currentData()
        try:
            self.engine.set_engine(kind)
        except Exception as e:  # noqa: BLE001
            self.lbl_status.setText(f"引擎切换失败: {e}")

    # ------------------------------------------------------------------
    # 引擎信号
    # ------------------------------------------------------------------
    def _on_state(self, state: str) -> None:
        if state == "playing":
            self.btn_play.setText("暂停")
        elif state == "paused":
            self.btn_play.setText("播放")
        else:
            self.btn_play.setText("播放")

    def _on_position(self, pos: float) -> None:
        if self._updating_slider:
            return
        self.slider.setValue(int(pos / max(self.engine.duration, 1e-6) * 1000))

    def _on_duration(self, dur: float) -> None:
        self.slider.setEnabled(dur > 0)
        self._update_time_label()

    def _on_stats(self, stats: dict) -> None:
        deg = " ⚠降级" if stats.get("degraded") else ""
        note = stats.get("note") or ""
        sr = stats.get("sr") or "关"
        sr_fb = " ⚠SR降级Lanczos" if stats.get("sr_fallback") else ""
        self.lbl_stats.setText(
            f"{stats['engine']} | 生成 {stats['gen_fps']:.0f} fps"
            f" (源 {stats['src_fps']:.0f}fps ×{stats['factor']}){deg}"
            f" | 超分 {sr}{sr_fb}"
            f"{' | ' + note if note else ''}")
        self._update_time_label(stats.get("pts"))

    def _update_time_label(self, pos: float | None = None) -> None:
        if pos is None:
            pos = self.slider.value() / 1000.0 * self.engine.duration
        cur = self._fmt(pos)
        total = self._fmt(self.engine.duration)
        self.time_label.setText(f"{cur} / {total}")

    @staticmethod
    def _fmt(sec: float) -> str:
        sec = max(0, int(sec))
        return f"{sec // 60:02d}:{sec % 60:02d}"

    def _on_finished(self) -> None:
        self.lbl_status.setText("播放结束")

    def _on_error(self, msg: str) -> None:
        self.lbl_status.setText(f"错误: {msg}")

    def _on_slider_moved(self, value: int) -> None:
        pos = value / 1000.0 * self.engine.duration
        self._update_time_label(pos)

    def _on_slider_released(self) -> None:
        self._updating_slider = True
        try:
            self.engine.seek(self.slider.value() / 1000.0 * self.engine.duration)
        finally:
            self._updating_slider = False

    # ------------------------------------------------------------------
    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:  # noqa: N802
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.isfile(path):
                self.open_file(path)
                return

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key_Space:
            self.engine.toggle_play()
        elif e.key() == Qt.Key_Left:
            self.engine.seek(max(0.0, self.engine.position() - 5.0))
        elif e.key() == Qt.Key_Right:
            self.engine.seek(self.engine.position() + 5.0)
        else:
            super().keyPressEvent(e)

    def _about(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        from .. import __version__
        QMessageBox.about(
            self, "关于",
            f"视频实时插帧播放器 v{__version__}\n\n"
            "引擎: RIFE (ONNX/CUDA) · OpenCV 光流\n"
            "快捷键: 空格 播放/暂停, ←/→ 快退/快进 5s")

    def closeEvent(self, e) -> None:  # noqa: N802
        self.engine.close()
        super().closeEvent(e)
