"""音频播放：QAudioSink 推流 + 基于已播放时长的时钟。

时钟约定：clock() 返回"当前应显示的视频时间"（秒）。
音频块 pts 为该块首采样时间，结合 QAudioSink.processedUSecs()
即可得到绝对播放位置。所有 QAudioSink 调用都在写入线程内完成，
避免跨线程使用 Qt 音频对象。
"""

from __future__ import annotations

import logging
import threading
import time

from PySide6.QtCore import QObject, QMetaObject, Qt, Slot
from PySide6.QtMultimedia import QAudioDevice, QAudioFormat, QAudioSink

from .decoder import AUDIO_CHANNELS, AUDIO_RATE

log = logging.getLogger(__name__)


class AudioPlayer(QObject):
    """推流式 PCM 播放器（48kHz/双声道/s16）。"""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.RLock()   # 注意：clock() 会在 advancing() 持锁时被调用
        self._sink: QAudioSink | None = None
        self._sink_io = None
        self._first_pts: float | None = None   # 首个写入块的 pts
        self._started_mono = 0.0               # 首次写入的单调时钟
        self._total_written_us = 0             # 累计写入的微秒数（用于无设备回退）
        self._paused = False
        self._last_clock = 0.0
        self._last_clock_mono = 0.0
        self._ok = True

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        """在主线程创建音频设备（Qt 音频后端非主线程创建会崩溃）。

        由播放引擎在 open()（主线程）时调用；之后 write() 只做数据推送。
        """
        self._ensure_sink()

    def _ensure_sink(self) -> QAudioSink | None:
        if self._sink is not None:
            return self._sink
        if self._ok is False:
            return None            # 已判定无设备，不再重试
        try:
            # Qt >= 6.7 使用 QMediaDevices；旧版本用 QAudioDevice.defaultOutputDevice
            try:
                from PySide6.QtMultimedia import QMediaDevices
                dev = QMediaDevices.defaultAudioOutput()
            except (ImportError, AttributeError):
                dev = QAudioDevice.defaultOutputDevice()
            if dev.isNull():
                log.warning("无可用音频输出设备，将无声播放")
                self._ok = False
                return None
            fmt = QAudioFormat()
            fmt.setSampleRate(AUDIO_RATE)
            fmt.setChannelCount(AUDIO_CHANNELS)
            fmt.setSampleFormat(QAudioFormat.Int16)
            if not dev.isFormatSupported(fmt):
                log.warning("音频格式不受支持，将无声播放")
                self._ok = False
                return None
            self._sink = QAudioSink(dev, fmt)
            self._sink.setBufferSize(AUDIO_RATE * 2 * AUDIO_CHANNELS * 2)
            self._sink_io = self._sink.start()
            return self._sink
        except Exception as e:  # noqa: BLE001
            log.warning("音频设备初始化失败: %s", e)
            self._ok = False
            self._sink = None
            return None

    def write(self, pts: float, data: bytes) -> None:
        """写入一段 PCM（pts 为首采样时间）。

        设备由 prepare() 在主线程创建；此处只做数据推送。
        无声模式下用墙钟模拟，保证 A/V 同步逻辑照常工作。
        """
        with self._lock:
            if self._first_pts is None:
                self._first_pts = pts
                self._started_mono = time.monotonic()
            if self._sink is None or self._sink_io is None:
                self._total_written_us += int(
                    len(data) / (2 * AUDIO_CHANNELS) * 1e6 / AUDIO_RATE)
                return
            if self._paused:
                return
            try:
                self._sink_io.write(data)
            except Exception as e:  # noqa: BLE001
                log.warning("音频写入失败: %s", e)

    # ------------------------------------------------------------------
    def clock(self) -> float | None:
        """当前音频播放位置（秒）；无声/未开始时返回 None。"""
        with self._lock:
            if self._first_pts is None:
                return None
            if self._sink is not None and self._ok:
                us = self._sink.processedUSecs()
                return self._first_pts + us / 1e6
            # 无声回退：按单调时钟推进
            return self._first_pts + (time.monotonic() - self._started_mono)

    def advancing(self) -> bool:
        """判断时钟是否在推进（检测音频饥饿/卡死）。"""
        now = time.monotonic()
        with self._lock:
            if self._first_pts is None:
                return False
            if self._paused:
                return False
            if now - self._last_clock_mono < 0.5:
                return True   # 采样间隔内不判定
            c = self.clock()
            advancing = c is not None and c > self._last_clock
            self._last_clock = c if c is not None else self._last_clock
            self._last_clock_mono = now
            return advancing

    # ------------------------------------------------------------------
    # 以下三个方法是主线程槽（QMetaObject 可调用），供 _call_main 跨线程调度。
    # QAudioSink 的方法本身不是 Qt 槽，invokeMethod 直接调用会报
    # "No such method"，因此经本类槽转发。
    # ------------------------------------------------------------------
    @Slot()
    def _do_suspend(self) -> None:
        try:
            if self._sink is not None:
                self._sink.suspend()
        except Exception:  # noqa: BLE001
            pass

    @Slot()
    def _do_resume(self) -> None:
        try:
            if self._sink is not None:
                self._sink.resume()
        except Exception:  # noqa: BLE001
            pass

    @Slot()
    def _do_flush(self) -> None:
        """seek 用：停止当前播放并立即重启设备（清缓冲、保留设备）。

        在旧实现中 stop() 会清掉 sink 引用，导致 seek 后 write()
        全部进入无声模式、声音永久消失。
        """
        try:
            if self._sink is not None:
                self._sink.stop()
                self._sink_io = self._sink.start()
        except Exception:  # noqa: BLE001
            pass

    def flush(self) -> None:
        """清空音频缓冲并重置时钟（seek 时调用），设备保留可继续写入。"""
        with self._lock:
            self._paused = False
            self._first_pts = None
        self._call_main("_do_flush")

    @Slot()
    def _do_stop(self) -> None:
        try:
            s = getattr(self, "_stopping_sink", None)
            if s is not None:
                s.stop()
            self._stopping_sink = None
        except Exception:  # noqa: BLE001
            pass

    def _call_main(self, method: str) -> None:
        """把无参槽方法封送到主线程执行（阻塞等待，保证顺序）。"""
        if threading.current_thread() is threading.main_thread():
            getattr(self, method)()
        else:
            QMetaObject.invokeMethod(self, method,
                                     Qt.BlockingQueuedConnection)

    def pause(self) -> None:
        with self._lock:
            self._paused = True
        self._call_main("_do_suspend")

    def resume(self) -> None:
        with self._lock:
            self._paused = False
        self._call_main("_do_resume")

    def stop(self) -> None:
        with self._lock:
            self._paused = False
            sink, self._sink, self._sink_io = self._sink, None, None
            self._first_pts = None
        self._stopping_sink = sink   # 经主线程槽停止旧设备
        self._call_main("_do_stop")
