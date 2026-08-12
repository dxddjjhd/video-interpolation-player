"""播放引擎：解码→插帧→渲染 三级流水线 + A/V 同步 + 实时降级。

线程模型（均为 Python 线程）：
- 解码线程：PyAV 逐帧解码，视频帧进有界队列；音频块直接写入
  QAudioSink（其阻塞写天然形成背压，解码速度受音频播放速度约束）
- 插帧线程：消费相邻帧对 → 插帧 → 按音频时钟节拍发射显示帧

同步策略：
- 有音频：以 QAudioSink 播放位置为时钟；音频饥饿（>0.5s 不推进）
  自动切换墙钟模式
- 无音频：墙钟模式（首次发射时刻为基准，累计暂停时间扣除）
- 降级：视频队列积压超阈值时跳过插帧直接输出原帧，保证流畅
"""

from __future__ import annotations

import logging
import threading
import time

from PySide6.QtCore import QObject, Signal

import cv2

from ..interpolate.base import Interpolator
from ..interpolate.factory import create_interpolator
from ..superres.base import SuperRes
from ..superres.factory import create_superres
from ..utils.hardware import HardwareInfo
from .audio import AudioPlayer
from .decoder import DecodedFrame, VideoDecoder
from .pipeline import BoundedQueue

log = logging.getLogger(__name__)

FRAME_PTS_EPS = 0.01          # 提前量容忍（秒）
LATE_FRAME_EPS = 0.1          # 迟到帧阈值：落后超过即立即显示（不追帧）
STALL_DETECT_SEC = 0.6        # 音频停滞判定时长

# 限频日志：同一消息 5 秒内只记录一次（防止异常刷屏）
_throttle_state: dict[str, float] = {}


def _log_throttled(msg: str, detail: Exception | str, interval: float = 5.0) -> None:
    now = time.monotonic()
    key = str(detail)[:120]
    last = _throttle_state.get(key, 0.0)
    if now - last >= interval:
        _throttle_state[key] = now
        log.warning("%s: %s", msg, detail)
    elif len(_throttle_state) > 64:
        _throttle_state.clear()


class _SuperResGuard:
    """AI 超分守护包装：滚动平均超预算时永久切换为 Lanczos。

    AI 超分通常远慢于实时（尤其虚拟 GPU），若继续用 AI 会导致
    每帧延迟累积、音画彻底脱节；切换 Lanczos 保证播放流畅。
    """

    def __init__(self, ai: SuperRes, lanczos: SuperRes, budget: float) -> None:
        self.ai = ai
        self.lanczos = lanczos
        self.budget = budget
        self._avg = 0.0          # 指数滑动平均（秒）
        self.switched = False
        self.name = ai.name
        self.factor = ai.factor

    def upscale(self, frame) -> np.ndarray:
        if self.switched:
            return self.lanczos.upscale(frame)
        t0 = time.perf_counter()
        out = self.ai.upscale(frame)
        dt = time.perf_counter() - t0
        self._avg = 0.7 * self._avg + 0.3 * dt
        if self._avg > self.budget:
            self.switched = True
            # 预期行为（本机算力不足时的自动降级），用 INFO 级别提示
            log.info("AI 超分 %.0fms/帧 超过预算 %.0fms，已切换 Lanczos",
                     self._avg * 1000, self.budget * 1000)
        return out

    def warmup(self) -> None:
        """转发给内部 AI 引擎（引擎对外只看到包装对象）。"""
        self.ai.warmup()

    def close(self) -> None:
        self.ai.close()


class PlayerEngine(QObject):
    frame_ready = Signal(object, object)    # (pts: float, frame: ndarray RGB)
    state_changed = Signal(str)          # stopped | playing | paused
    position_changed = Signal(float)     # 当前播放位置（秒）
    duration_ready = Signal(float)       # 总时长
    stats_changed = Signal(dict)         # 引擎名/生成fps/降级等
    playback_finished = Signal()
    error = Signal(str)

    def __init__(self, hw: HardwareInfo, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._hw = hw
        self._audio = AudioPlayer(self)
        self._lock = threading.RLock()

        self._decoder: VideoDecoder | None = None
        self._video_q = BoundedQueue(8)

        self._stop_ev = threading.Event()
        self._pause_ev = threading.Event()
        self._pause_ev.set()               # 默认不暂停
        self._seek_target: list[float | None] = [None]
        self._decoder_thread: threading.Thread | None = None
        self._interp_thread: threading.Thread | None = None

        self._interp: Interpolator | None = None
        self._engine_kind = "auto"
        self._model_ready = False

        # 超分状态
        self._sr: SuperRes | None = None
        self._sr_kind = "off"
        self._sr_model_ready = False

        self._factor = 2
        self._interp_scale = 1.0          # 插帧分辨率缩放 (0,1]，仅 RIFE 生效
        self._note = ""                   # 引擎切换/降级提示（显示在状态栏）
        self._prev_frame: DecodedFrame | None = None
        self._wall_start_mono = 0.0        # 墙钟基准
        self._wall_base_pts = 0.0
        self._paused_total = 0.0           # 累计暂停秒数
        self._use_wall = False
        self._presented_any = False

        self._gen_frames = 0
        self._gen_secs = 0.0
        self._degraded = False
        self._last_stats = 0.0
        self._finished = False
        self._state = "stopped"

        self.duration = 0.0
        self.src_fps = 0.0

    # ------------------------------------------------------------------
    # 对外控制
    # ------------------------------------------------------------------
    def open(self, path: str) -> None:
        self.close()
        try:
            dec = VideoDecoder(path)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"无法打开文件: {e}")
            return
        self._decoder = dec
        self.duration = dec.duration
        self.src_fps = dec.src_fps
        self._video_hw = (dec.height, dec.width)   # 供基准测试用（不碰解码器）
        self.duration_ready.emit(dec.duration)
        # 音频设备必须在主线程创建（Qt 音频后端跨线程创建会崩溃）
        self._audio.prepare()
        self._video_q = BoundedQueue(2 * self._factor + 2)
        self._interp = create_interpolator(self._engine_kind, self._hw,
                                           model_ready=self._model_ready)
        self._note = ""
        if self._interp is not None:
            self._interp.warmup()
            self._auto_benchmark(dec)
        # 重建超分引擎：close() 会清掉旧实例，
        # 用户可能在打开视频前就选好了超分模式（_sr_kind 保留）
        if self._sr_kind != "off":
            self._create_sr(self._sr_kind)
            if self._sr is not None:
                self._sr.warmup()
                self._benchmark_superres(dec)
        self._finished = False
        self._start_threads()
        self._set_state("playing")

    def play(self) -> None:
        self._audio.resume()
        self._pause_ev.set()
        self._set_state("playing")

    def pause(self) -> None:
        if self._state == "paused":
            return
        self._pause_ev.clear()
        self._audio.pause()
        self._set_state("paused")

    def toggle_play(self) -> None:
        if self._state == "playing":
            self.pause()
        else:
            self.play()

    def seek(self, seconds: float) -> None:
        if self._decoder is None:
            return
        seconds = max(0.0, min(seconds, self.duration))
        with self._lock:
            self._seek_target[0] = seconds
        log.info("seek → %.2fs", seconds)

    def set_factor(self, factor: int) -> None:
        if factor in (2, 4):
            self._factor = factor

    def set_interp_scale(self, scale: float) -> None:
        """设置插帧分辨率缩放 (0,1]；仅对 RIFE 类引擎生效。"""
        if 0.25 <= scale <= 1.0:
            self._interp_scale = scale

    def set_engine(self, kind: str) -> None:
        self._engine_kind = kind
        if self._decoder is not None:
            try:
                self._interp = create_interpolator(
                    kind, self._hw, model_ready=self._model_ready)
            except Exception as e:  # noqa: BLE001
                self.error.emit(str(e))
                return
            if self._interp is not None:
                self._interp.warmup()

    def position(self) -> float:
        """当前播放位置（秒），无时钟时返回 0。"""
        c = self._clock()
        return c if c is not None else 0.0

    def set_model_ready(self, ready: bool) -> None:
        self._model_ready = ready
        if ready and self._engine_kind in ("auto", "rife") and self._decoder is not None:
            self.set_engine(self._engine_kind)

    def set_sr_model_ready(self, ready: bool) -> None:
        """RealESRGAN 模型就绪回调（下载完成后由 UI 调用）。"""
        self._sr_model_ready = ready
        if ready and self._sr_kind.startswith("ai") and self._decoder is not None:
            self.set_superres(self._sr_kind)

    def _create_sr(self, kind: str) -> None:
        """按模式创建超分引擎（AI 模式包上守护包装）。失败时清空并提示。"""
        self._sr_kind = kind
        try:
            sr = create_superres(kind, self._hw, model_ready=self._sr_model_ready)
        except Exception as e:  # noqa: BLE001
            self._sr = None
            self.error.emit(str(e))
            return
        if sr is not None and kind.startswith("ai"):
            from ..superres.lanczos import LanczosSuperRes
            budget = 1.0 / (max(self.src_fps, 1.0) * self._factor)
            sr = _SuperResGuard(sr, LanczosSuperRes(sr.factor), budget)
        self._sr = sr

    def set_superres(self, kind: str) -> None:
        """设置超分模式: off / lanczos2 / lanczos4 / ai2 / ai4。"""
        self._create_sr(kind)
        if self._sr is not None and self._decoder is not None:
            try:
                self._sr.warmup()
                self._benchmark_superres(self._decoder)
            except Exception as e:  # noqa: BLE001
                # 预热/基准只是辅助，失败不应阻断超分生效
                log.warning("超分预热/基准失败（不影响使用）: %s", e)

    def _benchmark_superres(self, dec: VideoDecoder | None = None) -> None:
        """实测超分速度（AI 模式），记录提示信息。

        注意：基准只能使用缓存的分辨率构造测速帧，绝不能从解码器
        读帧 —— 播放中切换超分时解码线程正独占容器，并发读取会
        打乱解码顺序导致插帧失效（表现为"锁源帧率"）。
        """
        if self._sr is None or not self._sr_kind.startswith("ai"):
            return
        import numpy as np
        h, w = getattr(self, "_video_hw", (720, 1280))
        probe = np.zeros((max(h // 4, 32), max(w // 4, 32), 3),
                         dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(2):
            self._sr.upscale(probe)
        per_frame = (time.perf_counter() - t0) / 2
        budget = 1.0 / (max(self.src_fps, 1.0) * self._factor)
        log.info("超分基准: %s → %.1fms/帧 (预算 %.1fms)",
                 getattr(self._sr, "name", "?"), per_frame * 1000,
                 budget * 1000)
        if isinstance(self._sr, _SuperResGuard):
            self._note = f"超分 {self._sr.name} ×{self._sr.factor}"

    def close(self) -> None:
        self._stop_ev.set()
        self._pause_ev.set()
        for t in (self._decoder_thread, self._interp_thread):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        self._video_q.close()
        self._audio.stop()
        if self._decoder is not None:
            self._decoder.close()
            self._decoder = None
        if self._interp is not None:
            self._interp.close()
            self._interp = None
        if self._sr is not None:
            inner = getattr(self._sr, "ai", self._sr)
            inner.close()
            self._sr = None
        self._prev_frame = None
        self._stop_ev.clear()
        self._decoder_thread = None
        self._interp_thread = None
        self._set_state("stopped")

    @property
    def state(self) -> str:
        return self._state

    def _set_state(self, s: str) -> None:
        self._state = s
        self.state_changed.emit(s)

    # ------------------------------------------------------------------
    # 线程启动
    # ------------------------------------------------------------------
    def _start_threads(self) -> None:
        self._decoder_thread = threading.Thread(
            target=self._decode_loop, name="decoder", daemon=True)
        self._interp_thread = threading.Thread(
            target=self._interp_loop, name="interpolator", daemon=True)
        self._decoder_thread.start()
        self._interp_thread.start()

    # ------------------------------------------------------------------
    # 解码线程
    # ------------------------------------------------------------------
    def _decode_loop(self) -> None:
        dec = self._decoder
        assert dec is not None
        try:
            while not self._stop_ev.is_set():
                # 处理 seek 请求
                with self._lock:
                    target = self._seek_target[0]
                if target is not None:
                    with self._lock:
                        self._seek_target[0] = None
                    dec.seek(target)
                    self._video_q.clear()
                    self._audio.flush()   # 清缓冲+重启设备，保留音频输出
                    self._presented_any = False
                    self._use_wall = False
                    continue

                # 音频：直接写入（阻塞写形成背压，天然同步解码速度）
                chunk = dec.read_audio()
                if chunk is not None:
                    self._audio.write(chunk.pts, chunk.data)
                # 视频帧：限速入队（队列有空位才继续解码）。
                # 无声/音频背压缺失时，这保证解码不会跑在插帧前面
                frame = dec.read_video()
                if frame is not None:
                    if not self._video_q.put_limited(
                            frame, self._video_q.maxsize - 1):
                        break
                else:
                    # 视频 EOF
                    self._video_q.close()
                    break
        except Exception as e:  # noqa: BLE001
            if not self._stop_ev.is_set():
                log.exception("解码线程异常")
                self.error.emit(f"解码错误: {e}")

    # ------------------------------------------------------------------
    # 插帧线程
    # ------------------------------------------------------------------
    def _interp_loop(self) -> None:
        try:
            while not self._stop_ev.is_set():
                self._pause_ev.wait(0.02)
                if self._stop_ev.is_set():
                    break
                frame = self._video_q.get(timeout=0.05)
                if frame is None:
                    # 仅当队列已关闭（解码 EOF）且清空时才判定播放结束；
                    # 空但未关闭说明解码线程暂时被音频背压阻塞
                    if self._video_q.is_closed() and self._video_q.empty():
                        self._finish()
                    continue
                self._handle_frame(frame)
        except Exception as e:  # noqa: BLE001
            if not self._stop_ev.is_set():
                log.exception("插帧线程异常")
                self.error.emit(f"插帧错误: {e}")

    def _auto_benchmark(self, dec: VideoDecoder) -> None:
        """打开视频后实测插帧速度：RIFE 无法实时时自动降级。

        预算 = 1/源帧率 秒/对（2x/4x 每对都要在源帧周期内完成）。
        自动档：全分辨率不达标 → 0.5x 缩放再测 → 仍不达标切光流引擎。
        """
        if self._engine_kind == "optical":
            return
        if self._interp is None or "RIFE" not in self._interp.name:
            return
        import time as _t
        import numpy as _np

        frames: list[_np.ndarray] = []
        for _ in range(3):
            f = dec.read_video()
            if f is None:
                break
            frames.append(f.data)
        if len(frames) < 3:
            return
        budget = 1.0 / max(self.src_fps, 1.0)
        manual = self._engine_kind == "rife"

        def measure(fa, fb):
            _t.sleep(0.01)
            t0 = _t.perf_counter()
            for _ in range(3):
                self._interp.interpolate(fa, fb, [0.5])
            return (_t.perf_counter() - t0) / 3

        # 基准缩放档位：尊重用户在播放前手动设置的插帧缩放
        if not manual:
            if self._interp_scale < 1.0:
                scales = [self._interp_scale, 0.5]
            else:
                scales = [1.0, 0.5]
        else:
            scales = [self._interp_scale]
        for scale in scales:
            if scale < 1.0:
                import cv2
                fa = cv2.resize(frames[0], None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
                fb = cv2.resize(frames[1], None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
            else:
                fa, fb = frames[0], frames[1]
            per_pair = measure(fa, fb)
            log.info("插帧基准: %s %s → %.1fms/对 (预算 %.1fms)",
                     self._interp.name, f"{scale:.0%}", per_pair * 1000,
                     budget * 1000)
            if per_pair <= budget:
                self._interp_scale = scale
                if scale < 1.0:
                    self._note = f"RIFE 插帧缩放 {scale:.0%} 保实时"
                return
        # 所有档位都不达标
        if manual:
            self._note = "RIFE 超出本机实时能力，播放可能降级为原帧率"
            return
        self._interp.close()
        from ..interpolate.optical_flow import OpticalFlowInterpolator
        self._interp = OpticalFlowInterpolator()
        self._interp.warmup()
        self._note = (f"{'RIFE' if 'RIFE' in self._interp.name else 'AI'} 超出本机实时能力"
                      f"（{per_pair * 1000:.0f}ms/对 > 预算 {budget * 1000:.0f}ms），"
                      f"已自动切换光流引擎")

    def _handle_frame(self, frame: DecodedFrame) -> None:
        prev = self._prev_frame
        if prev is None:
            # 首帧：建立墙钟基准后直接展示
            self._prev_frame = frame
            self._reset_wall(frame)
            self._present(frame)
            return
        if frame.pts < prev.pts - 0.5:
            # seek 后时间回跳：重置基准
            self._prev_frame = frame
            self._reset_wall(frame)
            self._present(frame)
            return

        dt = frame.pts - prev.pts
        if dt <= 0:
            self._prev_frame = frame
            return

        # 视频中途分辨率变化（拼接/剪辑视频常见）：光流引擎无法处理
        # 不同尺寸的帧对，重置插帧基准当作新片段开始
        if frame.data.shape != prev.data.shape:
            log.info("视频分辨率变化 %dx%d → %dx%d，重置插帧基准",
                     prev.data.shape[1], prev.data.shape[0],
                     frame.data.shape[1], frame.data.shape[0])
            self._prev_frame = frame
            self._present(frame)
            return

        # 降级判定：视频落后音频时钟 → 跳过插帧直接出帧追赶
        clock = self._clock()
        lag = (clock - prev.pts) if clock is not None else 0.0
        self._degraded = lag > LATE_FRAME_EPS * 5 \
            or self._video_q.qsize() >= self._video_q.maxsize - 1
        if self._degraded:
            self._present(frame)
            self._prev_frame = frame
            return

        # 正常插帧（RIFE 按需缩放分辨率以保实时）
        times = [(i + 1) / self._factor for i in range(self._factor - 1)]
        scale = self._interp_scale
        try:
            if scale < 1.0:
                a = cv2.resize(prev.data, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
                b = cv2.resize(frame.data, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
                mids = self._interp.interpolate(a, b, times)
                mids = [cv2.resize(m, (frame.data.shape[1], frame.data.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
                        for m in mids]
            else:
                mids = self._interp.interpolate(prev.data, frame.data, times)
        except Exception as e:  # noqa: BLE001
            _log_throttled("插帧失败（跳过该对）", e)
            self._present(frame)
            self._prev_frame = frame
            return
        for t, mid in zip(times, mids):
            self._present(DecodedFrame(pts=prev.pts + dt * t, data=mid))
        self._present(frame)
        self._prev_frame = frame

    def _reset_wall(self, frame: DecodedFrame) -> None:
        self._presented_any = False
        self._wall_start_mono = time.monotonic()
        self._wall_base_pts = frame.pts
        self._paused_total = 0.0

    def _present(self, frame: DecodedFrame) -> None:
        """按时钟节拍发射一帧（在插帧线程内等待）。"""
        if self._stop_ev.is_set():
            return
        clock = self._clock()
        if clock is None or frame.pts < clock - LATE_FRAME_EPS:
            pass   # 无时钟或帧已迟到：直接显示
        else:
            self._wait_until(frame.pts)
        if self._sr is not None:
            frame.data = self._sr.upscale(frame.data)
        self._gen_frames += 1
        self._presented_any = True
        self.frame_ready.emit(frame.pts, frame.data)
        self._maybe_stats(frame.pts)

    def _wait_until(self, pts: float) -> None:
        """等待时钟到达 pts。音频饥饿时切换墙钟模式。"""
        while True:
            if self._stop_ev.is_set():
                return
            if not self._audio.advancing():
                self._use_wall = True
            # 暂停时长计入墙钟偏移
            if not self._pause_ev.is_set():
                paused_at = time.monotonic()
                self._pause_ev.wait(0.003)
                if self._pause_ev.is_set():
                    self._paused_total += time.monotonic() - paused_at
                continue
            clock = self._clock()
            if clock is None or pts <= clock + FRAME_PTS_EPS:
                return
            self._pause_ev.wait(0.003)

    def _clock(self) -> float | None:
        """当前播放位置：音频时钟优先，墙钟兜底。"""
        if not self._use_wall:
            c = self._audio.clock()
            if c is not None:
                return c
        if self._presented_any:
            elapsed = time.monotonic() - self._wall_start_mono - self._paused_total
            return self._wall_base_pts + elapsed
        return None

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._set_state("stopped")
        self.playback_finished.emit()

    def _maybe_stats(self, pts: float) -> None:
        now = time.monotonic()
        if now - self._last_stats >= 0.5:
            fps = self._gen_frames / (now - self._last_stats) \
                if self._last_stats else 0.0
            self._gen_frames = 0
            self._last_stats = now
            self.stats_changed.emit({
                "engine": self._interp.name if self._interp else "-",
                "gen_fps": round(fps, 1),
                "degraded": self._degraded,
                "src_fps": round(self.src_fps, 2),
                "factor": self._factor,
                "scale": self._interp_scale,
                "sr": (f"{self._sr.name} ×{self._sr.factor}"
                       if self._sr is not None else "关"),
                "sr_fallback": (getattr(self._sr, "switched", False)
                                if self._sr is not None else False),
                "note": self._note,
                "pts": pts,
            })
            self.position_changed.emit(pts)
