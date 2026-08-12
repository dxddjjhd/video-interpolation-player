"""视频解码器：基于 PyAV (FFmpeg) 的帧级解码与音频重采样。

输出约定：
- 视频帧为 RGB uint8 ndarray (H, W, 3)，pts 为秒（浮点）
- 音频为 48kHz / 双声道 / s16 交错 PCM 字节串，pts 为该块首采样时间

注意：PyAV 的 decode(指定流) 会丢弃其他流的包，因此视频与音频
各使用独立的 container（同一文件打开两次），保证互不干扰。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import av
import numpy as np

log = logging.getLogger(__name__)

# 音频输出统一格式（与 QAudioSink 匹配）
AUDIO_RATE = 48000
AUDIO_CHANNELS = 2
AUDIO_CHUNK_SAMPLES = 4096  # ~85ms


@dataclass
class DecodedFrame:
    pts: float          # 秒
    data: np.ndarray    # RGB uint8 (H, W, 3)


@dataclass
class AudioChunk:
    pts: float          # 首采样时间（秒）
    data: bytes         # s16 交错 PCM


class VideoDecoder:
    """打开一个媒体文件，按时间序输出视频帧与重采样音频块。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self.container = av.open(path)          # 视频专用
        self.vstream = next(
            (s for s in self.container.streams if s.type == "video"), None)
        if self.vstream is None:
            self.container.close()
            raise ValueError(f"文件中没有视频流: {path}")

        self.acontainer: av.container.InputContainer | None = None
        self.astream = None
        try:
            self.acontainer = av.open(path)     # 音频专用
            self.astream = next(
                (s for s in self.acontainer.streams if s.type == "audio"), None)
        except Exception as e:  # noqa: BLE001
            log.warning("音频流打开失败: %s", e)

        self.width = self.vstream.codec_context.width
        self.height = self.vstream.codec_context.height
        dur = float(self.vstream.duration or 0) * float(self.vstream.time_base)
        if not dur or dur <= 0:
            dur = float(self.container.duration or 0) / 1_000_000.0
        self.duration = dur

        rate = self.vstream.average_rate or self.vstream.base_rate
        self.src_fps = float(rate) if rate else 25.0

        self._resampler = None
        if self.astream is not None:
            self._resampler = av.AudioResampler(
                format="s16", layout="stereo", rate=AUDIO_RATE)
        self._audio_acc = bytearray()
        self._audio_acc_pts: float | None = None

    # ------------------------------------------------------------------
    def read_video(self) -> DecodedFrame | None:
        """读下一视频帧（展示序），文件结束返回 None。

        返回的数组保证 C 连续（QImage 等下游零拷贝路径的要求；
        某些 pix_fmt 转换出的 rgb24 存在行步长，需显式拷贝）。
        """
        try:
            for frame in self.container.decode(self.vstream):
                pts = float(frame.pts) * float(frame.time_base)
                img = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
                return DecodedFrame(pts=pts, data=img)
        except av.error.EOFError:
            pass    # 正常结束：码流耗尽
        return None

    def read_audio(self) -> AudioChunk | None:
        """读下一音频块（重采样为统一格式），文件结束返回 None。"""
        if self._resampler is None:
            return None
        while True:
            # 先消化累积缓冲区
            if len(self._audio_acc) >= AUDIO_CHUNK_SAMPLES * 2 * AUDIO_CHANNELS:
                return self._pop_chunk()
            try:
                frame = next(self.acontainer.decode(self.astream))
            except (StopIteration, av.error.EOFError):
                if self._audio_acc:
                    return self._pop_chunk()
                return None
            for out in self._resampler.resample(frame):
                arr = out.to_ndarray()          # (samples, channels) 交错
                if arr.ndim == 2 and arr.shape[0] == AUDIO_CHANNELS \
                        and arr.shape[1] != AUDIO_CHANNELS:
                    arr = arr.T                  # 平面格式转交错
                if self._audio_acc_pts is None:
                    self._audio_acc_pts = float(out.pts or 0) * float(out.time_base)
                self._audio_acc += arr.astype(np.int16).tobytes()

    def _pop_chunk(self) -> AudioChunk:
        n = AUDIO_CHUNK_SAMPLES * 2 * AUDIO_CHANNELS
        data = bytes(self._audio_acc[:n])
        del self._audio_acc[:n]
        chunk = AudioChunk(pts=self._audio_acc_pts or 0.0, data=data)
        if not self._audio_acc:
            self._audio_acc_pts = None
        return chunk

    # ------------------------------------------------------------------
    def seek(self, seconds: float) -> None:
        """跳到指定时间（精确到关键帧，随后由调用方丢弃早于目标的帧）。"""
        seconds = max(0.0, min(seconds, self.duration))
        stream = self.vstream
        ts = int(seconds / float(stream.time_base))
        self.container.seek(ts, backward=True, stream=stream)
        if self.acontainer is not None and self.astream is not None:
            ats = int(seconds / float(self.astream.time_base))
            self.acontainer.seek(ats, backward=True, stream=self.astream)
        self._audio_acc.clear()
        self._audio_acc_pts = None
        log.debug("seek to %.3fs", seconds)

    def close(self) -> None:
        for c in (self.container, self.acontainer):
            if c is not None:
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
