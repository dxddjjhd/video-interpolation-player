"""RealESRGAN 超分引擎（ONNX Runtime）。

模型：vs-mlrt 导出的 animevideo 系列（输入 [0,1] fp32 [1,3,H,W]，
输出 [1,3,H*s,W*s]，值可能略超 [0,1] 需 clamp）。CUDA 优先（带
失败缓存，与 RIFE 引擎一致），CPU 兜底。

输入尺寸会 pad 到 4 的倍数（RealESRGAN 内部上采样要求）。
"""

from __future__ import annotations

import logging

import numpy as np

from .base import SuperRes

log = logging.getLogger(__name__)

ALIGN = 4

# CUDA 可用性缓存（与 rife 模块共享判断逻辑，但各自独立探测一次）
_cuda_ok: bool | None = None


def _suppress_ort_logger() -> None:
    """屏蔽 onnxruntime C++ 默认日志噪音。"""
    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(3)   # 3 = ERROR
    except Exception:  # noqa: BLE001
        pass


_suppress_ort_logger()


class RealEsrganSuperRes(SuperRes):
    name = "RealESRGAN"

    def __init__(self, model_path: str, factor: int, use_cuda: bool = True) -> None:
        import onnxruntime as ort

        global _cuda_ok
        self.model_path = model_path
        self.factor = factor
        providers = []
        if use_cuda and _cuda_ok is not False:
            try:
                self._sess = ort.InferenceSession(
                    model_path,
                    providers=[("CUDAExecutionProvider", {
                        "device_id": 0,
                        "arena_extend_strategy": "kSameAsRequested",
                    }), "CPUExecutionProvider"])
                _cuda_ok = True
            except Exception as e:  # noqa: BLE001
                log.warning("RealESRGAN CUDA 会话创建失败，改用 CPU: %s", e)
                _cuda_ok = False
                self._sess = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"])
        else:
            self._sess = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
        self._in_name = self._sess.get_inputs()[0].name
        self._out_name = self._sess.get_outputs()[0].name

    # ------------------------------------------------------------------
    def upscale(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        ph = (-h) % ALIGN
        pw = (-w) % ALIGN
        if ph or pw:
            frame = np.pad(frame, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        x = frame.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None, ...]          # (1,3,H,W)
        out = self._sess.run([self._out_name], {self._in_name: x})[0][0]
        out = out[:, :h * self.factor, :w * self.factor]
        out = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.ascontiguousarray(out.transpose(1, 2, 0))

    def warmup(self) -> None:
        """用最小尺寸跑一次空推理，触发 CUDA 上下文/算子初始化。"""
        try:
            z = np.zeros((1, 3, ALIGN, ALIGN), dtype=np.float32)
            self._sess.run([self._out_name], {self._in_name: z})
            log.info("RealESRGAN 预热完成")
        except Exception as e:  # noqa: BLE001
            log.warning("RealESRGAN 预热失败: %s", e)

    def close(self) -> None:
        self._sess = None
