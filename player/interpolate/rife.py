"""RIFE 深度学习插帧引擎（ONNX Runtime）。

支持三类 ONNX 输入结构（加载时自动识别）：
1. vs-mlrt 风格单输入 [1, 11, H, W]（RIFE v4.0~v4.10 官方导出）:
   通道 = [img0_rgb(3), img1_rgb(3), timestep(1), X网格(1), Y网格(1),
           2/(W-1)(1), 2/(H-1)(1)]，帧归一化到 [0,1]
2. vs-mlrt 新风格单输入 [1, 7, H, W]（v4.12+）: 无坐标通道
3. 通用三输入 img0/img1/timestep（社区导出）

CUDA 优先、CPU 兜底；H/W 需 32 对齐（pad 后推理、crop 还原）。
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from .base import Interpolator

log = logging.getLogger(__name__)

# RIFE 网络 5 级下采样，要求 H/W 为 32 的倍数
ALIGN = 32

# CUDA 可用性缓存：首次创建会话失败（虚拟 GPU 上 DLL 加载不稳定）
# 后不再重复尝试，避免每次建会话都刷 ORT 警告
_cuda_ok: bool | None = None


def _suppress_ort_logger() -> None:
    """屏蔽 onnxruntime C++ 默认日志（直接写 stderr 的 WARNING 噪音）。"""
    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(3)   # 3 = ERROR，仅保留真正的错误
    except Exception:  # noqa: BLE001
        pass


_suppress_ort_logger()


class RifeInterpolator(Interpolator):
    name = "RIFE (AI)"

    def __init__(self, model_path: str, use_cuda: bool = True) -> None:
        import onnxruntime as ort

        global _cuda_ok
        self.model_path = model_path
        providers = []
        if use_cuda and _cuda_ok is not False:
            try:
                # 会话创建成功即视为 CUDA 可用（含 DLL 加载校验）
                sess = ort.InferenceSession(
                    model_path,
                    providers=[("CUDAExecutionProvider", {
                        "device_id": 0,
                        "arena_extend_strategy": "kSameAsRequested",
                    }), "CPUExecutionProvider"])
                self._sess = sess
                _cuda_ok = True
            except Exception as e:  # noqa: BLE001
                log.warning("CUDA 会话创建失败，本次及后续会话使用 CPU: %s", e)
                _cuda_ok = False
                self._sess = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"],
                    sess_options=self._options())
        else:
            self._sess = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"],
                sess_options=self._options())
        self._io = self._inspect_io()
        # 输入构造缓存：网格/常量按 (h,w) 缓存，避免每帧重建
        self._grid_cache: dict[tuple[int, int], np.ndarray] = {}
        self._inp_cache: tuple[tuple[int, int], np.ndarray] | None = None

    @staticmethod
    def _options():
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1      # 推理线程由 CUDA 流管理
        return so

    # ------------------------------------------------------------------
    def _inspect_io(self) -> dict:
        inputs = self._sess.get_inputs()
        names = [i.name for i in inputs]
        shapes = [i.shape for i in inputs]
        log.debug("RIFE ONNX inputs: %s %s", names, shapes)
        out_name = self._sess.get_outputs()[0].name
        io: dict = {"out": out_name, "mode": None}
        # 任何多输入模式下都可能有名为 timestep/time 的标量输入
        io["has_ts"] = any("time" in n.lower() for n in names)
        if len(inputs) == 1:
            ch = shapes[0][1] if len(shapes[0]) > 1 else 0
            if ch == 6:
                io["mode"] = "concat"          # [1,6,H,W] 两帧拼接
            elif ch >= 7:
                # vs-mlrt 风格: [1,7,H,W] 无坐标 / [1,11,H,W] 含坐标网格
                io["mode"] = "vsmlrt"
                io["has_coords"] = ch >= 11
            else:
                raise RuntimeError(f"无法识别的单输入通道数: {ch}")
            io["img_in"] = names[0]
        elif len(inputs) == 2:
            io["mode"] = "pair"
            io["img0"] = names[0]
            io["img1"] = names[1]
        elif len(inputs) == 3:
            io["mode"] = "pair_ts"
            io["img0"], io["img1"] = names[0], names[1]
            io["ts"] = names[2]
        else:
            raise RuntimeError(f"无法识别的 RIFE 模型输入结构: {names}")
        return io

    # ------------------------------------------------------------------
    def _prep(self, frame: np.ndarray) -> np.ndarray:
        """pad 到 32 对齐并归一化，返回 (1,3,H,W) float32。"""
        h, w = frame.shape[:2]
        ph = (-h) % ALIGN
        pw = (-w) % ALIGN
        if ph or pw:
            frame = np.pad(frame, ((0, ph), (0, pw), (0, 0)),
                           mode="edge")
        x = frame.astype(np.float32) / 255.0
        return x.transpose(2, 0, 1)[None, ...]

    def _vsmlrt_input(self, img0: np.ndarray, img1: np.ndarray,
                      t: float) -> np.ndarray:
        """构造 vs-mlrt 风格输入: [img0, img1, mask, X, Y, 2/(W-1), 2/(H-1)]。

        网格与常量通道按 (h,w) 缓存（帧内容通道原位写入），
        避免每次推理重建 ~40MB 缓冲区（实测可省 30-40ms/对）。
        """
        _, _, h, w = img0.shape
        key = (h, w)
        cached = self._grid_cache.get(key)
        if cached is None:
            x_grid = (2.0 * np.arange(w, dtype=np.float32) / (w - 1)) - 1.0
            y_grid = (2.0 * np.arange(h, dtype=np.float32) / (h - 1)) - 1.0
            if self._io.get("has_coords", False):
                cached = np.concatenate([
                    np.broadcast_to(x_grid[None, None, :], (1, h, w)),
                    np.broadcast_to(y_grid[None, :, None], (1, h, w)),
                    np.full((1, h, w), np.float32(2.0 / (w - 1))),
                    np.full((1, h, w), np.float32(2.0 / (h - 1))),
                ], axis=0).astype(np.float32)
            else:
                cached = np.zeros((0, h, w), dtype=np.float32)
            self._grid_cache[key] = cached
        # 掩码通道 + 网格
        if self._io.get("has_coords", False):
            extra = np.concatenate([
                np.full((1, h, w), np.float32(t), dtype=np.float32), cached],
                axis=0)
        else:
            extra = np.full((1, h, w), np.float32(t), dtype=np.float32)
        inp = np.concatenate([img0[0], img1[0], extra], axis=0)[None, ...]
        return np.ascontiguousarray(inp)

    def _run(self, img0: np.ndarray, img1: np.ndarray, t: float) -> np.ndarray:
        io = self._io
        if io["mode"] == "vsmlrt":
            feeds = {io["img_in"]: self._vsmlrt_input(img0, img1, t)}
        elif io["mode"] == "concat":
            x = np.concatenate([img0, img1], axis=1)
            feeds: dict = {io["img_in"]: x}
            if io["has_ts"]:
                feeds["timestep"] = np.array([t], dtype=np.float32)
        elif io["mode"] == "pair":
            feeds = {io["img0"]: img0, io["img1"]: img1}
            if "ts" in io:
                feeds[io["ts"]] = np.array([t], dtype=np.float32)
        else:
            feeds = {io["img0"]: img0, io["img1"]: img1,
                     io["ts"]: np.array([t], dtype=np.float32)}
        out = self._sess.run([io["out"]], feeds)[0][0]   # (3,H,W)
        return out

    # ------------------------------------------------------------------
    def interpolate(self, frame0, frame1, times: Sequence[float]) -> list[np.ndarray]:
        if not times:
            return []
        h, w = frame0.shape[:2]
        i0 = self._prep(frame0)
        i1 = self._prep(frame1)
        out: list[np.ndarray] = []
        for t in times:
            r = self._run(i0, i1, float(t))
            r = r[:, :h, :w]
            r = (np.clip(r, 0.0, 1.0) * 255.0).astype(np.uint8)
            out.append(r.transpose(1, 2, 0))
        return out

    def warmup(self) -> None:
        """用最小合法尺寸跑一次空推理，触发 CUDA 上下文初始化。"""
        try:
            z = np.zeros((1, 3, ALIGN, ALIGN), dtype=np.float32)
            io = self._io
            if io["mode"] == "vsmlrt":
                feeds = {io["img_in"]: self._vsmlrt_input(z, z, 0.5)}
            elif io["mode"] == "concat":
                x = np.concatenate([z, z], axis=1)
                feeds = {io["img_in"]: x}
                if io["has_ts"]:
                    feeds["timestep"] = np.array([0.5], dtype=np.float32)
            elif io["mode"] == "pair":
                feeds = {io["img0"]: z, io["img1"]: z}
                if "ts" in io:
                    feeds[io["ts"]] = np.array([0.5], dtype=np.float32)
            else:
                feeds = {io["img0"]: z, io["img1"]: z,
                         io["ts"]: np.array([0.5], dtype=np.float32)}
            self._sess.run([io["out"]], feeds)
            log.info("RIFE 预热完成")
        except Exception as e:  # noqa: BLE001
            log.warning("RIFE 预热失败（可能影响首次插帧延迟）: %s", e)

    def close(self) -> None:
        self._sess = None
