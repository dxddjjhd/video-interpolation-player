"""插帧引擎工厂：按用户选择与硬件条件创建引擎，带自动降级。"""

from __future__ import annotations

import logging
from pathlib import Path

from ..utils.hardware import HardwareInfo
from .base import Interpolator
from .optical_flow import OpticalFlowInterpolator

log = logging.getLogger(__name__)

ENGINE_CHOICES = ("auto", "rife", "optical")


def create_interpolator(
    kind: str,
    hw: HardwareInfo,
    model_dir: Path | None = None,
    model_ready: bool = False,
) -> Interpolator:
    """创建插帧引擎。

    kind: "auto" | "rife" | "optical"
    model_ready: RIFE 模型文件是否已就绪（由 UI 层在下载后置位）
    """
    if kind not in ENGINE_CHOICES:
        kind = "auto"

    want_rife = (kind == "rife") or (kind == "auto" and hw.suggested_engine == "rife")

    if want_rife and not model_ready and kind == "rife":
        # 用户显式选择 RIFE 但模型缺失：明确报错而非静默降级
        raise RuntimeError(
            "RIFE 模型未就绪（模型下载中或失败），可稍后再试或切换光流引擎")

    if want_rife and model_ready:
        try:
            from .rife import RifeInterpolator
            model_path = _locate_model(model_dir)
            if model_path is None:
                raise FileNotFoundError("models/ 目录下没有 .onnx 模型文件")
            eng = RifeInterpolator(str(model_path), use_cuda=hw.cuda_available)
            log.info("使用引擎: %s (%s)", eng.name, model_path.name)
            return eng
        except Exception as e:  # noqa: BLE001
            if kind == "rife":
                # 用户强制 RIFE 但加载失败——抛出并让 UI 提示，避免静默降级
                raise RuntimeError(f"RIFE 引擎加载失败: {e}") from e
            log.warning("RIFE 加载失败(%s)，自动降级到光流引擎", e)

    log.info("使用引擎: %s", OpticalFlowInterpolator.name)
    return OpticalFlowInterpolator()


def _locate_model(model_dir: Path | None) -> Path | None:
    from .download_model import find_model
    return find_model(model_dir)
