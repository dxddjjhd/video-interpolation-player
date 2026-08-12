"""硬件检测模块：探测 CUDA/GPU 可用性，决定默认插帧引擎。"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """硬件探测结果。"""

    cuda_available: bool      # onnxruntime 能否加载 CUDA provider
    cuda_device: str | None   # CUDA 设备名（如 "NVIDIA GeForce RTX 3060"）
    cuda_memory_mb: int | None
    cpu_count: int
    suggested_engine: str     # "rife" | "optical" 自动决策结果
    warnings: list[str]       # 降级说明等提示信息


def _query_cuda_device() -> tuple[bool, str | None, int | None]:
    """尝试用 nvidia-smi 查询 GPU 信息；不可用时返回 (False, None, None)。"""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return False, None, None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return False, None, None
        line = out.stdout.strip().splitlines()[0]
        name, mem = line.split(",")[0].strip(), line.split(",")[1].strip()
        mem_mb = int(mem.split()[0]) if mem else None
        return True, name, mem_mb
    except Exception as e:  # noqa: BLE001
        log.debug("nvidia-smi 查询失败: %s", e)
        return False, None, None


def _ensure_nvidia_dlls() -> None:
    """把 pip 安装的 NVIDIA 库（cuDNN/cuBLAS 等）DLL 目录加入 PATH。

    onnxruntime-gpu 通过 LoadLibrary 加载 cudnn64_9.dll 等，
    该 DLL 不会随 ORT 分发，需手动安装（如 pip install nvidia-cudnn-cu12）。
    """
    import os
    import site
    added = []
    for sp in site.getsitepackages():
        base = os.path.join(sp, "nvidia")
        if not os.path.isdir(base):
            continue
        for pkg in os.listdir(base):
            d = os.path.join(base, pkg, "bin")
            if os.path.isdir(d) and d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                added.append(d)
    if added:
        log.debug("NVIDIA DLL 目录已加入 PATH: %s", added)


def detect() -> HardwareInfo:
    """执行完整硬件探测并给出引擎建议。"""
    _ensure_nvidia_dlls()
    warnings: list[str] = []

    # 1. onnxruntime 是否可用及 CUDA provider 是否可加载（最权威）
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        cuda_ok = "CUDAExecutionProvider" in providers
        if not cuda_ok:
            warnings.append(
                "未检测到可用的 CUDA 执行环境，将使用 OpenCV 光流引擎"
                "（如需 RIFE 请安装 NVIDIA 显卡驱动与 CUDA 运行库）"
            )
    except ImportError:
        cuda_ok = False
        warnings.append("onnxruntime 未安装，将使用 OpenCV 光流引擎")

    # 2. nvidia-smi 补充设备名信息（仅作展示）
    gpu_name, mem_mb = None, None
    smi_ok, gpu_name, mem_mb = _query_cuda_device()
    if cuda_ok and not smi_ok:
        warnings.append("检测到 CUDA 环境，但无法读取 GPU 型号详情")

    import os
    cpu_count = os.cpu_count() or 1

    return HardwareInfo(
        cuda_available=cuda_ok,
        cuda_device=gpu_name,
        cuda_memory_mb=mem_mb,
        cpu_count=cpu_count,
        suggested_engine="rife" if cuda_ok else "optical",
        warnings=warnings,
    )


def print_report(info: HardwareInfo) -> None:
    """打印硬件报告到控制台。"""
    print("=" * 46)
    print("  硬件检测报告")
    print("=" * 46)
    if info.cuda_available:
        print(f"  CUDA 加速 : 可用  {info.cuda_device or ''}"
              f"{f'  ({info.cuda_memory_mb} MB)' if info.cuda_memory_mb else ''}")
    else:
        print("  CUDA 加速 : 不可用")
    print(f"  CPU 核心  : {info.cpu_count}")
    print(f"  建议引擎  : {'RIFE (AI 插帧)' if info.suggested_engine == 'rife' else 'OpenCV 光流'}")
    for w in info.warnings:
        print(f"  [!] {w}")
    print("=" * 46)
