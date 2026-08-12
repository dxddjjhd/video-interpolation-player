"""RealESRGAN ONNX 模型下载与定位。

来源：AmusementClub/vs-mlrt 模型发布（model-20211209，animevideo 系列）。
模型很小（总计约 6.5MB），自动下载带断点续传与重试。
也支持手动放置任意 realesrgan*.onnx 到 models/ 目录。
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

# 2x / 4x 模型（vs-mlrt 的 animevideo 小模型，质量/速度均衡）
MODEL_2X_NAME = "RealESRGANv2-animevideo-xsx2.onnx"
MODEL_4X_NAME = "RealESRGANv2-animevideo-xsx4.onnx"
ARCHIVES = {
    "RealESRGANv2_v1.7z": (
        "https://github.com/AmusementClub/vs-mlrt/releases/download/"
        "model-20211209/RealESRGANv2_v1.7z",
        (MODEL_2X_NAME, MODEL_4X_NAME),
    ),
}

MODELS_DIR = Path(__file__).resolve().parent / "models"


def find_model(factor: int = 2) -> Path | None:
    """按倍率查找已就绪的 ONNX 模型。"""
    want = MODEL_2X_NAME if factor == 2 else MODEL_4X_NAME
    d = MODELS_DIR
    if not d.is_dir():
        return None
    exact = d / want
    if exact.exists():
        return exact
    for name in sorted(os.listdir(d)):
        if name.lower().endswith(".onnx") and ("xsx" in name.lower()
                                               or "animevideo" in name.lower()):
            if (factor == 2 and "xsx2" in name.lower()) or \
               (factor == 4 and "xsx4" in name.lower()):
                return d / name
    return None


def ensure_model(factor: int = 2, progress_cb=None) -> Path | None:
    """确保模型就绪：已存在直接返回；否则下载并解压。失败返回 None。"""
    existing = find_model(factor)
    if existing is not None:
        return existing

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for arc_name, (url, targets) in ARCHIVES.items():
            arc = MODELS_DIR / arc_name
            if arc.exists() and arc.stat().st_size > 100_000:
                pass
            else:
                _download(url, arc, progress_cb)
            _extract(arc, targets)
    except Exception as e:  # noqa: BLE001
        log.error("RealESRGAN 模型获取失败: %s", e)
        if progress_cb:
            progress_cb(f"模型下载失败: {e}")
        return None
    return find_model(factor)


def _download(url: str, target: Path, progress_cb=None) -> None:
    """下载（requests + 重试 + 断点续传），模型小一般秒级完成。"""
    import requests

    existing = target.stat().st_size if target.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    for attempt in range(15):
        try:
            with requests.get(url, stream=True, timeout=60,
                              headers=headers) as r:
                if r.status_code == 416:
                    return
                r.raise_for_status()
                mode = "ab" if existing else "wb"
                with open(target, mode) as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
                if progress_cb:
                    progress_cb(f"模型下载完成: {target.name}")
                return
        except Exception as e:  # noqa: BLE001
            log.warning("下载 %s 第 %d 次失败: %s", target.name, attempt + 1,
                        str(e)[:80])
            time.sleep(3)
    raise RuntimeError(f"下载失败: {target.name}")


def _extract(archive: Path, targets: tuple[str, ...]) -> None:
    """解压 7z 并扁平化 ONNX 文件。"""
    import py7zr
    import shutil

    with py7zr.SevenZipFile(str(archive)) as z:
        names = z.getnames()
        onnx_names = [n for n in names if n.lower().endswith(".onnx")]
        z.extract(path=str(MODELS_DIR), targets=onnx_names)
    archive.unlink(missing_ok=True)
    for p in sorted(MODELS_DIR.rglob("*.onnx")):
        if p.parent != MODELS_DIR:
            dest = MODELS_DIR / p.name
            if not dest.exists():
                shutil.move(str(p), str(dest))
