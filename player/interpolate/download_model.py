"""RIFE ONNX 模型下载与定位。

模型来源：AmusementClub/vs-mlrt 的模型发布包（rife_v8.7z，
内含 rife_v4.0 ~ v4.10 官方 ONNX 导出，MIT 许可）。
首次使用时自动下载（约 190MB）并解压出所需版本。

也支持手动放置：把任意 rife*.onnx 放进 models/ 目录即可跳过下载。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# 下载源与目标模型
MODEL_ARCHIVE_URL = (
    "https://github.com/AmusementClub/vs-mlrt/releases/download/"
    "model-20220923/rife_v8.7z"
)
MODEL_ARCHIVE_NAME = "rife_v8.7z"
PREFERRED_MODEL = "rife_v4.6.onnx"

MODELS_DIR = Path(__file__).resolve().parent / "models"


def find_model(model_dir: Path | None = None) -> Path | None:
    """在 models/ 目录中查找可用的 RIFE ONNX 模型。

    优先使用 rife_v4.6.onnx（画质/速度均衡的经典版本），
    其次按版本号从高到低选择，最后任意 .onnx。
    """
    d = Path(model_dir) if model_dir else MODELS_DIR
    if not d.is_dir():
        return None
    onnx_files = sorted(
        (d / name for name in os.listdir(d) if name.lower().endswith(".onnx")),
        key=lambda p: p.name.lower())
    if not onnx_files:
        return None
    preferred = next((p for p in onnx_files
                      if p.name.lower() == "rife_v4.6.onnx"), None)
    if preferred is not None:
        return preferred
    # 按 (v4, 主版本, 次版本) 从高到低
    def ver_key(p: Path):
        import re
        m = re.search(r"v?4[._](\d+)", p.name.lower())
        return (int(m.group(1)) if m else 0, p.name.lower())
    return max(onnx_files, key=ver_key)


def ensure_model(model_dir: Path | None = None,
                 progress_cb=None) -> Path | None:
    """确保模型就绪：已存在直接返回；否则下载并解压。失败返回 None。"""
    d = Path(model_dir) if model_dir else MODELS_DIR
    existing = find_model(d)
    if existing is not None:
        return existing

    d.mkdir(parents=True, exist_ok=True)
    archive = d / MODEL_ARCHIVE_NAME
    try:
        _download(archive, progress_cb)
        _extract(archive, d)
        archive.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        log.error("RIFE 模型获取失败: %s", e)
        if progress_cb:
            progress_cb(f"模型下载失败: {e}")
        return None
    return find_model(d)


def _download(target: Path, progress_cb=None) -> None:
    """下载模型包，支持断点续传（已有部分文件则从断点继续）。"""
    import requests

    existing = target.stat().st_size if target.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    log.info("开始下载 RIFE 模型包 (~190MB)%s: %s",
             f"，续传 {existing // (1 << 20)}MB" if existing else "",
             MODEL_ARCHIVE_URL)
    if progress_cb:
        msg = f"正在下载 RIFE 模型包 (~190MB)..."
        if existing:
            msg += f" 已下载 {existing // (1 << 20)}MB，继续中"
        progress_cb(msg)
    with requests.get(MODEL_ARCHIVE_URL, stream=True, timeout=60,
                      headers=headers) as r:
        if r.status_code == 416:
            log.info("模型包已完整存在，跳过下载")
            return
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + existing
        mode = "ab" if existing else "wb"
        done = existing
        with open(target, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                done += len(chunk)
                if progress_cb and total:
                    progress_cb(f"下载中 {done // (1 << 20)}/{total // (1 << 20)} MB")
    log.info("模型包下载完成: %s", target)


def _extract(archive: Path, out_dir: Path) -> None:
    """解压 7z，只取出 ONNX 模型文件。"""
    import py7zr

    with py7zr.SevenZipFile(str(archive)) as z:
        names = z.getnames()
        onnx_names = [n for n in names if n.lower().endswith(".onnx")]
        log.info("模型包内含 %d 个 ONNX 模型", len(onnx_names))
        targets = onnx_names
        z.extract(path=str(out_dir), targets=targets)
    # 扁平化：把子目录里的 onnx 移到 models/ 根目录
    for p in sorted(out_dir.rglob("*.onnx")):
        if p.parent != out_dir:
            dest = out_dir / p.name
            if not dest.exists():
                p.rename(dest)
