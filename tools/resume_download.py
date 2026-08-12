"""GitHub Release 分块下载器（按连接限速时可用，支持断点续传）。

策略：8 个分块按序下载（CDN 对并发连接限速，串行最稳），
每分块用 curl Range 请求 + 本地已下载长度计算续传偏移，
失败自动重试；中断后重新运行可从中断处继续。

用法: python resume_download.py <输出路径> [连接数=8]
"""
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = "AmusementClub/vs-mlrt"
TAG = "model-20220923"
ASSET = "rife_v8.7z"
CONNS = 8

API_URL = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
DL_URL = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"


def get_asset_size() -> int:
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    for a in r.json()["assets"]:
        if a["name"] == ASSET:
            return a["size"]
    raise RuntimeError("资产未找到")


def download_chunk(chunk_idx: int, start: int, end: int, part: Path,
                   progress_cb) -> bool:
    """下载单个分块（curl Range + 外部续传偏移），无限重试。"""
    expect = end - start + 1
    tmp = part.with_suffix(part.suffix + ".tmp")
    while True:
        done = part.stat().st_size if part.exists() else 0
        if done >= expect:
            return True
        r = subprocess.run(
            ["curl", "-s", "-o", str(tmp),
             "-r", f"{start + done}-{end}", "-L",
             "--connect-timeout", "30",
             "--speed-limit", "2048", "--speed-time", "60",   # 低于2KB/s持续60s视为死连接
             "--max-time", "900",
             DL_URL],
            timeout=1000)
        # curl -o 会截断，因此先写临时文件再追加到 part（断点续传不丢数据）
        if tmp.exists():
            with open(part, "ab") as dst, open(tmp, "rb") as src:
                dst.write(src.read())
            tmp.unlink()
        done = part.stat().st_size if part.exists() else 0
        progress_cb(chunk_idx, done, expect)
        if r.returncode == 0 and done >= expect:
            return True
        time.sleep(3)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else Path(__file__).parent.parent / "player/interpolate/models/rife_v8.7z"
    total = get_asset_size()
    print(f"总大小: {total / 1024 / 1024:.1f} MB")
    out.parent.mkdir(parents=True, exist_ok=True)

    chunk = total // CONNS
    t0 = time.monotonic()

    def progress_cb(idx, done, expect):
        print(f"  [{time.monotonic() - t0:.0f}s] 分块{idx}: "
              f"{done / 1024 / 1024:.1f}/{expect / 1024 / 1024:.1f}MB "
              f"({done / max(expect, 1) * 100:.0f}%)", flush=True)

    # 按序下载全部分块（已有的自动跳过/续传）
    for i in range(CONNS):
        start = i * chunk
        end = (i + 1) * chunk - 1 if i < CONNS - 1 else total - 1
        part = out.with_suffix(out.suffix + f".part{i}")
        print(f"分块 {i}: [{start / 1024 / 1024:.0f}MB, {end / 1024 / 1024:.0f}MB]", flush=True)
        download_chunk(i, start, end, part, progress_cb)

    # 合并
    with open(out, "wb") as dst:
        for i in range(CONNS):
            part = out.with_suffix(out.suffix + f".part{i}")
            with open(part, "rb") as src:
                dst.write(src.read())
            part.unlink()
    print(f"✅ 完成: {out} ({out.stat().st_size / 1024 / 1024:.1f}MB) "
          f"总耗时 {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
