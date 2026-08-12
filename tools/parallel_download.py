"""GitHub Release 并行分块下载器（CDN 按连接限速时提速）。

用法: python parallel_download.py <url> <输出路径> [连接数=8]
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def get_size(url: str) -> int:
    """通过 GitHub API 获取 release 资产大小（CDN HEAD 常无 content-length）。"""
    if "api.github.com" not in url:
        # 尝试 HEAD 的 content-length
        out = subprocess.run(
            ["curl", "-sI", "-L", url], capture_output=True, text=True, timeout=30)
        for line in out.stdout.splitlines():
            if line.lower().startswith("content-length:"):
                return int(line.split(":", 1)[1].strip())
        raise RuntimeError("无法获取文件大小")
    import json
    out = subprocess.run(["curl", "-s", url], capture_output=True, text=True,
                         timeout=30)
    d = json.loads(out.stdout)
    if "assets" in d:
        # release 对象: 取第一个匹配名称的资产
        return max(a["size"] for a in d["assets"])
    return d.get("size", 0)


def release_asset_url(repo: str, tag: str, asset: str) -> str:
    """拼 GitHub API 资产查询 URL 与下载 URL。"""
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    dl = (f"https://github.com/{repo}/releases/download/{tag}/{asset}")
    return api, dl


def main() -> None:
    url, out_path = sys.argv[1], sys.argv[2]
    conns = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out = Path(out_path)
    if url.startswith("github:"):
        # github:owner/repo:tag:asset 简写
        _, repo, tag, asset = url.split(":")
        api_url, url = release_asset_url(repo, tag, asset)
        total = get_size(api_url)
    else:
        total = get_size(url)
    print(f"总大小: {total / 1024 / 1024:.1f} MB, 连接数: {conns}")

    chunk = total // conns
    results: list[tuple[int, int]] = []   # (byte_offset, exit_code)
    lock = threading.Lock()

    def worker(i: int) -> None:
        start = i * chunk
        end = (i + 1) * chunk - 1 if i < conns - 1 else total - 1
        part = out.with_suffix(out.suffix + f".part{i}")
        r = subprocess.run(
            ["curl", "-s", "-C", "-", "-o", str(part),
             "-r", f"{start}-{end}", "-L", "--retry", "1000",
             "--retry-delay", "3", "--retry-all-errors", url],
            timeout=7200)
        got = part.stat().st_size if part.exists() else 0
        with lock:
            results.append((i, 0 if (r.returncode == 0 and got == end - start + 1) else 1))
            print(f"  分块 {i}: {got / 1024 / 1024:.1f}MB (期望 {(end - start + 1) / 1024 / 1024:.1f}MB)", flush=True)

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(conns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if any(rc != 0 for _, rc in results):
        print("❌ 部分分块失败")
        sys.exit(1)

    with open(out, "wb") as dst:
        for i in range(conns):
            part = out.with_suffix(out.suffix + f".part{i}")
            with open(part, "rb") as src:
                dst.write(src.read())
            part.unlink()

    elapsed = time.monotonic() - t0
    print(f"✅ 合并完成: {out} ({out.stat().st_size / 1024 / 1024:.1f}MB) "
          f"耗时 {elapsed:.0f}s ({out.stat().st_size / elapsed / 1024:.0f} KB/s)")


if __name__ == "__main__":
    main()
