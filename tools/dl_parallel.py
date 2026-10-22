"""并行分块下载器（8 连接）。

实测：魔搭对**单连接**限速到 ~0.14 MiB/s，但 8 连接并行能到 ~10 MiB/s。
所以大文件必须分段并发拉。

特性：
- 每块 8 MiB，多线程并发，各自 Range 请求；
- 每块下完立即落盘（应用层续传：`<file>.part` + `<file>.part.json` 记录已完成块）；
- 中断后重跑会跳过已完成的块；

用法：
    python tools/dl_parallel.py            # 默认清单
    python tools/dl_parallel.py Qwen3-1.7B
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "models"
CHUNK = 8 << 20
WORKERS = 8

BASE = "https://modelscope.cn/models/Qwen/{repo}/resolve/master/{path}"

MANIFEST: dict[str, list[str]] = {
    "Qwen3-0.6B": [
        "config.json", "generation_config.json", "merges.txt", "vocab.json",
        "tokenizer.json", "tokenizer_config.json", "model.safetensors",
    ],
    "Qwen3-1.7B": [
        "config.json", "generation_config.json", "merges.txt", "vocab.json",
        "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json",
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    ],
}

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        cr = resp.headers.get("Content-Range") or ""
        if "/" in cr:
            return int(cr.rsplit("/", 1)[1])
        return int(resp.headers.get("Content-Length") or 0)


def fetch_chunk(url: str, fh, offset: int, length: int, retries: int = 8) -> None:
    want = f"bytes={offset}-{offset + length - 1}"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Range": want})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) != length:
                raise IOError(f"块长度不符 {len(data)} != {length}")
            fh.seek(offset)
            fh.write(data)
            return
        except Exception as exc:
            if attempt == retries:
                raise
            log(f"      ! 块 @{offset >> 20}MiB 第{attempt}次失败: {exc}")
            time.sleep(min(2 * attempt, 10))


def download_file(url: str, dest: Path) -> None:
    total = remote_size(url)
    if total <= 0:
        raise IOError("拿不到文件大小")
    part = dest.with_suffix(dest.suffix + ".part")
    meta_p = Path(str(part) + ".json")

    done: set[int] = set()
    if part.exists() and meta_p.exists():
        try:
            info = json.loads(meta_p.read_text())
            if info.get("total") == total:
                done = set(info.get("done", []))
                log(f"    续传: 已有 {len(done)} 块")
        except Exception:
            done = set()

    nchunks = (total + CHUNK - 1) // CHUNK
    jobs = [i for i in range(nchunks) if i not in done]
    if not jobs:
        os.replace(part, dest)
        log(f"    {dest.name}: 已完整")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(part, "r+b" if part.exists() else "wb") as fh:
        fh.truncate(total)
        t0 = time.time()
        base_have = len(done) * CHUNK
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {}
            for i in jobs:
                off = i * CHUNK
                ln = min(CHUNK, total - off)
                futs[ex.submit(fetch_chunk, url, fh, off, ln)] = i
            finished = 0
            for fut in futs:
                fut.result()          # 抛错即整体失败，保留已完成块供续传
                finished += 1
                done.add(futs[fut])
                if finished % 8 == 0 or finished == len(jobs):
                    got = base_have + finished * CHUNK
                    rate = (got - base_have) / max(time.time() - t0, 1e-6)
                    log(f"    {dest.name}: {min(got, total) / 2**20:.0f}/"
                        f"{total / 2**20:.0f} MiB  {rate / 2**20:.1f} MiB/s")
                    meta_p.write_text(json.dumps({"total": total, "done": sorted(done)}))
    meta_p.write_text(json.dumps({"total": total, "done": sorted(done)}))
    os.replace(part, dest)
    meta_p.unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    repos = argv[1:] or list(MANIFEST)
    for repo in repos:
        log(f"== {repo} ==")
        for name in MANIFEST[repo]:
            dest = ROOT / repo / name
            if dest.exists() and dest.stat().st_size > 0:
                log(f"  skip {name}")
                continue
            url = BASE.format(repo=repo, path=name)
            log(f"  {name}")
            try:
                download_file(url, dest)
            except Exception as exc:
                log(f"  !! {name}: {exc}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))