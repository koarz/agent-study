"""管理精选本地模型的下载、断点续传和校验，不接受任意下载地址或远程代码。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import httpx


MODEL_KEY = "qwen3-reranker-0.6b"
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"


def model_path(directory, model=MODEL_ID):
    if model != MODEL_ID:
        raise ValueError("当前下载目录只提供 Qwen3-Reranker-0.6B")
    path = Path(directory) / MODEL_KEY
    manifest = path / "ready.json"
    if not manifest.is_file():
        raise ValueError("本地重排模型尚未下载完成，请在模型设置中先下载模型")
    try:
        values = json.loads(manifest.read_text())
        required = {"model.safetensors", "config.json", "tokenizer_config.json", "tokenizer.json"}
        if values["revision"] != REVISION or not required.issubset(values["files"]) or not all(
                (path / name).stat().st_size == size for name, size in values["files"].items()):
            raise ValueError("模型文件不完整，请重新下载")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("模型文件不完整，请重新下载") from exc
    return path


@lru_cache(maxsize=1)
def runtime_capabilities():
    """缓存实际运行设备信息，不能仅凭系统装了驱动就宣称 GPU 可用。"""
    if importlib.util.find_spec("torch") is None:
        return {"cuda_available": False, "runtime_device": "cpu", "gpu_name": None}
    try:
        import torch
        available = torch.cuda.is_available()
        return {"cuda_available": available, "runtime_device": "cuda" if available else "cpu",
                "gpu_name": torch.cuda.get_device_name(0) if available else None}
    except (ImportError, OSError, RuntimeError):
        return {"cuda_available": False, "runtime_device": "cpu", "gpu_name": None}


class LocalModelManager:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-download")
        self.state = {"status": "not_downloaded", "stage": "尚未下载", "downloaded_bytes": 0, "total_bytes": 0}
        self.state_path = self.directory / "download.json"
        if self.state_path.is_file():
            try:
                self.state.update(json.loads(self.state_path.read_text()))
            except (ValueError, OSError):
                pass
        if self.state["status"] == "downloading":
            self.state.update(status="interrupted", stage="下载已中断，可继续下载", eta_seconds=None)

    def snapshot(self):
        with self.lock:
            row = dict(self.state)
        try:
            model_path(self.directory)
            row.update(status="ready", stage="已下载，可在本机重排", percent=100, eta_seconds=0)
        except ValueError:
            if row["status"] == "ready":
                row.update(status="not_downloaded", stage="模型文件不完整，请重新下载")
        row.update(id=MODEL_KEY, model=MODEL_ID, revision=REVISION,
                   dependencies_ready=all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers")))
        row.update(runtime_capabilities())
        return row

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.state_path)

    def start(self):
        with self.lock:
            row = self.snapshot()
            if row["status"] in {"downloading", "ready"}:
                return row
            self.update(status="downloading", stage="正在获取模型文件清单", error=None, percent=0, eta_seconds=None)
            self.pool.submit(self.download)
            return self.snapshot()

    def download(self):
        started = time.monotonic()
        transferred, last_update = 0, 0
        destination = self.directory / MODEL_KEY
        destination.mkdir(parents=True, exist_ok=True)
        try:
            # 固定官方仓库和版本，只下载数据文件，禁止 Python 远程代码。
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, connect=20)) as client:
                response = client.get(f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}", params={"blobs": "true"})
                response.raise_for_status()
                rows = response.json()["siblings"]
                files = [row for row in rows if Path(row["rfilename"]).suffix in {".json", ".txt", ".jinja", ".safetensors"}
                         and not Path(row["rfilename"]).is_absolute() and ".." not in Path(row["rfilename"]).parts]
                if not files or not any(row["rfilename"] == "model.safetensors" for row in files):
                    raise ValueError("官方仓库未提供有效模型文件")
                total = sum(row["size"] for row in files)
                completed = 0
                for row in files:
                    if self.stopping.is_set():
                        raise InterruptedError()
                    name, size = row["rfilename"], row["size"]
                    path = destination / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    expected = row.get("lfs", {}).get("sha256") if row.get("lfs") else row.get("blobId")
                    def valid(candidate):
                        if not candidate.is_file() or candidate.stat().st_size != size:
                            return False
                        algorithm = hashlib.sha256() if row.get("lfs") else hashlib.sha1(f"blob {size}\0".encode())
                        with candidate.open("rb") as reader:
                            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                                algorithm.update(chunk)
                        return expected is not None and algorithm.hexdigest() == expected
                    if valid(path):
                        completed += size
                        continue
                    partial = path.with_suffix(path.suffix + ".part")
                    offset = partial.stat().st_size if partial.exists() else 0
                    if offset > size:
                        partial.unlink()
                        offset = 0
                    self.update(stage="下载 " + name, downloaded_bytes=completed + offset, total_bytes=total)
                    for attempt in range(3):
                        if self.stopping.is_set():
                            raise InterruptedError()
                        offset = partial.stat().st_size if partial.exists() else 0
                        if offset == size:
                            break
                        try:
                            headers = {"Range": f"bytes={offset}-"} if offset else {}
                            with client.stream("GET", f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}", headers=headers) as download:
                                download.raise_for_status()
                                if offset and download.status_code != 206:
                                    offset = 0
                                if offset and not download.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                                    raise ValueError("模型下载的续传范围不一致")
                                with partial.open("ab" if offset else "wb") as writer:
                                    for chunk in download.iter_bytes(1024 * 1024):
                                        if self.stopping.is_set():
                                            raise InterruptedError()
                                        writer.write(chunk)
                                        offset += len(chunk)
                                        transferred += len(chunk)
                                        now = time.monotonic()
                                        if now - last_update >= 0.5:
                                            speed = transferred / max(now - started, 0.01)
                                            self.update(downloaded_bytes=completed + offset, total_bytes=total,
                                                percent=round((completed + offset) / total * 100, 1), bytes_per_second=round(speed),
                                                eta_seconds=round((total - completed - offset) / speed) if speed else None)
                                            last_update = now
                            break
                        except httpx.HTTPError:
                            if attempt == 2:
                                raise
                            if self.stopping.wait(2 ** attempt):
                                raise InterruptedError()
                    self.update(stage="正在校验 " + name, eta_seconds=None)
                    if not valid(partial):
                        partial.unlink(missing_ok=True)
                        raise ValueError("模型文件校验失败，请重新下载")
                    partial.replace(path)
                    completed += size
                (destination / "ready.json").write_text(json.dumps({"model": MODEL_ID, "revision": REVISION,
                    "files": {row["rfilename"]: row["size"] for row in files}}), encoding="utf-8")
                self.update(status="ready", stage="已下载，可在本机重排", percent=100,
                            downloaded_bytes=total, total_bytes=total, eta_seconds=0)
        except InterruptedError:
            self.update(status="interrupted", stage="下载已中断，可继续下载", eta_seconds=None)
        except Exception as exc:
            # 外部错误可能包含重定向签名和网络配置，不直接返回异常正文。
            message = str(exc) if isinstance(exc, ValueError) else "下载失败，请检查网络后继续；已下载的部分会保留"
            self.update(status="failed", stage=message, error=message, eta_seconds=None)

    def close(self):
        self.stopping.set()
        self.pool.shutdown(wait=True, cancel_futures=True)
