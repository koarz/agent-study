"""按 Qwen 官方 yes/no 打分协议运行 0.6B 重排，复用已加载模型，完全离线推理。"""

from __future__ import annotations

import asyncio
import threading

from .logging_utils import emit, record_local_usage
from .api_calls import notify, check_stopping
from .progress import ProgressTracker
from .local_models import MODEL_ID, REVISION, model_path


_loaded = {}
_load_lock = threading.Lock()


def resolve_device(requested, *, cuda_available=None):
    """自动优先 CUDA；明确选 CUDA 但环境不支持时给出可操作说明。"""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("本地设备必须为 auto、cpu 或 cuda")
    if requested == "cpu":
        return "cpu"
    if cuda_available is None:
        import torch
        cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise ValueError("CUDA 不可用：请安装 CUDA 版 PyTorch 并检查 NVIDIA 驱动，或选择自动 / CPU")
    return "cuda" if cuda_available else "cpu"


class QwenLocalReranker:
    local = True

    def __init__(self, directory, *, device="auto", window_chars=1200):
        from .corpus import digest

        if not 300 <= window_chars <= 1600:
            raise ValueError("本地重排窗口须为 300 到 1600 字")
        self.window_chars = window_chars
        self.overlap = min(200, window_chars // 5)
        self.progress_callback = None
        self.path = model_path(directory)
        self.device = resolve_device(device)
        self.model = MODEL_ID
        self.identity = digest(f"local:{MODEL_ID}:{REVISION}:{self.device}:official-windows-v2:8192:{window_chars}:{self.overlap}")

    def windows(self, document):
        """按原文字符区间滑窗，重叠保留跨边界的事件，不裁掉段落末尾。"""
        result, start = [], 0
        while start < len(document):
            end = min(len(document), start + self.window_chars)
            result.append(document[start:end])
            if end == len(document):
                break
            start = end - self.overlap
        return result or [""]

    def window_count(self, document):
        return len(self.windows(document))

    @staticmethod
    def token_windows(ids, budget):
        """罕见的高 token 密度原文继续分窗，禁止静默截断。"""
        if budget < 128:
            raise ValueError("问题过长，本地模型无法容纳原文；请缩短当前问题")
        start = 0
        while start < len(ids):
            end = min(len(ids), start + budget)
            yield ids[start:end]
            if end == len(ids):
                return
            start = end - min(128, budget // 5)
        if not ids:
            yield []

    def _score(self, question, documents):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        check_stopping()
        key = (str(self.path.resolve()), self.device)
        with _load_lock:
            if key not in _loaded:
                emit("加载本地重排模型", model=MODEL_ID, device=self.device)
                notify("正在加载本地重排模型，首次加载需要等待")
                # CPU 使用 float32，兼容没有 BF16 加速的普通电脑。
                torch.set_num_threads(min(8, torch.get_num_threads()))
                tokenizer = AutoTokenizer.from_pretrained(self.path, padding_side="left", local_files_only=True,
                                                          trust_remote_code=False)
                model = AutoModelForCausalLM.from_pretrained(self.path, local_files_only=True, trust_remote_code=False,
                    use_safetensors=True, torch_dtype=torch.float32 if self.device == "cpu" else torch.float16).to(self.device).eval()
                _loaded[key] = tokenizer, model, threading.Lock()
            tokenizer, model, lock = _loaded[key]
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        no_id, yes_id = tokenizer.convert_tokens_to_ids("no"), tokenizer.convert_tokens_to_ids("yes")
        query_ids = tokenizer.encode(
            f"<Instruct>: Given a query about a book or text, retrieve original passages that answer the query\n<Query>: {question}\n<Document>: ",
            add_special_tokens=False)
        budget = 8192 - len(prefix_ids) - len(suffix_ids) - len(query_ids)
        if budget < 128:
            raise ValueError("问题过长，本地模型无法容纳原文；请缩短当前问题")
        document_windows = []
        for document in documents:
            windows = []
            for text in self.windows(document):
                ids = tokenizer.encode(text, add_special_tokens=False)
                windows.extend(self.token_windows(ids, budget))
            document_windows.append(windows)
        total = sum(map(len, document_windows))
        # token 密度异常时，把新增窗口加入总量，预计时间仍按实际处理量计算。
        extra = total - sum(self.window_count(text) for text in documents)
        callback = self.progress_callback
        if callback and extra:
            callback(0, extra)
        tracker, completed = ProgressTracker(total, unit="窗口"), 0
        values = [0.0] * len(documents)
        entries = [(index, ids) for index, windows in enumerate(document_windows) for ids in windows]
        # CPU 保持单窗口；GPU 用小批次提高吞吐，按补齐后的 token 总量限制内存。
        max_batch = 4 if self.device == "cuda" else 1
        offset = 0
        with lock, torch.inference_mode():
            while offset < len(entries):
                check_stopping()
                count = min(max_batch, len(entries) - offset)
                while count > 1 and count * max(
                        len(prefix_ids) + len(query_ids) + len(ids) + len(suffix_ids)
                        for _, ids in entries[offset:offset + count]) > 8192:
                    count -= 1
                batch = entries[offset:offset + count]
                inputs = tokenizer.pad({"input_ids": [prefix_ids + query_ids + ids + suffix_ids for _, ids in batch]},
                                       return_tensors="pt")
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                try:
                    logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1, :]
                except torch.cuda.OutOfMemoryError as exc:
                    del inputs
                    torch.cuda.empty_cache()
                    if count == 1:
                        raise ValueError("GPU 显存不足，请关闭占用显存的程序或减小本地原文窗口") from exc
                    # 显存被其他程序占用时缩小批次重试，不丢窗口或更换问题。
                    max_batch = max(1, count // 2)
                    continue
                scores = torch.nn.functional.softmax(logits[:, [no_id, yes_id]], dim=-1)[:, 1].float().tolist()
                for (index, _), score in zip(batch, scores):
                    values[index] = max(values[index], score)
                offset += count
                completed += count
                if callback:
                    callback(count, 0)
                else:
                    notify(tracker.update(f"本地分窗重排 {completed}/{total}（{self.device}）",
                                          current=completed, stage="本地分窗重排"))
        emit("本地重排完成", model=MODEL_ID, documents=len(documents), api_tokens=0)
        return values

    async def score(self, question, documents):
        try:
            result = await asyncio.to_thread(self._score, question, documents)
        except Exception:
            record_local_usage("本地原文重排", MODEL_ID, documents=len(documents), failed=True)
            raise
        else:
            record_local_usage("本地原文重排", MODEL_ID, documents=len(documents))
            return result
        finally:
            notify(None)

    async def close(self):
        # 常驻模型供后续问答复用，避免每次重新加载权重。
        pass
