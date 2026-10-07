"""验证下载续传、校验、接口和本地用量，测试文件为模拟权重，不下载真实模型。"""

import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from reader_agent.local_models import LocalModelManager, MODEL_KEY, REVISION, model_path
from reader_agent.logging_utils import Observability
from reader_agent.local_reranker import QwenLocalReranker, resolve_device, _loaded
from reader_agent.server import create_app


def mock_download(manager, *, ignore_range=False, corrupt=False):
    files = {"model.safetensors": b"simulated weights", "config.json": b"{}", "tokenizer_config.json": b"{}", "tokenizer.json": b"{}"}
    ranges = []
    def response(request):
        if "/api/models/" in request.url.path:
            return httpx.Response(200, json={"siblings": [{"rfilename": name, "size": len(body),
                "lfs": {"sha256": hashlib.sha256(body).hexdigest()}} for name, body in files.items()]})
        name = request.url.path.rsplit("/", 1)[-1]
        content = files[name]
        offset = int(request.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        ranges.append(offset)
        if corrupt:
            content = b"x" * len(content)
        if offset and not ignore_range:
            return httpx.Response(206, content=content[offset:], headers={"Content-Range": f"bytes {offset}-{len(content)-1}/{len(content)}"})
        return httpx.Response(200, content=content)
    client = httpx.Client(transport=httpx.MockTransport(response), follow_redirects=True)
    with patch("reader_agent.local_models.httpx.Client", return_value=client):
        manager.download()
    return ranges


class LocalModelsTest(unittest.TestCase):
    def test_device_choice_prioritizes_available_cuda_and_has_explicit_fallback(self):
        self.assertEqual(resolve_device('auto', cuda_available=True), 'cuda')
        self.assertEqual(resolve_device('auto', cuda_available=False), 'cpu')
        self.assertEqual(resolve_device('cpu', cuda_available=True), 'cpu')
        self.assertEqual(resolve_device('cuda', cuda_available=True), 'cuda')
        with self.assertRaisesRegex(ValueError, 'CUDA 不可用'):
            resolve_device('cuda', cuda_available=False)
        with self.assertRaises(ValueError):
            resolve_device('unknown', cuda_available=True)

    def test_resume_and_server_ignoring_range(self):
        for ignore in (False, True):
            with tempfile.TemporaryDirectory() as root:
                manager = LocalModelManager(root)
                directory = Path(root) / MODEL_KEY
                directory.mkdir()
                (directory / "model.safetensors.part").write_bytes(b"simulated")
                ranges = mock_download(manager, ignore_range=ignore)
                self.assertEqual(ranges[0], 9)
                self.assertEqual(manager.snapshot()["status"], "ready")
                self.assertEqual((model_path(root) / "model.safetensors").read_bytes(), b"simulated weights")
                self.assertEqual(manager.snapshot()["revision"], REVISION)
                manager.close()

    def test_corrupt_weights_rejected_and_duplicate_download_not_started(self):
        with tempfile.TemporaryDirectory() as root:
            manager = LocalModelManager(root)
            mock_download(manager, corrupt=True)
            self.assertEqual(manager.snapshot()["status"], "failed")
            with self.assertRaises(ValueError):
                model_path(root)
            with patch.object(manager.pool, "submit") as submit:
                manager.start()
                manager.start()
                submit.assert_called_once()
            manager.close()

    def test_windows_cover_end_overlap_and_cache_changes(self):
        with tempfile.TemporaryDirectory() as root:
            manager = LocalModelManager(root)
            mock_download(manager)
            manager.close()
            scorer = QwenLocalReranker(root, window_chars=300)
            text = ''.join(chr(0x4e00 + i) for i in range(1001))
            windows = scorer.windows(text)
            restored = windows[0] + ''.join(window[scorer.overlap:] for window in windows[1:])
            self.assertEqual(restored, text)
            self.assertEqual(windows[-1][-1], text[-1])
            self.assertTrue(all(len(window) <= 300 for window in windows))
            self.assertNotEqual(scorer.identity, QwenLocalReranker(root, window_chars=1200).identity)
            # 半精度 GPU 与全精度 CPU 分数分开缓存，不能混用旧值。
            with patch('reader_agent.local_reranker.resolve_device', return_value='cpu'):
                cpu = QwenLocalReranker(root, window_chars=1200)
            with patch('reader_agent.local_reranker.resolve_device', return_value='cuda'):
                gpu = QwenLocalReranker(root, window_chars=1200)
            self.assertNotEqual(cpu.identity, gpu.identity)

            # 高 token 密度也覆盖末尾，不使用截断编码。
            tokens = list(range(1000))
            split = list(scorer.token_windows(tokens, 200))
            restored_tokens = split[0] + [item for window in split[1:] for item in window[40:]]
            self.assertEqual(restored_tokens, tokens)
            with self.assertRaises(ValueError):
                list(scorer.token_windows(tokens, 100))

    def test_model_endpoints_and_local_settings(self):
        with tempfile.TemporaryDirectory() as root:
            app = create_app(Path(root) / "data", config_path=Path(root) / ".env", model_directory=Path(root) / "models")
            with TestClient(app) as client:
                self.assertEqual(client.get("/api/models").json()[0]["id"], MODEL_KEY)
                self.assertEqual(client.post("/api/models/unknown/download").status_code, 404)
                with patch.object(app.state.local_models, "start", return_value={"status": "downloading"}) as start:
                    self.assertEqual(client.post(f"/api/models/{MODEL_KEY}/download").status_code, 202)
                    start.assert_called_once()
                self.assertEqual(client.put("/api/settings", json={"values": {"RERANK_PROVIDER": "local", "LOCAL_DEVICE": "cpu",
                    "LOCAL_RERANK_MODEL_ID": "Qwen/Qwen3-Reranker-0.6B"}}).status_code, 200)
                self.assertEqual(client.put("/api/settings", json={"values": {"RERANK_PROVIDER": "unknown"}}).status_code, 400)
                self.assertEqual(client.put("/api/settings", json={"values": {"LOCAL_RERANK_PASSES": "2",
                    "LOCAL_RERANK_BATCH_SIZE": "8", "LOCAL_RERANK_WINDOW_CHARS": "1200"}}).status_code, 200)
                for device in ('auto', 'cuda', 'cpu'):
                    self.assertEqual(client.put('/api/settings', json={'values': {'LOCAL_DEVICE': device}}).status_code, 200)
                self.assertEqual(client.put('/api/settings', json={'values': {'LOCAL_DEVICE': 'other'}}).status_code, 400)
                for key, invalid in [("LOCAL_RERANK_PASSES", "0"), ("LOCAL_RERANK_PASSES", "4"),
                                     ("LOCAL_RERANK_BATCH_SIZE", "33"), ("LOCAL_RERANK_WINDOW_CHARS", "200")]:
                    self.assertEqual(client.put("/api/settings", json={"values": {key: invalid}}).status_code, 400)



class LocalScoringTest(unittest.IsolatedAsyncioTestCase):
    async def test_gpu_batches_reduce_on_oom_without_losing_tail_windows(self):
        import torch
        from types import SimpleNamespace
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [4, 5] if '证据' in text else [4]
            def convert_tokens_to_ids(self, text): return {'no': 0, 'yes': 1}[text]
            def pad(self, values, **kwargs):
                rows = values['input_ids']
                length = max(map(len, rows))
                return {'input_ids': torch.tensor([[9] * (length - len(row)) + row for row in rows])}
        class Model:
            device = 'cpu'
            def __init__(self): self.batches = []
            def __call__(self, input_ids, **kwargs):
                self.batches.append(len(input_ids))
                if len(input_ids) > 2:
                    raise torch.cuda.OutOfMemoryError('模拟显存不足')
                logits = torch.zeros((len(input_ids), 1, 10))
                logits[:, 0, 1] = torch.where((input_ids == 5).any(dim=1), 3.0, -3.0)
                return SimpleNamespace(logits=logits)
        with tempfile.TemporaryDirectory() as root:
            manager = LocalModelManager(root)
            mock_download(manager)
            manager.close()
            with patch('reader_agent.local_reranker.resolve_device', return_value='cuda'):
                scorer = QwenLocalReranker(root, device='cuda', window_chars=300)
            model = Model()
            key = (str(scorer.path.resolve()), 'cuda')
            documents = ['背景' * 200 + '证据', '闲谈' * 300]
            counts = []
            scorer.progress_callback = lambda count, extra: counts.append(count)
            with patch.dict(_loaded, {key: (Tokenizer(), model, threading.Lock())}), \
                 patch('torch.cuda.empty_cache') as clear:
                scores = scorer._score('问题', documents)
            self.assertGreater(scores[0], 0.9)
            self.assertLess(scores[1], 0.1)
            self.assertEqual(sum(counts), sum(scorer.window_count(text) for text in documents))
            self.assertEqual(model.batches[0], 4)
            self.assertTrue(all(count <= 2 for count in model.batches[1:]))
            clear.assert_called_once()

    async def test_local_scoring_has_zero_api_usage_and_distinct_cache_identity(self):
        with tempfile.TemporaryDirectory() as root:
            manager = LocalModelManager(Path(root) / "models")
            mock_download(manager)
            manager.close()
            scorer = QwenLocalReranker(Path(root) / "models")
            logs = Observability(Path(root) / "logs")
            with logs.bind(), patch.object(scorer, "_score", return_value=[0.1, 0.9]):
                self.assertEqual(await scorer.score("问题", ["原文一", "原文二"]), [0.1, 0.9])
            summary = logs.usage.summary()
            self.assertEqual(summary["total"]["local_calls"], 1)
            self.assertEqual(summary["total"]["total_tokens"], 0)
            self.assertEqual(summary["total"]["missing_usage"], 0)
            self.assertEqual(summary["models"][0]["operation"], "本地原文重排")
            logs.close()
