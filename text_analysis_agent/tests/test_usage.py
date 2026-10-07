"""离线验证实际请求记账、历史补入与统计汇总，不调用付费模型。"""

import json
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient
from openai import RateLimitError

from reader_agent.api_calls import api_call
from reader_agent.logging_utils import Observability, model_operation
from reader_agent.server import create_app
from reader_agent.usage import UsageStore, normalize_usage


class UsageTest(unittest.TestCase):
    def test_partial_usage_and_embedding_total(self):
        self.assertEqual(normalize_usage({"total_tokens": 30}, "rerank"),
            {"input_tokens": None, "output_tokens": None, "total_tokens": 30, "cached_input_tokens": None})
        self.assertEqual(normalize_usage({"total_tokens": 30}, "embedding")["input_tokens"], 30)
        row = normalize_usage({"prompt_tokens": 8, "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 5}}, "llm")
        self.assertEqual(row["total_tokens"], 10)
        self.assertEqual(row["cached_input_tokens"], 5)
        self.assertIsNone(normalize_usage({"total_tokens": True, "input_tokens": -1}, "llm")["total_tokens"])

    def test_history_idempotence_dates_and_new_calls_not_reimported(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            old = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
            recent = datetime.now(timezone.utc).isoformat()
            rows = [{"event": "模型调用完成", "call_id": "old", "operation": "回答或提取", "model": "共享模型",
                     "time": old, "input_tokens": 80, "output_tokens": 20, "prompt": "不能入库的原文"},
                    {"event": "模型调用完成", "call_id": "embedding", "operation": "原文嵌入", "model": "共享模型",
                     "time": recent, "total_tokens": 40},
                    {"event": "模型调用完成", "call_id": "verify", "operation": "结论复核", "model": "复核模型",
                     "time": recent},
                    {"event": "模型调用完成", "call_id": "new", "operation": "原文重排", "model": "重排模型",
                     "time": recent, "usage_tracking": "api-v1", "total_tokens": 999}]
            (directory / "runtime.log").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows))
            store = UsageStore(directory / "usage.sqlite3")
            store.import_logs(directory)
            store.import_logs(directory)
            self.assertEqual(store.summary()["total"]["requests"], 3)
            self.assertEqual(store.summary()["total"]["total_tokens"], 140)
            self.assertEqual(store.summary(7)["total"]["total_tokens"], 40)
            self.assertEqual(store.summary(7)["total"]["missing_usage"], 1)
            categories = {row["category"]: row for row in store.summary()["categories"]}
            self.assertEqual(categories["llm"]["requests"], 2)
            self.assertEqual(categories["embedding"]["requests"], 1)
            self.assertNotIn("不能入库", (directory / "usage.sqlite3").read_bytes().decode(errors="ignore"))

    def test_endpoint_validation_and_empty_statistics(self):
        with tempfile.TemporaryDirectory() as root:
            app = create_app(Path(root) / "data", config_path=Path(root) / ".env")
            with TestClient(app) as client:
                result = client.get("/api/usage").json()
                self.assertEqual(result["total"]["requests"], 0)
                self.assertEqual(len(result["categories"]), 3)
                self.assertEqual(client.get("/api/usage?days=-1").status_code, 422)
                self.assertEqual(client.get("/api/usage?days=7").json()["days"], 7)


class UsageAPITest(unittest.IsolatedAsyncioTestCase):
    async def test_retry_and_invalid_content_are_counted_without_prompts(self):
        with tempfile.TemporaryDirectory() as root:
            logs = Observability(root)
            attempts = []
            async def request():
                attempts.append(1)
                if len(attempts) == 1:
                    raise RateLimitError("私密原文", response=httpx.Response(429,
                        request=httpx.Request("POST", "https://test.invalid"), headers={"Retry-After": "0"}), body=None)
                return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120))
            with logs.bind(book_id="book", job_id="job"):
                with self.assertRaisesRegex(ValueError, "无效回答"):
                    with model_operation("结论复核", "回答模型"):
                        await api_call(request, operation="结论复核", api_key=uuid.uuid4().hex, endpoint="https://test.invalid")
                        raise ValueError("无效回答")
            summary = logs.usage.summary()
            self.assertEqual(summary["total"]["requests"], 2)
            self.assertEqual(summary["total"]["failed_requests"], 1)
            self.assertEqual(summary["total"]["total_tokens"], 120)
            self.assertEqual(summary["models"][0]["operation"], "结论复核")
            logs.close()
            reopened = Observability(root)
            self.assertEqual(reopened.usage.summary()["total"]["requests"], 2)
            reopened.close()

    async def test_native_rerank_usage_missing_usage_and_cached_input(self):
        with tempfile.TemporaryDirectory() as root:
            logs = Observability(root)
            responses = [httpx.Response(200, json={"usage": {"total_tokens": 400}, "output": {"results": []}}),
                         SimpleNamespace(usage=None),
                         SimpleNamespace(usage={"prompt_tokens": 40, "completion_tokens": 10,
                                                "prompt_cache_hit_tokens": 30})]
            with logs.bind():
                for operation, response in zip(("原文重排", "查询嵌入", "回答或提取"), responses):
                    with model_operation(operation, "同名模型", documents=2):
                        async def request():
                            return response
                        await api_call(request, operation=operation, api_key="不应存储的密钥", endpoint="https://test.invalid")
            total = logs.usage.summary()["total"]
            self.assertEqual(total["total_tokens"], 450)
            self.assertEqual(total["missing_usage"], 1)
            self.assertEqual(total["cached_input_tokens"], 30)
            self.assertEqual(len(logs.usage.summary()["models"]), 3)
            logs.close()
            self.assertNotIn("不应存储的密钥", (Path(root) / "usage.sqlite3").read_bytes().decode(errors="ignore"))
