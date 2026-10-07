from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from reader_agent.library import JobService, Library
from reader_agent.logging_utils import Observability
from reader_agent.server import create_app


class LoggingTest(unittest.TestCase):
    def test_rotation_and_error_separation(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Observability(temporary, max_bytes=350, backups=2)
            for number in range(20):
                logs.event("测试轮转", number=number)
            logs.event("测试错误", level=logging.ERROR)
            logs.close()
            paths = list(Path(temporary).glob("runtime.log*"))
            self.assertLessEqual(len(paths), 3)
            self.assertGreater(len(paths), 1)
            errors = (Path(temporary) / "errors.log").read_text()
            self.assertIn("测试错误", errors)
            self.assertNotIn("测试轮转", errors)
            for path in paths:
                for line in path.read_text().splitlines():
                    self.assertIn("time", json.loads(line))

    def test_exception_trace_and_secret_redaction_do_not_save_original_text(self):
        secret, original = "假密钥-123", "绝不写入日志的小说原文"
        with tempfile.TemporaryDirectory() as temporary, patch("reader_agent.logging_utils.get_config", return_value={"LLM_API_KEY": secret}):
            logs = Observability(temporary)
            with logs.bind(request_id="请求测试", job_id="任务测试"):
                logs.event("脱敏测试", model=secret)
                try:
                    raise RuntimeError(secret + original)
                except RuntimeError as exc:
                    logs.error("异常测试", exc)
            logs.close()
            text = "\n".join(path.read_text() for path in Path(temporary).glob("*.log"))
            self.assertNotIn(secret, text)
            self.assertNotIn(original, text)
            self.assertIn("[已隐藏]", text)
            row = json.loads((Path(temporary) / "errors.log").read_text())
            self.assertEqual(row["job_id"], "任务测试")
            self.assertEqual(row["exception"]["chain"][0]["type"], "RuntimeError")
            self.assertTrue(row["exception"]["chain"][0]["frames"])

    def test_request_and_task_share_ids_without_logging_bodies_or_query(self):
        original = "不应进入日志的私密原文"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(root / "library", config_path=root / ".env")
            with TestClient(app) as client:
                response = client.post("/api/books?private_query=不记录", files={"file": ("测试.txt", original.encode())})
                request_id = response.headers["X-Request-ID"]
                job_id = response.json()["id"]
                with app.state.library.connection() as db:
                    self.assertIsNotNone(db.execute("SELECT id FROM jobs WHERE id=?", (job_id,)).fetchone())
            directory = root / "library" / "logs"
            rows = [json.loads(line) for line in (directory / "runtime.log").read_text().splitlines()]
            queued = next(row for row in rows if row["event"] == "任务加入队列")
            self.assertEqual(queued["request_id"], request_id)
            self.assertEqual(queued["job_id"], job_id)
            access = json.loads((directory / "access.log").read_text().splitlines()[0])
            self.assertEqual(access["request_id"], request_id)
            self.assertEqual(access["route"], "/api/books")
            text = "\n".join(path.read_text() for path in directory.glob("*.log"))
            self.assertNotIn(original, text)
            self.assertNotIn("private_query", text)

    def test_failed_job_has_exception_frames_in_error_file(self):
        async def failure(job):
            raise RuntimeError("模拟服务异常")
        with tempfile.TemporaryDirectory() as temporary:
            library = Library(temporary)
            service = JobService(library)
            with patch.object(service, "_execute", side_effect=failure):
                job = library.create_job("graph", {})
                service._run(job["id"], "请求来源")
            service.close()
            self.assertEqual(library.job(job["id"])["status"], "failed")
            error = json.loads((Path(temporary) / "logs" / "errors.log").read_text())
            self.assertEqual(error["job_id"], job["id"])
            self.assertEqual(error["request_id"], "请求来源")
            self.assertEqual(error["exception"]["chain"][0]["type"], "RuntimeError")

    def test_unexpected_request_error_returns_trace_id_and_safe_message(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = create_app(Path(temporary) / "library")
            @app.get("/api/test-error")
            def error():
                raise RuntimeError("响应中不能显示的内部信息")
            with TestClient(app) as client:
                response = client.get("/api/test-error")
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])
                self.assertNotIn("内部信息", response.text)
            self.assertIn("接口异常", (Path(temporary) / "library" / "logs" / "errors.log").read_text())
