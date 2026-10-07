"""持久化结构化日志，关联请求与任务，并避免保存密钥和原文内容。"""

from __future__ import annotations

import logging
import sqlite3
import re
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
import json

from .config import get_config
from .usage import UsageStore, category_for, normalize_usage


_current = ContextVar("reader_observability", default=None)
_context = ContextVar("reader_log_context", default={})
_model_context = ContextVar("reader_model_usage", default=None)


class JSONFormatter(logging.Formatter):
    def format(self, record):
        values = {"time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                  "level": record.levelname, "event": record.getMessage(),
                  **getattr(record, "event_fields", {})}
        if record.exc_info and record.exc_info[1]:
            values["exception"] = exception_details(record.exc_info[1])
        encoded = json.dumps(values, ensure_ascii=False, default=str)
        # 每次读取最新密钥，页面更新设置后也能脱敏；不记录配置本身。
        for key, value in get_config().items():
            if key.endswith("API_KEY") and value:
                encoded = encoded.replace(json.dumps(value, ensure_ascii=False)[1:-1], "[已隐藏]")
        return re.sub(r"(?i)Bearer\s+[^\s\"\\]+", "Bearer [已隐藏]", encoded)


def exception_details(exc: BaseException) -> dict:
    """保留异常链、HTTP 状态和调用栈，不保存可能含原文或密钥的异常响应体。"""
    chain, seen = [], set()
    while exc is not None and id(exc) not in seen and len(chain) < 8:
        seen.add(id(exc))
        item = {"type": type(exc).__name__, "frames": [
            {"file": frame.filename, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(exc.__traceback__)]}
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)
        if type(status) is int:
            item["http_status"] = status
        if isinstance(exc, sqlite3.Error):
            item["sqlite_code"] = getattr(exc, "sqlite_errorcode", None)
            item["sqlite_name"] = getattr(exc, "sqlite_errorname", None)
        code = getattr(exc, "code", None)
        if isinstance(code, str) and code in {"invalid_api_key", "model_not_found", "rate_limit_exceeded", "insufficient_quota",
                    "context_length_exceeded", "invalid_request_error", "account_deactivated", "permission_denied",
                    "unsupported_parameter", "input_exceeds_model_context_window"}:
            item["provider_code"] = code
        # 外部异常信息可能携带完整提示词，所以仅保留本项目定义的格式错误说明。
        if type(exc).__name__ in {"ModelOutputError", "APIServiceError"}:
            item["reason"] = str(exc)[:500]
        chain.append(item)
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return {"chain": chain}


class Observability:
    def __init__(self, directory: Path | str, *, max_bytes=5 * 1024 * 1024, backups=5):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.handlers = []
        self.loggers = {}
        self.server_loggers = []
        name = "reader." + uuid.uuid4().hex
        for channel in ("runtime", "access"):
            logger = logging.getLogger(name + "." + channel)
            logger.setLevel(logging.INFO)
            logger.propagate = False
            handler = self._handler(channel + ".log", logging.INFO, max_bytes, backups)
            logger.addHandler(handler)
            self.loggers[channel] = logger
        self.loggers["runtime"].addHandler(self._handler("errors.log", logging.ERROR, max_bytes, backups))
        self.usage = UsageStore(self.directory / "usage.sqlite3")
        self.usage.import_logs(self.directory)

    def _handler(self, filename, level, max_bytes, backups):
        handler = RotatingFileHandler(self.directory / filename, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(JSONFormatter())
        self.handlers.append(handler)
        return handler

    def event(self, event, *, level=logging.INFO, channel="runtime", **fields):
        self.loggers[channel].log(level, event, extra={"event_fields": {**_context.get(), **fields}})

    def error(self, event, exc, **fields):
        self.event(event, level=logging.ERROR, exception=exception_details(exc), **fields)

    @contextmanager
    def bind(self, **fields):
        observer_token = _current.set(self)
        context_token = _context.set({**_context.get(), **fields})
        try:
            yield
        finally:
            _context.reset(context_token)
            _current.reset(observer_token)

    def capture_server(self):
        # 保留服务原有控制台输出，同时将服务启动、停止和错误写入文件。
        logger = logging.getLogger("uvicorn.error")
        for handler in self.loggers["runtime"].handlers:
            logger.addHandler(handler)
            self.server_loggers.append((logger, handler))

    def close(self):
        for logger, handler in self.server_loggers:
            logger.removeHandler(handler)
        self.server_loggers.clear()
        for logger in self.loggers.values():
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
        for handler in self.handlers:
            handler.close()


def emit(event, *, level=logging.INFO, **fields):
    observer = _current.get()
    if observer is not None:
        observer.event(event, level=level, **fields)


def request_id():
    return _context.get().get("request_id")


def record_api_usage(response=None, *, failed=False):
    """每次实际请求结束时计数，内容校验失败也保留接口已报告的消耗。"""
    observer, model = _current.get(), _model_context.get()
    if observer is None or model is None:
        return
    status = getattr(response, "status_code", None)
    failed = failed or (type(status) is int and status >= 400)
    value = getattr(response, "usage", None)
    if value is None and response is not None and callable(getattr(response, "json", None)):
        try:
            payload = response.json()
            if isinstance(payload, dict):
                value = payload.get("usage")
        except (ValueError, TypeError):
            pass
    try:
        observer.usage.record({**model, **{key: _context.get().get(key) for key in ("book_id", "job_id")},
            "id": uuid.uuid4().hex, "source": "api", "status": "failed" if failed else "success",
            **normalize_usage(value, model["category"])})
    except Exception as exc:
        # 统计文件故障写入安全日志，避免中断原文分析任务。
        observer.error("保存 API 用量失败", exc)


def record_local_usage(operation, model, *, documents=0, failed=False):
    """本地评分单独计数，明确记录其 API token 消耗为零。"""
    observer = _current.get()
    if observer is None:
        return
    try:
        identifier = uuid.uuid4().hex
        observer.usage.record({"id": identifier, "call_id": identifier, "source": "local",
            "category": category_for(operation), "operation": operation, "model": model,
            "status": "failed" if failed else "success", "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "documents": documents,
            **{key: _context.get().get(key) for key in ("book_id", "job_id")}})
    except Exception as exc:
        observer.error("保存本地模型调用计数失败", exc)


@contextmanager
def model_operation(operation, model, **fields):
    """记录模型调用次数、耗时及失败，不接收请求正文或模型回复。"""
    observer = _current.get()
    call_id = uuid.uuid4().hex
    started = time.perf_counter()
    metadata = {"call_id": call_id, "operation": operation, "model": model, **fields,
                "usage_tracking": "api-v1"}
    token = _model_context.set({"call_id": call_id, "operation": operation, "model": model,
                               "category": category_for(operation), **fields})
    emit("模型调用开始", **metadata)
    result = {}
    try:
        yield result
    except Exception as exc:
        if observer is not None:
            observer.error("模型调用失败", exc, **metadata, elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
        raise
    else:
        emit("模型调用完成", **metadata, **result, elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
    finally:
        _model_context.reset(token)
