"""统一控制 API 请求并发、限流等待和安全错误说明，跨后台线程共享冷却时间。"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

from .logging_utils import emit, record_api_usage


class APIServiceError(RuntimeError):
    """仅包含程序定义的错误说明，不包含服务商原始响应。"""


_notice = ContextVar("api_wait_notice", default=None)
_stopping = ContextVar("api_stopping", default=None)
_priority = ContextVar("api_priority", default=10)
_gates, _gate_lock = {}, threading.Lock()


@contextmanager
def api_context(notice, stopping, *, priority=10):
    tokens = (_notice.set(notice), _stopping.set(stopping), _priority.set(priority))
    try:
        yield
    finally:
        _notice.reset(tokens[0])
        _stopping.reset(tokens[1])
        _priority.reset(tokens[2])


def notify(message):
    callback = _notice.get()
    if callback is not None:
        callback(message)


def check_stopping():
    """本地推理在每个窗口开始前检查停止信号，避免关服一直等待。"""
    stop = _stopping.get()
    if stop is not None and stop():
        raise RuntimeError("服务正在停止，任务可继续执行")


class RequestGate:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.until = 0.0
        self.waiters = {}
        self.sequence = 0

    async def acquire(self):
        with self.lock:
            self.sequence += 1
            ticket = self.sequence
            self.waiters[ticket] = (_priority.get(), ticket)
        try:
            while True:
                stop = _stopping.get()
                if stop is not None and stop():
                    raise RuntimeError("服务正在停止，任务可继续执行")
                with self.lock:
                    if not self.active and time.monotonic() >= self.until and min(self.waiters, key=self.waiters.get) == ticket:
                        self.active = True
                        return
                await asyncio.sleep(0.1)
        finally:
            with self.lock:
                self.waiters.pop(ticket, None)

    def release(self):
        with self.lock:
            self.active = False

    def cool_down(self, seconds):
        with self.lock:
            self.until = max(self.until, time.monotonic() + seconds)


def request_gate(endpoint, api_key):
    # 密钥仅参与内存中的分组散列，不能进入日志或持久化数据。
    origin = urlsplit(str(endpoint)).netloc.lower()
    key = hashlib.sha256((origin + "\0" + api_key).encode()).digest()
    with _gate_lock:
        return _gates.setdefault(key, RequestGate())


def quota_exhausted(exc):
    body = getattr(exc, "body", None)
    if body is None:
        response = getattr(exc, "response", None)
        try:
            body = response.json() if response is not None else {}
        except ValueError:
            body = {}
    error = body.get("error", body) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        return False
    code = str(error.get("code", "")).lower()
    message = str(error.get("message", "")).lower()
    return code in {"insufficient_quota", "insufficient_balance", "quota_exhausted"} or any(
        text in message for text in ("insufficient balance", "credit balance is too low", "余额不足", "额度已用尽"))


def retry_after(exc):
    headers = getattr(getattr(exc, "response", None), "headers", {})
    for key, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
        value = headers.get(key)
        if value is None:
            continue
        try:
            seconds = float(value) / divisor
        except (ValueError, TypeError):
            try:
                seconds = parsedate_to_datetime(value).timestamp() - time.time() if key == "retry-after" else None
            except (ValueError, TypeError, OverflowError):
                seconds = None
        if seconds is not None and math.isfinite(seconds) and seconds >= 0:
            return seconds
    return None


async def api_call(call, *, operation, api_key, endpoint, attempts=5, base_delay=2, wait_budget=120):
    gate = request_gate(endpoint, api_key)
    waited = 0.0
    announced = False
    try:
        for attempt in range(attempts):
            await gate.acquire()
            try:
                response = await call()
                record_api_usage(response)
                return response
            except Exception as exc:
                record_api_usage(getattr(exc, "response", None), failed=True)
                status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                transient = status in {408, 409, 429} or isinstance(status, int) and status >= 500 or type(exc).__name__ in {
                    "APIConnectionError", "APITimeoutError", "ConnectError", "ReadError", "ReadTimeout", "ConnectTimeout", "PoolTimeout", "WriteTimeout"}
                if not transient:
                    raise
                exhausted = status == 429 and quota_exhausted(exc)
                error_label = f"HTTP {status}" if status is not None else "连接异常"
                emit("API 暂时不可用", level=logging.WARNING, operation=operation, http_status=status,
                     category="quota" if exhausted else "rate_or_capacity" if status == 429 else "temporary_failure", attempt=attempt + 1)
                if exhausted:
                    raise APIServiceError(f"{operation}服务报告账户额度或余额不足，请检查该服务商的账户额度后继续任务。") from exc
                if attempt + 1 >= attempts:
                    raise APIServiceError(f"{operation}服务持续返回 {error_label}，已等待并重试 {attempts - 1} 次；请稍后继续任务，并检查服务商的请求限额、账户额度及服务状态。") from exc
                delay = retry_after(exc)
                if delay is None:
                    delay = base_delay * 2 ** attempt + random.uniform(0, base_delay / 2)
                if waited + delay > wait_budget:
                    raise APIServiceError(f"{operation}服务返回 {error_label}，要求的等待时间超过本次重试上限；请稍后继续任务。") from exc
                waited += delay
                # 所有使用同一服务及密钥的任务一起等待，避免相互触发限流。
                gate.cool_down(delay)
                announced = True
                notify(f"{operation}服务返回 {error_label}，等待约 {math.ceil(delay)} 秒后自动重试（{attempt + 1}/{attempts - 1}）")
                emit("API 等待后重试", level=logging.WARNING, operation=operation, delay_seconds=round(delay, 2), retry=attempt + 1)
            finally:
                gate.release()
    finally:
        if announced:
            notify(None)
