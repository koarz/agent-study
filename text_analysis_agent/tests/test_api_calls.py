from __future__ import annotations

import asyncio
import unittest
import uuid

import httpx
from openai import RateLimitError, AuthenticationError

from reader_agent.api_calls import APIServiceError, RequestGate, api_call, api_context, retry_after


def rate_error(*, body=None, headers=None):
    response = httpx.Response(429, request=httpx.Request('POST', 'https://test.invalid/v1'), headers=headers)
    return RateLimitError('外部原文和密钥不能显示', response=response, body=body or {'error': {'code': 'rate_limit_exceeded'}})


class APICallsTest(unittest.IsolatedAsyncioTestCase):
    async def test_rate_limit_retries_same_operation_and_reports_wait(self):
        calls, notices = [], []
        async def request():
            calls.append('同一请求')
            if len(calls) < 3:
                raise rate_error(headers={'Retry-After': '0'})
            return '复核成功'
        with api_context(notices.append, lambda: False):
            result = await api_call(request, operation='结论复核', api_key=uuid.uuid4().hex, endpoint='https://test.invalid')
        self.assertEqual(result, '复核成功')
        self.assertEqual(len(calls), 3)
        self.assertTrue(all('HTTP 429' in n for n in notices[:-1]))
        self.assertIsNone(notices[-1])
        self.assertNotIn('外部原文', str(notices))

    async def test_quota_and_authentication_are_not_retried(self):
        for error in (rate_error(body={'error': {'code': 'insufficient_quota'}}),
                      AuthenticationError('私密响应', response=httpx.Response(401, request=httpx.Request('POST', 'https://test.invalid')), body=None)):
            calls = []
            async def request():
                calls.append(1)
                raise error
            with self.assertRaises(APIServiceError if isinstance(error, RateLimitError) else AuthenticationError):
                await api_call(request, operation='结论复核', api_key=uuid.uuid4().hex, endpoint='https://test.invalid')
            self.assertEqual(len(calls), 1)

    async def test_persistent_rate_limit_is_bounded_and_wait_header_is_honored(self):
        calls = []
        async def request():
            calls.append(1)
            raise rate_error(headers={'Retry-After': '0'})
        with self.assertRaises(APIServiceError) as caught:
            await api_call(request, operation='结论复核', api_key=uuid.uuid4().hex, endpoint='https://test.invalid', attempts=3)
        self.assertEqual(len(calls), 3)
        self.assertIn('已等待并重试 2 次', str(caught.exception))
        self.assertNotIn('外部原文', str(caught.exception))
        self.assertEqual(retry_after(rate_error(headers={'retry-after-ms': '1500'})), 1.5)
        async def long_wait():
            raise rate_error(headers={'Retry-After': '300'})
        with self.assertRaisesRegex(APIServiceError, '等待时间超过'):
            await api_call(long_wait, operation='回答', api_key=uuid.uuid4().hex, endpoint='https://test.invalid', wait_budget=120)

    async def test_shared_gate_prioritizes_question_and_releases_on_cancellation(self):
        gate, order = RequestGate(), []
        await gate.acquire()
        async def acquire(name, priority):
            with api_context(None, lambda: False, priority=priority):
                await gate.acquire()
                order.append(name)
                gate.release()
        background = asyncio.create_task(acquire('索引', 10))
        question = asyncio.create_task(acquire('问答', 0))
        canceled = asyncio.create_task(acquire('取消', 0))
        await asyncio.sleep(0)
        canceled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await canceled
        gate.release()
        await asyncio.gather(background, question)
        self.assertEqual(order, ['问答', '索引'])
        self.assertFalse(gate.waiters)
        self.assertFalse(gate.active)

    async def test_service_stop_interrupts_cooldown(self):
        gate = RequestGate()
        gate.cool_down(60)
        with api_context(None, lambda: True):
            with self.assertRaisesRegex(RuntimeError, '服务正在停止'):
                await gate.acquire()
        self.assertFalse(gate.waiters)
