from __future__ import annotations

import json
import unittest

import httpx
from openai import AsyncOpenAI, AuthenticationError

from reader_agent import schemas
from reader_agent.llm import CompatibleBackend, ModelOutputError, parse_json_object, request_json


class ReplyBackend:
    """模拟格式错误和重试，不调用付费接口。"""
    json_attempts = 2

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    async def complete(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0)


class JSONTest(unittest.IsolatedAsyncioTestCase):
    def test_fenced_json_crlf_and_ambiguous_prose(self):
        self.assertEqual(parse_json_object('```JSON\r\n{"facts":[]}\r\n```'), {"facts": []})
        for text in ('说明：{"facts":[]}', '{"facts":[]}\n{"facts":[]}', '{"facts": [', '[]'):
            with self.assertRaises(ModelOutputError):
                parse_json_object(text)

    async def test_prose_or_wrong_schema_regenerates_once_using_same_evidence(self):
        for bad in ("自然语言说明", '{"relations":[]}'):
            backend = ReplyBackend([bad, '{"status":"not_found","facts":[]}'])
            result = await request_json(backend, "只输出 JSON", {"evidence": [{"text": "固定原文"}]}, schema=schemas.GRAPH)
            self.assertEqual(result["facts"], [])
            self.assertEqual(len(backend.calls), 2)
            self.assertEqual(json.loads(backend.calls[0][1]["content"])["evidence"], json.loads(backend.calls[1][1]["content"])["evidence"])

    async def test_persistent_failure_is_bounded_and_never_renames_fields(self):
        backend = ReplyBackend(['{"relations":[]}', '{"relations":[]}'])
        with self.assertRaises(ModelOutputError):
            await request_json(backend, "JSON", {}, schema=schemas.GRAPH)
        self.assertEqual(len(backend.calls), 2)

    async def test_search_plan_rejects_answers_in_place_of_keyword_groups(self):
        bad = {'literal': '未找到答案', 'paraphrase': '资料不足，无法回答。', 'broader': '未知', 'entities': []}
        valid = {'literal': ['林舟', '获得', '铜钥匙'], 'paraphrase': ['林舟', '拿到', '铜钥匙'],
                 'broader': ['林舟', '钥匙'], 'entities': [{'name': '林舟', 'kind': 'person'}, {'name': '铜钥匙', 'kind': 'object'}]}
        backend = ReplyBackend([json.dumps(bad), json.dumps(valid)])
        plan = await request_json(backend, '只生成查询', {}, schema=schemas.QUERY_PLAN)
        self.assertEqual(plan, valid)
        self.assertEqual(len(backend.calls), 2)
        for words in ([], ['孤词'], ['带 空格', '查询'], ['长' * 25, '查询']):
            self.assertFalse(schemas.matches({**valid, 'paraphrase': words}, schemas.QUERY_PLAN))

    async def mocked_backend(self, respond):
        backend = CompatibleBackend(api_key="fake", model="测试模型", base_url="https://model.test/v1")
        await backend.close()
        backend.client = AsyncOpenAI(api_key="fake", base_url="https://model.test/v1", max_retries=0,
                                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        return backend

    async def test_schema_negotiation_falls_back_and_remembers_provider_support(self):
        requests = []
        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            if body["response_format"]["type"] == "json_schema":
                return httpx.Response(400, json={"error": {"message": "response_format json_schema is not supported"}})
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1, "model": "测试模型",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"status":"not_found","facts":[]}'}}]})
        backend = await self.mocked_backend(respond)
        try:
            await request_json(backend, "JSON", {}, schema=schemas.GRAPH)
            await request_json(backend, "JSON", {}, schema=schemas.GRAPH)
            self.assertEqual([r["response_format"]["type"] for r in requests], ["json_schema", "json_object", "json_object"])
        finally:
            await backend.close()

    async def test_authentication_error_does_not_trigger_format_fallback(self):
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(401, json={"error": {"message": "invalid key"}})
        backend = await self.mocked_backend(respond)
        try:
            with self.assertRaises(AuthenticationError):
                await request_json(backend, "JSON", {}, schema=schemas.GRAPH)
            self.assertEqual(len(requests), 1)
        finally:
            await backend.close()
