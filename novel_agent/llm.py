"""Small, provider-agnostic asynchronous LLM client for the novel agent."""

from __future__ import annotations

import inspect
import json
import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol, TypeVar


Message = dict[str, str]
Validator = Callable[[Any], str | None | bool]
AuditCallback = Callable[[dict[str, Any]], None | Awaitable[None]]
T = TypeVar("T")


class FatalLLMError(RuntimeError):
    """A provider/configuration failure that should abort the whole run."""


class InvalidLLMResponse(ValueError):
    """A response that can potentially be fixed by prompting the model again."""


class ChatBackend(Protocol):
    """Minimal backend contract, intentionally shared with the game agent."""

    async def complete(self, messages: list[Message], temperature: float) -> str: ...


class OpenAICompatibleBackend:
    """Chat-completions backend for OpenAI and compatible endpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str,
        timeout: int | float = 120,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**kwargs)

    async def complete(self, messages: list[Message], temperature: float) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason not in {None, "stop", "eos"}:
            raise RuntimeError(f"模型输出未正常结束（finish_reason={finish_reason}）")
        return choice.message.content or ""


_FATAL_RESPONSE_MARKERS = (
    "quota exhausted",
    "quota exceeded",
    "insufficient_quota",
    "insufficient quota",
    "please recharge",
    "please top up",
    "billing limit",
    "invalid api key",
    "invalid_api_key",
    "authentication failed",
    "authorization failed",
    "unauthorized",
    "access forbidden",
    "账户余额不足",
    "余额不足",
    "额度不足",
    "请充值",
    "鉴权失败",
    "未授权",
)


def _fatal_response_reason(text: str) -> str | None:
    """Recognize providers that return an error as a successful text body.

    The length guard avoids treating a legitimate chapter that happens to
    mention billing or a quota as a provider failure.
    """

    cleaned = text.strip()
    if not cleaned or len(cleaned) > 4000:
        return None
    lowered = cleaned.casefold()
    # Only classify provider-like error bodies.  Searching for a marker
    # anywhere would incorrectly reject legitimate prose such as “他的余额
    # 不足，只好离开酒馆”。
    prefixes = (
        "quota",
        "insufficient_quota",
        "insufficient quota",
        "please recharge",
        "please top up",
        "billing limit",
        "invalid api key",
        "invalid_api_key",
        "authentication failed",
        "authorization failed",
        "unauthorized",
        "access forbidden",
        "账户余额不足",
        "当前账户余额不足",
        "额度不足",
        "请充值",
        "鉴权失败",
        "未授权",
        "error:",
    )
    if any(lowered.startswith(prefix) for prefix in prefixes):
        return cleaned[:1000]
    if lowered.startswith("{") and '"error"' in lowered and any(
        marker in lowered for marker in _FATAL_RESPONSE_MARKERS
    ):
        return cleaned[:1000]
    return None


def _response_excerpt(response: str | None, limit: int = 6000) -> str:
    """Keep retry prompts bounded when a provider returned a huge response."""

    value = response or ""
    if len(value) <= limit:
        return value
    half = limit // 2
    return value[:half] + "\n...[旧响应已截断]...\n" + value[-half:]


def _fatal_exception_reason(exc: BaseException) -> str | None:
    """Classify authentication, permission, and exhausted-credit errors."""

    status_code = getattr(exc, "status_code", None)
    message = str(exc)
    lowered = message.casefold()
    if status_code in {401, 402, 403}:
        return message
    if status_code in {400, 404, 422}:
        return message
    if status_code == 413 or any(
        marker in lowered
        for marker in ("context length", "maximum context", "too many tokens", "request too large")
    ):
        return message
    if any(marker in lowered for marker in _FATAL_RESPONSE_MARKERS):
        return message
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object from plain or fenced model output.

    Besides clean JSON this accepts a Markdown fenced object or a short prose
    preamble.  ``JSONDecoder.raw_decode`` is used instead of a brace regex so
    nested objects and braces inside quoted strings remain valid.
    """

    cleaned = text.strip()
    if not cleaned:
        raise InvalidLLMResponse("模型返回了空内容")
    fatal_reason = _fatal_response_reason(cleaned)
    if fatal_reason:
        raise FatalLLMError(f"模型服务不可用：{fatal_reason}")

    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        closing_fence = cleaned.rfind("```")
        if first_newline >= 0 and closing_fence > first_newline:
            cleaned = cleaned[first_newline + 1 : closing_fence].strip()

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value = None
        last_error: json.JSONDecodeError | None = None
        for index, char in enumerate(cleaned):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(candidate, dict):
                value = candidate
                break
        if value is None:
            detail = f"：{last_error}" if last_error else ""
            raise InvalidLLMResponse(f"模型输出不是有效的 JSON 对象{detail}") from last_error

    if not isinstance(value, dict):
        raise InvalidLLMResponse("模型输出必须是 JSON 对象")
    return value


def _validation_error(validator: Validator | None, value: Any) -> str | None:
    if validator is None:
        return None
    result = validator(value)
    if isinstance(result, str):
        return result or None
    if result is False:
        return "自定义校验未通过"
    return None


class LLMClient:
    """Adds parsing, validation, retries, and optional call auditing."""

    def __init__(
        self,
        backend: ChatBackend,
        *,
        temperature: float = 0.7,
        max_retries: int = 3,
        audit_callback: AuditCallback | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries 必须至少为 1")
        self.backend = backend
        self.temperature = temperature
        self.max_retries = max_retries
        self.audit_callback = audit_callback

    async def _audit(self, payload: dict[str, Any]) -> None:
        if self.audit_callback is None:
            return
        result = self.audit_callback(payload)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _copy_messages(messages: list[Mapping[str, str]]) -> list[Message]:
        copied: list[Message] = []
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"不支持的消息角色：{role!r}")
            copied.append({"role": role, "content": content})
        if not copied:
            raise ValueError("messages 不能为空")
        return copied

    async def request_json(
        self,
        messages: list[Mapping[str, str]],
        *,
        purpose: str = "json",
        validator: Validator | None = None,
        schema_hint: str = "",
        temperature: float | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Request and validate a JSON object, retrying malformed output."""

        request_messages = self._copy_messages(messages)
        attempts = self._attempt_count(max_retries)
        call_temperature = self.temperature if temperature is None else temperature
        last_error = ""

        for attempt in range(1, attempts + 1):
            response: str | None = None
            try:
                response = await self.backend.complete(request_messages, call_temperature)
                value = parse_json_object(response)
                validation_error = _validation_error(validator, value)
                if validation_error:
                    raise InvalidLLMResponse(validation_error)
            except FatalLLMError as exc:
                await self._audit_event(purpose, attempt, request_messages, response, str(exc), fatal=True)
                raise
            except Exception as exc:
                fatal_reason = _fatal_exception_reason(exc)
                if fatal_reason:
                    fatal = FatalLLMError(f"模型服务不可用：{fatal_reason}")
                    await self._audit_event(purpose, attempt, request_messages, response, str(fatal), fatal=True)
                    raise fatal from exc
                last_error = str(exc)
                await self._audit_event(purpose, attempt, request_messages, response, last_error, fatal=False)
                if attempt >= attempts:
                    break
                correction = f"上次输出无效：{last_error}。只输出一个符合要求的 JSON 对象"
                if schema_hint:
                    correction += f"，结构必须符合：{schema_hint}"
                correction += "。不要输出 Markdown 或解释。"
                if not isinstance(exc, InvalidLLMResponse):
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
                request_messages.extend(
                    [
                        {"role": "assistant", "content": _response_excerpt(response)},
                        {"role": "user", "content": correction},
                    ]
                )
                continue

            await self._audit_event(purpose, attempt, request_messages, response, None, fatal=False)
            return value

        raise InvalidLLMResponse(f"模型连续 {attempts} 次未返回有效 JSON：{last_error}")

    async def request_text(
        self,
        messages: list[Mapping[str, str]],
        *,
        purpose: str = "text",
        validator: Validator | None = None,
        temperature: float | None = None,
        max_retries: int | None = None,
    ) -> str:
        """Request non-empty prose, retrying empty or rejected responses."""

        request_messages = self._copy_messages(messages)
        attempts = self._attempt_count(max_retries)
        call_temperature = self.temperature if temperature is None else temperature
        last_error = ""

        for attempt in range(1, attempts + 1):
            response: str | None = None
            try:
                response = await self.backend.complete(request_messages, call_temperature)
                fatal_reason = _fatal_response_reason(response)
                if fatal_reason:
                    raise FatalLLMError(f"模型服务不可用：{fatal_reason}")
                value = response.strip()
                if not value:
                    raise InvalidLLMResponse("模型返回了空内容")
                validation_error = _validation_error(validator, value)
                if validation_error:
                    raise InvalidLLMResponse(validation_error)
            except FatalLLMError as exc:
                await self._audit_event(purpose, attempt, request_messages, response, str(exc), fatal=True)
                raise
            except Exception as exc:
                fatal_reason = _fatal_exception_reason(exc)
                if fatal_reason:
                    fatal = FatalLLMError(f"模型服务不可用：{fatal_reason}")
                    await self._audit_event(purpose, attempt, request_messages, response, str(fatal), fatal=True)
                    raise fatal from exc
                last_error = str(exc)
                await self._audit_event(purpose, attempt, request_messages, response, last_error, fatal=False)
                if attempt >= attempts:
                    break
                if not isinstance(exc, InvalidLLMResponse):
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
                request_messages.extend(
                    [
                        {"role": "assistant", "content": _response_excerpt(response)},
                        {
                            "role": "user",
                            "content": f"上次输出无效：{last_error}。请重新输出完整正文，不要解释或使用代码围栏。",
                        },
                    ]
                )
                continue

            await self._audit_event(purpose, attempt, request_messages, response, None, fatal=False)
            return value

        raise InvalidLLMResponse(f"模型连续 {attempts} 次未返回有效文本：{last_error}")

    def _attempt_count(self, override: int | None) -> int:
        attempts = self.max_retries if override is None else override
        if attempts < 1:
            raise ValueError("max_retries 必须至少为 1")
        return attempts

    async def _audit_event(
        self,
        purpose: str,
        attempt: int,
        messages: list[Message],
        response: str | None,
        error: str | None,
        *,
        fatal: bool,
    ) -> None:
        await self._audit(
            {
                "purpose": purpose,
                "attempt": attempt,
                "messages": [dict(message) for message in messages],
                "response": response,
                "error": error,
                "fatal": fatal,
            }
        )

    # Compatibility aliases used by some early orchestration prototypes.
    complete_json = request_json
    complete_text = request_text


NovelLLMClient = LLMClient


__all__ = [
    "Message",
    "Validator",
    "AuditCallback",
    "FatalLLMError",
    "InvalidLLMResponse",
    "ChatBackend",
    "OpenAICompatibleBackend",
    "parse_json_object",
    "LLMClient",
    "NovelLLMClient",
]
