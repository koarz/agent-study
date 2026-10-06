from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .models import Player, ROLE_NAMES, Role
from .recorder import GameRecorder


class FatalLLMError(RuntimeError):
    """A provider failure that cannot be fixed by asking the model again."""


class ChatBackend(Protocol):
    async def complete(self, messages: list[dict[str, str]], temperature: float) -> str: ...


class OpenAICompatibleBackend:
    def __init__(self, *, api_key: str, base_url: str, model: str, timeout: int = 120):
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        return response.choices[0].message.content or ""


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("模型返回了空内容")
    lowered = cleaned.lower()
    fatal_markers = (
        "quota",
        "recharge",
        "topup",
        "rate limit",
        "insufficient",
        "unauthorized",
        "forbidden",
        "额度",
        "余额",
        "充值",
        "鉴权失败",
    )
    if any(marker in lowered for marker in fatal_markers):
        raise FatalLLMError(f"模型服务不可用：{cleaned[:500]}")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("模型输出必须是 JSON 对象")
    return value


@dataclass
class LLMPlayerAgent:
    player: Player
    backend: ChatBackend
    recorder: GameRecorder
    temperature: float = 0.7
    max_retries: int = 3

    async def decide(
        self,
        *,
        day: int,
        phase: str,
        purpose: str,
        task: str,
        schema: str,
        validator,
        fallback: dict[str, Any],
        private_status: str = "",
    ) -> dict[str, Any]:
        visible = self.recorder.visible_events(self.player.player_id)
        event_text = "\n".join(
            f"[{item['seq']}] 第{item['day']}天/{item['phase']} "
            f"{item.get('speaker') or '系统'}: {item['content']}"
            for item in visible[-80:]
        ) or "尚无事件。"
        role_text = ROLE_NAMES[self.player.role]
        system = (
            "你正在参加一局狼人杀。你只能使用本提示中提供的信息，绝不能假装知道其他玩家的私密行动。"
            "你的目标是帮助自己的阵营获胜。严格输出一个 JSON 对象，不要输出 Markdown、分析过程或额外文字。"
            f"你是 {self.player.name}（{self.player.player_id}），身份是{role_text}，阵营是"
            f"{'狼人' if self.player.role is Role.WEREWOLF else '好人'}。"
        )
        user = (
            f"当前：第 {day} 天，阶段 {phase}。\n"
            f"你的私有状态：{private_status or '无'}\n"
            f"你可见的完整对局记录：\n{event_text}\n\n"
            f"任务：{task}\n输出格式：{schema}"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            response = None
            self.recorder.emit(
                f"[LLM][请求] {self.player.name}({self.player.player_id}) "
                f"任务={purpose} 尝试={attempt}/{self.max_retries}"
            )
            try:
                response = await self.backend.complete(messages, self.temperature)
                value = parse_json_object(response)
                error = validator(value)
                if error:
                    raise ValueError(error)
                self.recorder.record_llm_call(
                    player_id=self.player.player_id,
                    purpose=purpose,
                    messages=messages,
                    response=response,
                    error=None,
                    attempt=attempt,
                )
                self.recorder.emit(
                    f"[LLM][成功] {self.player.name}({self.player.player_id}) 任务={purpose}"
                )
                return value
            except FatalLLMError as exc:
                last_error = str(exc)
                self.recorder.record_llm_call(
                    player_id=self.player.player_id,
                    purpose=purpose,
                    messages=messages,
                    response=response,
                    error=last_error,
                    attempt=attempt,
                )
                self.recorder.emit(
                    f"[LLM][致命错误] {self.player.name}({self.player.player_id}) "
                    f"任务={purpose} 错误={last_error}"
                )
                raise
            except Exception as exc:
                last_error = str(exc)
                response_preview = (response or "<空响应>").replace("\n", " ")[:300]
                self.recorder.record_llm_call(
                    player_id=self.player.player_id,
                    purpose=purpose,
                    messages=messages,
                    response=response,
                    error=last_error,
                    attempt=attempt,
                )
                self.recorder.emit(
                    f"[LLM][重试] {self.player.name}({self.player.player_id}) "
                    f"任务={purpose} 错误={last_error} 响应={response_preview}"
                )
                messages = messages + [
                    {"role": "assistant", "content": response or ""},
                    {"role": "user", "content": f"上次输出无效：{last_error}。只输出符合格式的 JSON。"},
                ]

        self.recorder.record(
            day=day,
            phase=phase,
            kind="llm_fallback",
            content=f"{self.player.name} 的模型连续输出无效，引擎采用合法保底动作。",
            visibility="private",
            recipients=[self.player.player_id],
            data={"purpose": purpose, "error": last_error},
        )
        return fallback
