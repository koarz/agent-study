from __future__ import annotations

import asyncio
import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

from werewolf_agent.game import GameConfig, WerewolfGame
from werewolf_agent.llm import FatalLLMError
from werewolf_agent.models import Role


class DeterministicBackend:
    """Produces legal JSON decisions while leaving a trace in wolf-only speech."""

    def __init__(self):
        self.calls: list[list[dict[str, str]]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        self.calls.append(messages)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            prompt = messages[-1]["content"]
            task = prompt.split("任务：", 1)[-1].split("\n输出格式：", 1)[0]
            ids = list(dict.fromkeys(re.findall(r"P\d{2}", task)))
            schema = prompt.split("输出格式：", 1)[-1]
            if '"action"' in schema:
                return json.dumps({"action": "none", "target": None}, ensure_ascii=False)
            if '"speech"' in schema and '"target"' in schema:
                return json.dumps({"speech": "WOLF_SECRET_TOKEN", "target": ids[0]}, ensure_ascii=False)
            if '"speech"' in schema:
                return json.dumps({"speech": "我根据当前公开信息进行判断。"}, ensure_ascii=False)
            if "玩家ID或null" in schema:
                return json.dumps({"target": None}, ensure_ascii=False)
            return json.dumps({"target": ids[0]}, ensure_ascii=False)
        finally:
            self.in_flight -= 1


class QuotaExceededBackend:
    def __init__(self):
        self.calls = 0

    async def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        self.calls += 1
        return "Quota exhausted. Please recharge your account."


class WerewolfGameTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.backend = DeterministicBackend()
        self.game = WerewolfGame(
            self.backend,
            GameConfig(output_dir=Path(self.temp_dir.name), seed=7, max_days=8),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_game_completes_and_writes_all_artifacts(self):
        result = await self.game.run()
        self.assertIn(result["winner"], {"wolves", "village", "draw"})
        run_dir = self.game.run_dir
        for relative in (
            "config.json",
            "console.log",
            "result.json",
            "public_events.jsonl",
            "public_transcript.md",
            "audit/events.jsonl",
            "audit/roles.json",
            "factions/wolves.jsonl",
        ):
            self.assertTrue((run_dir / relative).is_file(), relative)
        self.assertTrue(list((run_dir / "audit/llm_calls").glob("*.json")))
        for player in self.game.players:
            self.assertTrue((run_dir / "players" / player.player_id / "visible_events.jsonl").is_file())

    async def test_wolf_chat_never_reaches_non_wolf_context(self):
        await self.game.run()
        non_wolf_ids = {p.player_id for p in self.game.players if p.role is not Role.WEREWOLF}
        for messages in self.backend.calls:
            system = messages[0]["content"]
            player_id = re.search(r"（(P\d{2})）", system).group(1)
            combined = "\n".join(message["content"] for message in messages)
            if player_id in non_wolf_ids:
                self.assertNotIn("WOLF_SECRET_TOKEN", combined)

    async def test_public_log_contains_no_wolf_secret(self):
        await self.game.run()
        public_log = (self.game.run_dir / "public_events.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("WOLF_SECRET_TOKEN", public_log)
        wolf_log = (self.game.run_dir / "factions/wolves.jsonl").read_text(encoding="utf-8")
        self.assertIn("WOLF_SECRET_TOKEN", wolf_log)

    async def test_independent_model_calls_run_concurrently(self):
        await self.game.run()
        self.assertGreater(self.backend.max_in_flight, 1)

    async def test_verbose_mode_prints_live_progress(self):
        self.game.recorder.verbose = True
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            await self.game.run()
        terminal = output.getvalue()
        self.assertIn("[LLM][请求]", terminal)
        self.assertIn("[LLM][成功]", terminal)
        self.assertIn("[公开]", terminal)

    async def test_provider_quota_error_aborts_without_retries(self):
        backend = QuotaExceededBackend()
        game = WerewolfGame(
            backend,
            GameConfig(output_dir=Path(self.temp_dir.name), seed=42, max_days=2),
        )
        with self.assertRaises(FatalLLMError):
            await game.run()
        self.assertEqual(backend.calls, 1)


if __name__ == "__main__":
    unittest.main()
