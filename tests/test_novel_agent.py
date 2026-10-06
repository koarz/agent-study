from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from collections import deque
from pathlib import Path

from novel_agent.llm import FatalLLMError, LLMClient
from novel_agent.models import (
    Chapter,
    ChapterPlan,
    Character,
    Foreshadow,
    NovelBrief,
    NovelProject,
    StoryBible,
)


class ScriptedBackend:
    """Offline backend that returns scripted responses and records every request."""

    def __init__(self, responses: list[str | Exception]):
        self.responses = deque(responses)
        self.calls: list[dict[str, object]] = []

    async def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        self.calls.append({"messages": messages, "temperature": temperature})
        if not self.responses:
            raise AssertionError("fake backend response queue is empty")
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


class NovelWorkflowBackend:
    """Semantic fake for the complete planning/writing workflow.

    It recognizes the stable task wording from the prompt builders instead of
    relying on a fragile exact call count.  Every response is local JSON/text.
    """

    def __init__(self):
        self.calls: list[list[dict[str, str]]] = []
        self.plan_number = 0
        self.draft_number = 0
        self.memory_number = 0

    async def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        self.calls.append(messages)
        prompt = "\n".join(message["content"] for message in messages)
        if "设计一部能够按章连续创作的完整小说" in prompt:
            return json.dumps(self._story_bible(), ensure_ascii=False)
        if "把目标章节扩展为可直接写作的场景节拍表" in prompt:
            self.plan_number += 1
            return json.dumps(self._chapter_plan(self.plan_number), ensure_ascii=False)
        if "根据资料写出本章完整正文" in prompt:
            self.draft_number += 1
            return self._draft(self.draft_number)
        if "审查本章草稿" in prompt:
            return json.dumps(self._review(), ensure_ascii=False)
        if "按照审稿意见改写本章" in prompt:
            raise AssertionError("passing reviews must not trigger a revision call")
        if "从定稿正文提取后续创作所需的增量记忆" in prompt:
            self.memory_number += 1
            return json.dumps(self._memory_update(self.memory_number), ensure_ascii=False)
        raise AssertionError(f"unrecognized novel-agent prompt: {prompt[:160]}")

    @staticmethod
    def _story_bible() -> dict[str, object]:
        return {
            "synopsis": "失忆邮差收到来自未来的信，在时间机构的追杀下寻找被抹去的过去，并阻止一座港口陷入永恒的失序。",
            "short_synopsis": "一封来自未来的信，把失忆邮差拖进了时间阴谋。",
            "world_setting": "雾港每逢午夜会失去一小时。",
            "central_conflict": "林岚必须在港口彻底遗忘前找出寄信者。",
            "themes": ["记忆", "选择"],
            "tone": "克制而紧张",
            "style_guide": ["第三人称限知", "对白简洁克制"],
            "rules": ["午夜后钟表倒退一小时", "第二章前不得揭示寄信者身份"],
            "characters": [
                {
                    "id": "char_linlan",
                    "name": "林岚",
                    "role": "主角",
                    "description": "失忆的雾港邮差",
                    "age": "二十八岁",
                    "appearance": "总穿旧邮差制服",
                    "personality": ["谨慎", "执着"],
                    "background": "曾主动舍弃一段记忆。",
                    "goals": ["找到寄信者", "守住雾港的共同记忆"],
                    "conflicts": ["拒绝相信自己", "害怕彻底遗忘"],
                    "arc": "从逃避过去到承担自己的选择。",
                    "relationships": {},
                    "state": {"location": "旧邮局", "knows": [], "inventory": []},
                }
            ],
            "outline": [
                {
                    "number": 1,
                    "title": "没有邮戳的信",
                    "summary": "林岚收到未来来信并发现铜钥匙。",
                    "purpose": "让林岚决定追查来信。",
                    "pov": "林岚",
                    "characters": ["char_linlan"],
                    "beats": ["发现未来日期", "从信封夹层取得铜钥匙"],
                    "foreshadow_ids": ["clue_key"],
                    "target_words": 800,
                    "status": "planned",
                },
                {
                    "number": 2,
                    "title": "倒走的钟",
                    "summary": "林岚用钥匙打开钟楼暗门。",
                    "purpose": "推进谜团并回收铜钥匙。",
                    "pov": "林岚",
                    "characters": ["char_linlan"],
                    "beats": ["攀爬循环楼梯", "用钥匙打开暗门"],
                    "foreshadow_ids": ["clue_key"],
                    "target_words": 800,
                    "status": "planned",
                },
            ],
            "foreshadows": [
                {
                    "id": "clue_key",
                    "description": "信封夹层中的铜钥匙将在钟楼回收。",
                    "status": "planned",
                    "plant_chapter": 1,
                    "payoff_chapter": 2,
                    "related_characters": ["char_linlan"],
                    "notes": ["第二章用于打开暗门"],
                },
            ],
        }

    @staticmethod
    def _chapter_plan(number: int) -> dict[str, object]:
        if number == 1:
            return {
                "number": 1,
                "title": "没有邮戳的信",
                "summary": "林岚因未来来信而决定前往钟楼。",
                "purpose": "让林岚决定追查未来来信。",
                "pov": "林岚",
                "characters": ["char_linlan"],
                "beats": ["发现未来日期", "铜钥匙从夹层滑落", "决定前往钟楼"],
                "foreshadow_ids": ["clue_key"],
                "target_words": 800,
                "status": "planned",
            }
        return {
            "number": 2,
            "title": "倒走的钟",
            "summary": "林岚用钥匙终止循环并进入暗门。",
            "purpose": "推进谜团并回收铜钥匙。",
            "pov": "林岚",
            "characters": ["char_linlan"],
            "beats": ["攀爬循环楼梯", "用钥匙终止循环", "暗门后传来自己的声音"],
            "foreshadow_ids": ["clue_key"],
            "target_words": 800,
            "status": "planned",
        }

    @staticmethod
    def _draft(number: int) -> str:
        if number == 1:
            return "潮湿的清晨，林岚从分拣台上拾起一封没有邮戳的信。日期写着明天。信封夹层裂开，一把刻着钟楼编号的铜钥匙落进他的掌心。"
        return "午夜的楼梯一次次把林岚送回原地。他把铜钥匙插进暗门，倒走的钟声忽然恢复了方向。门缝里传出的，却是他自己的声音。"

    @staticmethod
    def _review() -> dict[str, object]:
        return {
            "score": 96,
            "decision": "pass",
            "strengths": ["完成章节目标且保持限知视角"],
            "issues": [],
            "continuity_checks": [
                {"constraint": "人物知识边界", "result": "pass", "note": "未提前揭示寄信者"}
            ],
            "foreshadowing_checks": [
                {"clue_id": "clue_key", "result": "pass", "note": "按计划处理"}
            ],
            "revision_instructions": [],
        }

    @staticmethod
    def _memory_update(number: int) -> dict[str, object]:
        first = number == 1
        return {
            "summary": (
                "林岚因收到未来来信并发现铜钥匙，决定前往钟楼。"
                if first
                else "林岚用铜钥匙终止钟楼循环并打开暗门，听见自己的声音。"
            ),
            "timeline_events": [
                {
                    "order": 1,
                    "time": "第一日清晨" if first else "第一日午夜",
                    "location": "旧邮局" if first else "钟楼",
                    "event": "发现未来来信" if first else "打开钟楼暗门",
                }
            ],
            "character_updates": {
                "char_linlan": {
                    "location": "旧邮局" if first else "钟楼暗门",
                    "physical_state": "无伤",
                    "emotional_state": "决心追查" if first else "震惊但清醒",
                    "knows": ["钥匙属于钟楼"] if first else ["门后声音与自己相同"],
                    "goals": ["前往钟楼"] if first else ["查明声音来源"],
                    "inventory": ["铜钥匙"] if first else [],
                }
            },
            "relationship_updates": [],
            "new_facts": [
                {
                    "id": f"fact_{number}",
                    "fact": "铜钥匙刻有钟楼编号" if first else "铜钥匙能终止钟楼时间循环",
                    "source": "本章正文",
                }
            ],
            "foreshadow_updates": {"clue_key": "planted" if first else "resolved"},
            "unresolved_threads": ["寄信者是谁"] if first else ["门后为何是林岚的声音"],
            "continuity_warnings": [],
            "next_chapter_context": "林岚带钥匙前往钟楼" if first else "林岚站在已打开的暗门前",
        }


class RevisionWorkflowBackend(NovelWorkflowBackend):
    def __init__(self):
        super().__init__()
        self.revision_number = 0

    async def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        prompt = "\n".join(message["content"] for message in messages)
        if "按照审稿意见改写本章" in prompt:
            self.calls.append(messages)
            self.revision_number += 1
            return "潮湿的清晨，林岚仔细核对日期，确认那封无邮戳的信来自明天。铜钥匙从夹层滑落，他没有猜出寄信者，只决定前往钟楼查证。"
        return await super().complete(messages, temperature)

    @staticmethod
    def _review() -> dict[str, object]:
        return {
            "score": 55,
            "decision": "pass",
            "strengths": [],
            "issues": [
                {
                    "severity": "major",
                    "category": "continuity",
                    "evidence": "原稿暗示已经知道寄信者",
                    "problem": "提前泄露谜底",
                    "fix": "保留怀疑，不确认身份",
                }
            ],
            "continuity_checks": [],
            "foreshadowing_checks": [],
            "revision_instructions": ["删除提前揭示"],
        }


class FakeRetriever:
    def __init__(self):
        self.prepare_calls: list[tuple[str, bool]] = []
        self.retrieve_calls = 0
        self.indexed_chapters: list[int] = []

    async def prepare(self, project, project_dir, rebuild: bool = False):
        self.prepare_calls.append((str(project_dir), rebuild))

    async def retrieve(self, project, plan, memory):
        self.retrieve_calls += 1
        return [
            {
                "chapter": 1,
                "kind": "chapter_chunk",
                "score": 0.93,
                "text": "RAG_SECRET_早期章节中铜钥匙曾沾有海盐。",
            }
        ]

    async def index_chapter(self, chapter, memory_update=None):
        self.indexed_chapters.append(chapter.number)


class FailingPreflightRetriever:
    async def preflight(self, project_dir):
        raise ValueError("本地 embedding 未安装")


def sample_project() -> NovelProject:
    brief = NovelBrief(
        title="雾港来信",
        premise="失忆的邮差在封闭海港追查一封来自未来的信。",
        genre="悬疑奇幻",
        target_chapters=2,
        target_words_per_chapter=800,
        tone="克制而紧张",
        style="第三人称限知",
        themes=["记忆", "选择"],
    )
    bible = StoryBible(
        world_setting="雾港每逢午夜会失去一小时。",
        central_conflict="林岚必须在港口彻底遗忘前找出寄信者。",
        themes=["记忆", "选择"],
        tone="克制而紧张",
        style_guide=["使用具体感官细节", "不使用全知视角"],
        rules=["午夜十二点后钟表会倒退一小时"],
        characters=[
            Character(
                id="char_linlan",
                name="林岚",
                role="主角",
                description="雾港邮差",
                personality=["谨慎", "执着"],
                goals=["找到寄信者"],
                state={"location": "旧邮局", "knows": []},
            )
        ],
        outline=[
            ChapterPlan(
                number=1,
                title="没有邮戳的信",
                summary="林岚收到写有明日日期的信。",
                purpose="引出谜团并埋下铜钥匙伏笔",
                pov="林岚",
                characters=["char_linlan"],
                beats=["收信", "发现未来日期", "找到铜钥匙"],
                foreshadow_ids=["hook_key"],
                target_words=800,
            ),
            ChapterPlan(
                number=2,
                title="倒走的钟",
                summary="林岚用钥匙打开钟楼暗门。",
                purpose="推进谜团并回收铜钥匙",
                pov="林岚",
                characters=["char_linlan"],
                beats=["进入钟楼", "打开暗门"],
                foreshadow_ids=["hook_key"],
                target_words=800,
            ),
        ],
        foreshadows=[
            Foreshadow(
                id="hook_key",
                description="信封里藏着一把铜钥匙",
                status="planned",
                related_characters=["char_linlan"],
            )
        ],
    )
    return NovelProject(brief=brief, bible=bible)


class LLMClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_json_is_retried_with_correction_message(self):
        backend = ScriptedBackend(
            [
                "这不是 JSON",
                '```json\n{"answer": "ok"}\n```',
            ]
        )
        client = LLMClient(backend, temperature=0.25, max_retries=2)

        result = await client.request_json(
            purpose="retry_contract",
            messages=[{"role": "user", "content": "只返回 JSON"}],
            validator=lambda value: None if value.get("answer") == "ok" else "answer 无效",
            schema_hint='{"answer": "ok"}',
        )

        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[0]["temperature"], 0.25)
        retry_messages = backend.calls[1]["messages"]
        retry_text = "\n".join(item["content"] for item in retry_messages)
        self.assertIn("这不是 JSON", retry_text)
        self.assertIn("answer", retry_text)

    async def test_fictional_balance_phrase_is_not_treated_as_provider_failure(self):
        backend = ScriptedBackend(["他看着余额不足的账本，决定先离开酒馆。"])
        client = LLMClient(backend, max_retries=1)
        result = await client.request_text(
            messages=[{"role": "user", "content": "写一句正文"}],
            purpose="prose_false_positive",
        )
        self.assertIn("余额不足", result)

    async def test_provider_quota_body_is_fatal_without_retry(self):
        backend = ScriptedBackend(["Quota exhausted. Please recharge.", "备用正文"])
        client = LLMClient(backend, max_retries=3)
        with self.assertRaises(FatalLLMError):
            await client.request_text(
                messages=[{"role": "user", "content": "写正文"}],
                purpose="quota_contract",
            )
        self.assertEqual(len(backend.calls), 1)


class NovelAgentWorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_project_write_two_chapters_and_resume_from_disk(self):
        from novel_agent.engine import NovelAgent

        backend = NovelWorkflowBackend()
        brief = NovelBrief(
            title="雾港来信",
            premise="失忆的邮差在封闭海港追查一封来自未来的信。",
            genre="悬疑奇幻",
            target_chapters=2,
            target_words_per_chapter=800,
            tone="克制而紧张",
            style="第三人称限知",
            themes=["记忆", "选择"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            agent = NovelAgent(backend, output_dir=Path(temp_dir), temperature=0.2)
            project = await agent.create_project(brief, project_name="offline-smoke")

            self.assertEqual(project.brief.title, "雾港来信")
            self.assertEqual(len(project.bible.characters), 1)
            self.assertEqual([plan.number for plan in project.bible.outline], [1, 2])
            self.assertEqual(len(project.bible.foreshadows), 1)
            self.assertIn("失忆邮差", project.bible.synopsis)
            self.assertIn("未来的信", project.bible.short_synopsis)

            await agent.write_chapters(project, 2)

            self.assertEqual(project.current_chapter, 2)
            self.assertEqual([chapter.number for chapter in project.chapters], [1, 2])
            self.assertTrue(all(chapter.content.strip() for chapter in project.chapters))
            self.assertEqual(project.status, "completed")
            self.assertIn("naturalness_report", project.chapters[0].metadata["review"])
            memory = project.metadata["memory"]
            self.assertEqual(len(memory["chapter_summaries"]), 2)
            self.assertEqual(len(memory["facts"]), 2)
            self.assertEqual(
                memory["character_updates"]["char_linlan"]["knows"],
                ["钥匙属于钟楼", "门后声音与自己相同"],
            )

            project_dir = Path(temp_dir) / "offline-smoke"
            self.assertTrue((project_dir / "project.json").is_file())
            self.assertTrue((project_dir / "chapters" / "001.md").is_file())
            self.assertTrue((project_dir / "chapters" / "002.md").is_file())
            self.assertTrue((project_dir / "synopsis.md").is_file())
            self.assertIn("失忆邮差", (project_dir / "synopsis.md").read_text(encoding="utf-8"))

            loaded = NovelProject.load(project_dir)
            self.assertEqual(loaded.to_dict(), project.to_dict())
            self.assertEqual(loaded.current_chapter, 2)

        self.assertEqual(backend.plan_number, 2)
        self.assertEqual(backend.draft_number, 2)
        self.assertEqual(backend.memory_number, 2)

    async def test_generate_synopsis_refreshes_legacy_project_and_persists_copy(self):
        from novel_agent.engine import NovelAgent

        project = sample_project()
        response = {
            "synopsis": "一名失去过去的邮差收到来自未来的信，被迫在追杀与时间谜团中寻找真相。",
            "short_synopsis": "来自未来的信，打开了邮差被抹去的人生。",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "legacy-project"
            project.save(project_dir)
            loaded = NovelProject.load(project_dir)
            self.assertEqual(loaded.bible.synopsis, "")

            backend = ScriptedBackend([json.dumps(response, ensure_ascii=False)])
            agent = NovelAgent(backend, output_dir=Path(temp_dir))
            result = await agent.generate_synopsis(loaded)

            self.assertEqual(result["short_synopsis"], response["short_synopsis"])
            persisted = NovelProject.load(project_dir)
            self.assertEqual(persisted.bible.synopsis, response["synopsis"])
            self.assertEqual(persisted.bible.short_synopsis, response["short_synopsis"])
            synopsis_file = project_dir / "synopsis.md"
            self.assertTrue(synopsis_file.is_file())
            file_text = synopsis_file.read_text(encoding="utf-8")
            self.assertIn(response["synopsis"], file_text)
            self.assertIn(response["short_synopsis"], file_text)
            prompt = "\n".join(item["content"] for item in backend.calls[0]["messages"])
            self.assertIn("雾港来信", prompt)
            self.assertIn("不要透露最终结局", prompt)

    async def test_major_review_issue_triggers_revision_before_commit(self):
        from novel_agent.engine import NovelAgent

        backend = RevisionWorkflowBackend()
        brief = NovelBrief(
            title="雾港来信",
            premise="失忆的邮差在封闭海港追查一封来自未来的信。",
            target_chapters=2,
            target_words_per_chapter=800,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            agent = NovelAgent(backend, output_dir=Path(temp_dir), temperature=0.2)
            project = await agent.create_project(brief, project_name="revision-smoke")
            chapter = await agent.write_next_chapter(project)

            self.assertIsNotNone(chapter)
            self.assertEqual(backend.revision_number, 1)
            self.assertIn("没有猜出寄信者", chapter.content)
            self.assertEqual(chapter.status, "final")
            self.assertTrue(chapter.review_notes)

    async def test_resume_keeps_audit_sequence_and_accumulates_usage(self):
        from novel_agent.engine import NovelAgent

        brief = NovelBrief(
            title="雾港来信",
            premise="失忆的邮差在封闭海港追查一封来自未来的信。",
            target_chapters=2,
            target_words_per_chapter=800,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            first_backend = NovelWorkflowBackend()
            first_agent = NovelAgent(first_backend, output_dir=Path(temp_dir), temperature=0.2)
            project = await first_agent.create_project(brief, project_name="resume-smoke")
            await first_agent.write_next_chapter(project)
            first_usage = project.metadata["usage"]["calls"]

            project_dir = Path(temp_dir) / "resume-smoke"
            resumed = NovelProject.load(project_dir)
            resumed.metadata["project_dir"] = str(project_dir)
            second_backend = NovelWorkflowBackend()
            second_backend.plan_number = 1
            second_backend.draft_number = 1
            second_backend.memory_number = 1
            second_agent = NovelAgent(second_backend, output_dir=Path(temp_dir), temperature=0.2)
            await second_agent.write_next_chapter(resumed)

            calls = sorted((project_dir / "audit" / "llm_calls").glob("*.json"))
            self.assertEqual([item.stem for item in calls], [f"{i:04d}" for i in range(1, 10)])
            self.assertEqual(first_usage, 5)
            self.assertEqual(resumed.metadata["usage"]["calls"], 9)

    async def test_implicit_same_title_projects_get_distinct_reserved_directories(self):
        from novel_agent.engine import NovelAgent

        brief = NovelBrief(
            title="同名故事",
            premise="两个独立的创作项目使用同一个标题。",
            target_chapters=2,
            target_words_per_chapter=800,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            first = await NovelAgent(
                NovelWorkflowBackend(), output_dir=Path(temp_dir)
            ).create_project(brief)
            second = await NovelAgent(
                NovelWorkflowBackend(), output_dir=Path(temp_dir)
            ).create_project(brief)

            self.assertNotEqual(first.metadata["project_dir"], second.metadata["project_dir"])
            self.assertTrue(Path(first.metadata["project_dir"]).is_dir())
            self.assertTrue(Path(second.metadata["project_dir"]).is_dir())

    async def test_rag_history_is_used_for_writing_but_not_memory_extraction(self):
        from novel_agent.engine import NovelAgent

        backend = NovelWorkflowBackend()
        retriever = FakeRetriever()
        brief = NovelBrief(
            title="雾港来信",
            premise="失忆的邮差追查一封来自未来的信。",
            target_chapters=2,
            target_words_per_chapter=800,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = NovelAgent(
                backend,
                output_dir=Path(temp_dir),
                retriever=retriever,
            )
            project = await agent.create_project(brief, project_name="rag-engine-smoke")
            await agent.write_next_chapter(project)

        prompts = ["\n".join(item["content"] for item in messages) for messages in backend.calls]
        planning = next(item for item in prompts if "把目标章节扩展为可直接写作" in item)
        drafting = next(item for item in prompts if "根据资料写出本章完整正文" in item)
        reviewing = next(item for item in prompts if "审查本章草稿" in item)
        memory = next(item for item in prompts if "从定稿正文提取后续创作所需的增量记忆" in item)
        self.assertIn("RAG_SECRET", planning)
        self.assertIn("RAG_SECRET", drafting)
        self.assertIn("RAG_SECRET", reviewing)
        self.assertNotIn("在时间机构的追杀下寻找被抹去的过去", planning)
        self.assertNotIn("在时间机构的追杀下寻找被抹去的过去", drafting)
        self.assertNotIn("RAG_SECRET", memory)
        self.assertEqual(retriever.retrieve_calls, 1)
        self.assertEqual(retriever.indexed_chapters, [1])

    async def test_disabling_deai_review_removes_style_signals_from_engine_prompts(self):
        from novel_agent.engine import NovelAgent

        backend = NovelWorkflowBackend()
        brief = NovelBrief(
            title="关闭文风审稿",
            premise="测试关闭本地去模板化检查后仍能正常写作。",
            target_chapters=2,
            target_words_per_chapter=800,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = NovelAgent(
                backend,
                output_dir=Path(temp_dir),
                ai_style_review=False,
            )
            project = await agent.create_project(brief, project_name="deai-off")
            chapter = await agent.write_next_chapter(project)

            self.assertNotIn("naturalness_report", chapter.metadata["review"])
            prompts = [
                "\n".join(item["content"] for item in messages)
                for messages in backend.calls
            ]
            draft_prompt = next(item for item in prompts if "根据资料写出本章完整正文" in item)
            review_prompt = next(item for item in prompts if "审查本章草稿" in item)
            self.assertNotIn("natural_prose_rules", draft_prompt)
            self.assertIn("style_checks 必须返回空数组", review_prompt)

    async def test_standalone_polish_updates_existing_chapter_without_regenerating_memory(self):
        from novel_agent.engine import NovelAgent

        project = sample_project()
        original = (
            "林岚猛地抬头，仿佛一道看不见的闪电劈过房间。"
            "紧接着，他咬紧牙关，心脏狂跳。"
        ) * 30
        candidate = (
            original.replace("猛地", "")
            .replace("仿佛一道看不见的闪电劈过房间", "窗外的钟声停了一拍")
            .replace("紧接着", "随后")
            .replace("咬紧牙关", "按住桌沿")
            .replace("心脏狂跳", "没有说话")
        )
        memory_update = {"summary": "林岚听见钟声。", "new_facts": []}
        chapter = Chapter(
            number=1,
            title="没有邮戳的信",
            content=original,
            summary="林岚听见钟声。",
            status="final",
            metadata={"memory_update": memory_update},
        )
        project.chapters = [chapter]
        project.current_chapter = 1
        project.status = "writing"
        project.bible.outline[0].status = "final"
        review = NovelWorkflowBackend._review()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "polish-project"
            project.save(project_dir)
            check_backend = ScriptedBackend([json.dumps(review, ensure_ascii=False)])
            check_agent = NovelAgent(check_backend, output_dir=Path(temp_dir))
            check_result = await check_agent.polish_chapter(project, 1, apply=False)
            self.assertFalse(check_result["changed"])
            self.assertEqual(project.chapters[0].content, original)
            self.assertNotIn("deai_review", project.chapters[0].metadata)

            backend = ScriptedBackend(
                [json.dumps(review, ensure_ascii=False), candidate]
            )
            agent = NovelAgent(backend, output_dir=Path(temp_dir))
            result = await agent.polish_chapter(project, 1, apply=True)

            self.assertTrue(result["changed"])
            self.assertEqual(project.chapters[0].content, candidate)
            self.assertEqual(
                project.chapters[0].metadata["memory_update"],
                memory_update,
            )
            self.assertIn("deai_review", project.chapters[0].metadata)
            self.assertTrue(Path(result["backup"]).is_file())
            saved = (project_dir / "chapters" / "001.md").read_text(encoding="utf-8")
            self.assertIn("窗外的钟声停了一拍", saved)
            self.assertNotIn("仿佛一道看不见的闪电", saved)

    async def test_rag_preflight_fails_before_story_llm_and_cleans_reserved_directory(self):
        from novel_agent.engine import NovelAgent

        backend = NovelWorkflowBackend()
        brief = NovelBrief(
            title="预检失败",
            premise="本地 embedding 尚未安装。",
            target_chapters=2,
            target_words_per_chapter=800,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                await NovelAgent(
                    backend,
                    output_dir=Path(temp_dir),
                    retriever=FailingPreflightRetriever(),
                ).create_project(brief, project_name="preflight-failure")
            self.assertEqual(backend.calls, [])
            self.assertFalse((Path(temp_dir) / "preflight-failure").exists())


class NovelProjectPersistenceTest(unittest.TestCase):
    def test_project_save_load_round_trip_after_two_chapters(self):
        project = sample_project()
        project.chapters.extend(
            [
                Chapter(
                    number=1,
                    title="没有邮戳的信",
                    content="潮湿的清晨，林岚在分拣台上发现了那封信。",
                    summary="林岚收到未来来信，并找到铜钥匙。",
                    status="final",
                    character_updates={"char_linlan": {"knows": ["信来自未来"]}},
                    foreshadow_updates={"hook_key": "planted"},
                ),
                Chapter(
                    number=2,
                    title="倒走的钟",
                    content="午夜钟声倒着响起，铜钥匙打开了钟楼暗门。",
                    summary="林岚进入钟楼暗门。",
                    status="final",
                    character_updates={"char_linlan": {"location": "钟楼暗门"}},
                    foreshadow_updates={"hook_key": "resolved"},
                ),
            ]
        )
        project.current_chapter = 2
        project.status = "completed"
        project.bible.outline[0].status = "final"
        project.bible.outline[1].status = "final"
        project.bible.foreshadows[0].status = "resolved"
        project.bible.foreshadows[0].plant_chapter = 1
        project.bible.foreshadows[0].payoff_chapter = 2

        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "fog-harbor"
            saved_file = project.save(project_dir)
            loaded = NovelProject.load(project_dir)

            self.assertEqual(saved_file, project_dir / "project.json")
            self.assertEqual(loaded.to_dict(), project.to_dict())
            self.assertEqual([chapter.number for chapter in loaded.chapters], [1, 2])
            self.assertEqual(loaded.bible.foreshadows[0].status, "resolved")
            raw = json.loads(saved_file.read_text(encoding="utf-8"))
            self.assertEqual(raw["current_chapter"], 2)

    def test_loading_a_copied_project_rebinds_runtime_path(self):
        project = sample_project()
        with tempfile.TemporaryDirectory() as temp_dir:
            original = Path(temp_dir) / "original"
            copied = Path(temp_dir) / "copied"
            project.save(original)
            shutil.copytree(original, copied)

            loaded = NovelProject.load(copied)

            self.assertEqual(loaded.metadata["project_dir"], str(copied.resolve()))
            self.assertEqual(loaded.to_dict(), project.to_dict())

    def test_foreshadow_status_cannot_regress(self):
        from novel_agent.engine import NovelAgent

        project = sample_project()
        project.bible.foreshadows[0].status = "resolved"
        project.bible.foreshadows[0].plant_chapter = 1
        project.bible.foreshadows[0].payoff_chapter = 2
        project.metadata["memory"] = {
            "foreshadow_updates": {"hook_key": "resolved"}
        }

        NovelAgent._apply_memory(
            project,
            {
                "summary": "后续章节继续追查声音。",
                "character_updates": {},
                "foreshadow_updates": {"hook_key": "planned"},
            },
            3,
        )

        self.assertEqual(project.bible.foreshadows[0].status, "resolved")
        self.assertEqual(project.bible.foreshadows[0].payoff_chapter, 2)
        self.assertEqual(project.metadata["memory"]["foreshadow_updates"]["hook_key"], "resolved")

    def test_revision_guard_rejects_missing_protected_information(self):
        from novel_agent.engine import NovelAgent

        project = sample_project()
        plan = project.bible.outline[0]
        reason = NovelAgent._revision_candidate_error(
            "林岚看着停在 3:14 的怀表，决定继续追查。",
            "他决定离开。",
            project,
            plan,
        )
        self.assertIsNotNone(reason)


if __name__ == "__main__":
    unittest.main()
