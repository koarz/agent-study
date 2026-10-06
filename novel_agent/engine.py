"""Orchestration for the autonomous novel-writing workflow.

The engine intentionally keeps the model boundary structured.  A planning
call creates a story bible and a foreshadowing ledger; each chapter then goes
through scene planning, drafting, review, optional revision, and memory
extraction before the project is atomically checkpointed.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from .llm import AuditCallback, ChatBackend, LLMClient
from .models import (
    Chapter,
    ChapterPlan,
    Character,
    Foreshadow,
    NovelBrief,
    NovelProject,
    StoryBible,
)
from .prompts import (
    CHAPTER_PLAN_SCHEMA,
    MEMORY_UPDATE_SCHEMA,
    REVIEW_SCHEMA,
    STORY_BIBLE_SCHEMA,
    SYNOPSIS_SCHEMA,
    build_chapter_draft_messages,
    build_chapter_plan_messages,
    build_chapter_revision_messages,
    build_chapter_review_messages,
    build_memory_update_messages,
    build_synopsis_messages,
    build_story_bible_messages,
)
from .storage import ProjectStorage, atomic_write_json, atomic_write_text, resolve_project_file
from .style import analyze_prose

if TYPE_CHECKING:
    from .retrieval import ContextRetriever


class NovelGenerationError(RuntimeError):
    """Raised when a model response cannot be mapped to the project schema."""


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _jsonable(item: Any) -> Any:
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if isinstance(item, Mapping):
        return {str(k): _jsonable(v) for k, v in item.items()}
    if isinstance(item, (list, tuple, set)):
        return [_jsonable(v) for v in item]
    return item


def _text(item: Any, default: str = "") -> str:
    if item is None:
        return default
    if isinstance(item, str):
        return item
    if isinstance(item, (int, float, bool)):
        return str(item)
    try:
        return json.dumps(_jsonable(item), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(item)


def _strings(item: Any) -> list[str]:
    if item is None:
        return []
    if isinstance(item, str):
        return [item]
    if not isinstance(item, Sequence) or isinstance(item, (bytes, bytearray)):
        return [_text(item)]
    return [_text(value) for value in item if _text(value).strip()]


def _int(value: Any, default: int) -> int:
    if type(value) is int and value > 0:
        return value
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", value.strip(), flags=re.UNICODE)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-_")
    return cleaned[:80] or "novel-project"


def _beat_text(beat: Any) -> str:
    if isinstance(beat, str):
        return beat
    if isinstance(beat, Mapping):
        preferred = (
            ("action", "冲突", "conflict", "变化", "emotional_shift"),
            ("location", "地点"),
            ("information_revealed", "揭示"),
        )
        parts: list[str] = []
        for group in preferred:
            for key in group:
                if key in beat and beat[key] not in (None, "", []):
                    parts.append(_text(beat[key]))
                    break
        return "；".join(parts) or _text(beat)
    return _text(beat)


def _compact_records(values: Any, *, max_items: int, max_chars: int) -> list[Any]:
    """Keep the newest records under both item and approximate char budgets."""

    if not isinstance(values, list):
        return []
    chosen: list[Any] = []
    used = 0
    for value in reversed(values):
        size = len(_text(value))
        if chosen and (len(chosen) >= max_items or used + size > max_chars):
            break
        chosen.append(value)
        used += size
    return list(reversed(chosen))


def _status(value: Any, *, default: str = "planned") -> str:
    value = str(value or default).lower()
    aliases = {
        "seeded": "planted",
        "paid_off": "resolved",
        "paid-off": "resolved",
        "paid off": "resolved",
        "developed": "developing",
        "open": "planned",
    }
    value = aliases.get(value, value)
    return value if value in Foreshadow.STATUSES else default


def _clip_synopsis(value: str, limit: int) -> str:
    """Keep fallback/platform copy bounded without adding new prose."""

    cleaned = " ".join(str(value).split())
    if len(cleaned) <= limit:
        return cleaned
    clipped = cleaned[: max(1, limit - 1)].rstrip("，。；、:： ")
    return clipped + "…"


def _normalise_synopsis(raw: Any, brief: NovelBrief) -> tuple[str, str]:
    """Read synopsis fields from both new and common provider response shapes."""

    root: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    source: Mapping[str, Any] = root
    nested = root.get("bible") or root.get("story_bible")
    if isinstance(nested, Mapping):
        source = nested

    candidate = source.get("synopsis")
    if not candidate and source is not root:
        candidate = root.get("synopsis")
    if isinstance(candidate, Mapping):
        long_value = (
            candidate.get("long")
            or candidate.get("description")
            or candidate.get("long_synopsis")
            or candidate.get("full")
        )
        short_value = (
            candidate.get("short")
            or candidate.get("short_synopsis")
            or candidate.get("tagline")
        )
        if not short_value and source is not root:
            short_value = root.get("short_synopsis") or root.get("tagline")
    else:
        long_value = candidate or source.get("long_synopsis")
        short_value = source.get("short_synopsis") or source.get("tagline")
        if not short_value and source is not root:
            short_value = root.get("short_synopsis") or root.get("tagline")

    fallback = brief.premise.strip()
    long_text = " ".join(_text(long_value).split()) if long_value else fallback
    short_text = " ".join(_text(short_value).split()) if short_value else long_text
    return _clip_synopsis(long_text, 600), _clip_synopsis(short_text, 140)


_FORESHADOW_RANK = {
    "planned": 0,
    "planted": 1,
    "developing": 2,
    "resolved": 3,
    "abandoned": 3,
}


def _advance_foreshadow_status(previous: str, incoming: str) -> str:
    """Apply a monotonic ledger transition and preserve terminal states."""

    previous = _status(previous)
    incoming = _status(incoming)
    if previous in {"resolved", "abandoned"}:
        return previous
    if _FORESHADOW_RANK[incoming] < _FORESHADOW_RANK[previous]:
        return previous
    return incoming


def _normalise_story_bible(raw: Mapping[str, Any], brief: NovelBrief) -> StoryBible:
    """Map both the canonical schema and common richer model schemas.

    Different providers often return semantically equivalent keys such as
    ``world``/``plot``/``chapter_outlines``.  Keeping this adapter at the
    boundary lets the persisted domain model stay strict and predictable.
    """

    nested_bible = raw.get("bible") or raw.get("story_bible")
    source = nested_bible if isinstance(nested_bible, Mapping) else raw
    world = source.get("world", {}) if isinstance(source, Mapping) else {}
    plot = source.get("plot", {}) if isinstance(source, Mapping) else {}
    if not isinstance(world, Mapping):
        world = {}
    if not isinstance(plot, Mapping):
        plot = {}

    world_setting = source.get("world_setting", "")
    if not isinstance(world_setting, str) or not world_setting.strip():
        world_setting = world.get("description") or world.get("era") or ""
    rules = _strings(source.get("rules"))
    rules.extend(_strings(world.get("rules")))
    rules.extend(_strings(source.get("continuity_constraints")))
    style_guide = _strings(source.get("style_guide"))
    if isinstance(source.get("style_guide"), Mapping):
        style_guide = [f"{key}: {_text(value)}" for key, value in source["style_guide"].items()]

    characters_source = source.get("characters", []) or []
    if isinstance(characters_source, Mapping):
        characters_source = list(characters_source.values())
    characters: list[Character] = []
    for index, item in enumerate(characters_source, start=1):
        if not isinstance(item, Mapping):
            continue
        character_id = _text(item.get("id") or item.get("character_id") or f"char_{index}")
        goals = _strings(item.get("goals"))
        for key in ("desire", "need"):
            if item.get(key):
                goals.append(_text(item[key]))
        conflicts = _strings(item.get("conflicts"))
        for key in ("fear", "flaw"):
            if item.get(key):
                conflicts.append(_text(item[key]))
        relationships = item.get("relationships", {})
        if isinstance(relationships, list):
            converted: dict[str, str] = {}
            for relation in relationships:
                if isinstance(relation, Mapping):
                    rid = relation.get("id") or relation.get("character_id") or relation.get("to")
                    if rid:
                        converted[_text(rid)] = _text(relation.get("description") or relation.get("relation") or relation)
            relationships = converted
        if not isinstance(relationships, Mapping):
            relationships = {}
        background = item.get("background", "")
        if item.get("secret"):
            background = f"{_text(background)}；秘密：{_text(item['secret'])}".strip("；")
        description = item.get("description", "")
        if not description:
            description = "；".join(
                part for part in (_text(item.get("role")), _text(item.get("voice"))) if part
            )
        characters.append(
            Character(
                id=character_id,
                name=_text(item.get("name") or character_id),
                role=_text(item.get("role")),
                description=_text(description),
                age=_text(item.get("age")),
                appearance=_text(item.get("appearance")),
                personality=_strings(item.get("personality")),
                background=_text(background),
                goals=list(dict.fromkeys(goals)),
                conflicts=list(dict.fromkeys(conflicts)),
                arc=_text(item.get("arc")),
                relationships={_text(k): _text(v) for k, v in relationships.items()},
                state=dict(item.get("state", {})) if isinstance(item.get("state", {}), Mapping) else {},
            )
        )

    outline_source = (
        source.get("outline")
        or source.get("chapter_outlines")
        or source.get("chapter_plans")
        or []
    )
    if isinstance(outline_source, Mapping):
        outline_source = list(outline_source.values())
    outline: list[ChapterPlan] = []
    character_name_to_id = {item.name: item.id for item in characters}
    for index, item in enumerate(outline_source, start=1):
        if not isinstance(item, Mapping):
            continue
        number = _int(item.get("number", item.get("chapter_number")), index)
        title = _text(item.get("title") or f"第{number}章")
        summary = item.get("summary") or item.get("chapter_goal") or item.get("goal")
        if not summary:
            summary = "；".join(
                _text(item.get(key))
                for key in ("conflict", "turning_point", "ending_hook")
                if item.get(key)
            )
        characters_ids = [
            character_name_to_id.get(value, value)
            for value in _strings(item.get("characters") or item.get("participants"))
        ]
        pov = _text(item.get("pov") or item.get("pov_character"))
        pov = character_name_to_id.get(pov, pov)
        if not characters_ids and pov:
            characters_ids = [pov]
        foreshadow_ids = _strings(item.get("foreshadow_ids"))
        foreshadow_ids.extend(_strings(item.get("required_clues")))
        foreshadow_ids.extend(_strings(item.get("resolved_clues")))
        beats = [_beat_text(beat) for beat in (item.get("beats") or [])]
        outline.append(
            ChapterPlan(
                number=number,
                title=title,
                summary=_text(summary or title),
                purpose=_text(item.get("purpose") or item.get("chapter_goal") or item.get("goal")),
                pov=pov,
                characters=list(dict.fromkeys(characters_ids)),
                beats=beats,
                foreshadow_ids=list(dict.fromkeys(foreshadow_ids)),
                target_words=_int(item.get("target_words"), brief.target_words_per_chapter),
                status="planned",
            )
        )

    foreshadow_source = (
        source.get("foreshadows")
        or source.get("foreshadowing")
        or source.get("foreshadowing_ledger")
        or []
    )
    if isinstance(foreshadow_source, Mapping):
        foreshadow_source = list(foreshadow_source.values())
    foreshadows: list[Foreshadow] = []
    for index, item in enumerate(foreshadow_source, start=1):
        if not isinstance(item, Mapping):
            continue
        fid = _text(item.get("id") or item.get("clue_id") or f"clue_{index}")
        plant = item.get("plant_chapter", item.get("setup_chapter"))
        payoff = item.get("payoff_chapter")
        state = _status(item.get("status"))
        if state in {"planted", "developing", "resolved"} and plant is None:
            plant = 1
        if state == "resolved" and payoff is None:
            payoff = _int(plant, 1)
        notes = _strings(item.get("notes"))
        notes.extend(_strings(item.get("development_chapters")))
        description = item.get("description") or item.get("setup") or item.get("payoff") or fid
        if item.get("payoff") and item.get("setup"):
            description = f"{item['setup']}；回收：{item['payoff']}"
        foreshadows.append(
            Foreshadow(
                id=fid,
                description=_text(description),
                status=state,
                plant_chapter=_int(plant, 1) if plant is not None else None,
                payoff_chapter=_int(payoff, 1) if payoff is not None else None,
                related_characters=[
                    character_name_to_id.get(value, value)
                    for value in _strings(item.get("related_characters"))
                ],
                notes=notes,
            )
        )

    # A provider occasionally omits a foreshadowing ledger while referencing
    # clue IDs in the outline.  Preserve the reference as a planned item so
    # future chapter calls can still track it.
    known = {item.id for item in foreshadows}
    referenced = {fid for plan in outline for fid in plan.foreshadow_ids}
    for fid in sorted(referenced - known):
        foreshadows.append(Foreshadow(id=fid, description=f"待定义伏笔：{fid}"))

    central_conflict = source.get("central_conflict", "")
    if not isinstance(central_conflict, str) or not central_conflict.strip():
        central_conflict = plot.get("central_conflict") or brief.premise
    synopsis, short_synopsis = _normalise_synopsis(raw, brief)
    bible = StoryBible(
        world_setting=_text(world_setting),
        central_conflict=_text(central_conflict),
        themes=_strings(source.get("themes") or brief.themes),
        tone=_text(source.get("tone") or brief.tone),
        style_guide=style_guide or _strings([brief.style] if brief.style else []),
        rules=list(dict.fromkeys(rules)),
        characters=characters,
        outline=outline,
        foreshadows=foreshadows,
        synopsis=synopsis,
        short_synopsis=short_synopsis,
    )
    if not bible.outline:
        raise NovelGenerationError("故事规划没有返回任何章节大纲")
    bible.validate()
    return bible


def _normalise_scene_plan(raw: Mapping[str, Any], base: ChapterPlan) -> dict[str, Any]:
    source = raw.get("chapter_plan") if isinstance(raw.get("chapter_plan"), Mapping) else raw
    result = dict(source)
    result.setdefault("number", result.get("chapter_number", base.number))
    result.setdefault("title", base.title)
    result.setdefault("summary", result.get("chapter_goal", base.summary))
    result.setdefault("purpose", result.get("chapter_goal", base.purpose))
    result.setdefault("pov", result.get("pov_character", base.pov))
    result.setdefault("characters", base.characters)
    result.setdefault("foreshadow_ids", base.foreshadow_ids)
    result.setdefault("target_words", base.target_words)
    return result


def _normalise_review(raw: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(raw.get("review") if isinstance(raw.get("review"), Mapping) else raw)
    if "decision" in result:
        decision_value = result.get("decision")
    elif "status" in result:
        decision_value = result.get("status")
    elif "pass" in result:
        decision_value = "pass" if result.get("pass") else "revise"
    elif "approved" in result:
        decision_value = "pass" if result.get("approved") else "revise"
    else:
        decision_value = "pass"
    decision = str(decision_value).lower()
    if decision in {"approved", "ok", "accept", "通过"}:
        decision = "pass"
    elif decision in {"需要修改", "needs_revision", "needs-revision", "fail"}:
        decision = "revise"
    if decision not in {"pass", "revise"}:
        decision = "revise" if result.get("issues") else "pass"
    result["decision"] = decision
    issues = result.get("issues", [])
    result["issues"] = issues if isinstance(issues, list) else [_text(issues)]
    instructions = result.get("revision_instructions", [])
    result["revision_instructions"] = (
        instructions if isinstance(instructions, list) else [_text(instructions)]
    )
    style_checks = result.get("style_checks", [])
    result["style_checks"] = style_checks if isinstance(style_checks, list) else []
    if any(
        isinstance(issue, Mapping)
        and str(issue.get("severity", "")).lower()
        in {"critical", "major", "严重", "关键", "致命", "重要"}
        for issue in result["issues"]
    ):
        result["decision"] = "revise"
    return result


def _merge_style_report(
    review: dict[str, Any],
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Attach local naturalness signals without making an automatic verdict."""

    if not report:
        return review
    result = dict(review)
    result["naturalness_report"] = dict(report)
    raw_issues = report.get("issues", [])
    if not isinstance(raw_issues, list):
        return result
    # Keep heuristic findings in their own advisory field.  They are evidence
    # for the LLM reviewer, not proof of a defect and not automatic requests to
    # rewrite a whole chapter; this prevents a false positive from changing
    # plot or foreshadowing.
    checks = result.setdefault("style_checks", [])
    if isinstance(checks, list):
        checks.extend(
            {
                "category": "naturalness",
                "result": "fail",
                "evidence": _text(item.get("evidence")),
                "note": _text(item.get("problem")),
            }
            for item in raw_issues
            if isinstance(item, Mapping)
        )
    return result


def _normalise_memory(raw: Mapping[str, Any]) -> dict[str, Any]:
    source = raw.get("memory") if isinstance(raw.get("memory"), Mapping) else raw
    result = dict(source)
    if "summary" not in result and "chapter_summary" in result:
        result["summary"] = result["chapter_summary"]
    updates = result.get("character_updates", {})
    if isinstance(updates, list):
        converted: dict[str, Any] = {}
        for item in updates:
            if isinstance(item, Mapping):
                cid = item.get("character_id") or item.get("id")
                if cid:
                    converted[_text(cid)] = {
                        str(key): value
                        for key, value in item.items()
                        if key not in {"character_id", "id"}
                    }
        updates = converted
    result["character_updates"] = updates if isinstance(updates, Mapping) else {}
    raw_hooks = result.get("foreshadow_updates", {})
    if not raw_hooks:
        raw_hooks = result.get("foreshadowing_updates", {})
    if isinstance(raw_hooks, list):
        converted_hooks: dict[str, str] = {}
        for item in raw_hooks:
            if isinstance(item, Mapping):
                fid = item.get("clue_id") or item.get("id")
                if fid:
                    converted_hooks[_text(fid)] = _status(item.get("status"))
        raw_hooks = converted_hooks
    result["foreshadow_updates"] = {
        _text(key): _status(value) for key, value in (raw_hooks.items() if isinstance(raw_hooks, Mapping) else [])
    }
    return result


class NovelAgent:
    """Stateful orchestrator for one or more resumable novel projects."""

    def __init__(
        self,
        backend: ChatBackend,
        *,
        output_dir: str | Path = "novels",
        temperature: float = 0.8,
        max_retries: int = 3,
        revise: bool = True,
        ai_style_review: bool = True,
        progress_callback: Callable[[str], None] | None = None,
        retriever: ContextRetriever | None = None,
    ) -> None:
        self.backend = backend
        self.output_dir = Path(output_dir).expanduser()
        self.revise = revise
        self.ai_style_review = ai_style_review
        self.progress_callback = progress_callback
        self.retriever = retriever
        self._active_dir: Path | None = None
        self._audit_seq = 0
        self._usage = {"calls": 0, "input_chars": 0, "output_chars": 0}
        self.llm = LLMClient(
            backend,
            temperature=temperature,
            max_retries=max_retries,
            audit_callback=self._audit_callback,
        )

    async def _audit_callback(self, event: dict[str, Any]) -> None:
        self._audit_seq += 1
        messages = event.get("messages") or []
        response = event.get("response") or ""
        self._usage["calls"] += 1
        self._usage["input_chars"] += sum(len(_text(item.get("content"))) for item in messages)
        self._usage["output_chars"] += len(_text(response))
        if self._active_dir is None:
            return
        path = self._active_dir / "audit" / "llm_calls" / f"{self._audit_seq:04d}.json"
        atomic_write_json(path, event)

    def _progress(self, message: str) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(message)
        except Exception:
            # Progress reporting must never break a generation run.
            return

    def _activate(self, project_dir: Path) -> None:
        project_dir = project_dir.expanduser().resolve()
        if self._active_dir == project_dir:
            return
        self._active_dir = project_dir
        calls_dir = project_dir / "audit" / "llm_calls"
        calls_dir.mkdir(parents=True, exist_ok=True)
        existing_sequences = [
            int(match.group(1))
            for path in calls_dir.glob("*.json")
            if (match := re.match(r"^(\d+)", path.stem))
        ]
        self._audit_seq = max(existing_sequences, default=0)

    def _seed_usage(self, project: NovelProject) -> None:
        existing = project.metadata.get("usage", {})
        if not isinstance(existing, Mapping):
            existing = {}
        self._usage = {
            "calls": max(0, int(existing.get("calls", 0) or 0)),
            "input_chars": max(0, int(existing.get("input_chars", 0) or 0)),
            "output_chars": max(0, int(existing.get("output_chars", 0) or 0)),
        }

    @staticmethod
    def _project_dir(project: NovelProject) -> Path | None:
        value = project.metadata.get("project_dir") if isinstance(project.metadata, Mapping) else None
        return Path(value).expanduser().resolve() if value else None

    def _storage(self, project: NovelProject) -> ProjectStorage:
        project_dir = self._project_dir(project)
        if project_dir is None:
            project_dir = (self.output_dir / _slug(project.brief.title)).expanduser().resolve()
            project.metadata["project_dir"] = str(project_dir)
        self._activate(project_dir)
        self._seed_usage(project)
        return ProjectStorage(project_dir)

    def _sync_metadata(self, project: NovelProject) -> None:
        project.metadata["usage"] = dict(self._usage)
        project.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

    async def create_project(
        self,
        brief: NovelBrief,
        project_name: str | None = None,
    ) -> NovelProject:
        brief.validate()
        name = _slug(project_name or brief.title)
        project_dir = (self.output_dir / name).expanduser().resolve()
        if project_dir.exists():
            if project_name is not None:
                raise FileExistsError(f"项目目录已存在：{project_dir}")
            base_name = name
            suffix = datetime.now().strftime("%Y%m%d%H%M%S%f")
            counter = 0
            while project_dir.exists():
                counter += 1
                name = f"{base_name}-{suffix}-{counter}"
                project_dir = (self.output_dir / name).expanduser().resolve()
        # Reserve the selected directory before the first model call.  This
        # prevents two processes creating the same implicit title from
        # writing into one project; an incomplete failed attempt is harmless
        # and will receive a new suffix on the next implicit attempt.
        try:
            project_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if project_name is not None:
                raise FileExistsError(f"项目目录已被占用：{project_dir}") from None
            base_name = _slug(project_name or brief.title)
            counter = 0
            while True:
                counter += 1
                name = f"{base_name}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}-{counter}"
                project_dir = (self.output_dir / name).expanduser().resolve()
                try:
                    project_dir.mkdir(parents=True, exist_ok=False)
                    break
                except FileExistsError:
                    continue
        self._activate(project_dir)
        self._usage = {"calls": 0, "input_chars": 0, "output_chars": 0}
        if self.retriever is not None:
            preflight = getattr(self.retriever, "preflight", None)
            if callable(preflight):
                self._progress("检查 ChromaDB 和本地 embedding 模型")
                try:
                    await preflight(project_dir)
                except Exception:
                    # The directory was reserved by this create attempt and
                    # contains no project checkpoint yet.  Remove its empty
                    # audit/RAG scaffolding so an explicit-name retry is safe.
                    if not (project_dir / "project.json").exists():
                        shutil.rmtree(project_dir, ignore_errors=True)
                    raise
        self._progress("正在生成故事圣经、人物、全书大纲和伏笔账本")
        raw = await self.llm.request_json(
            build_story_bible_messages(brief),
            purpose="story_bible",
            schema_hint=STORY_BIBLE_SCHEMA,
            validator=lambda value: self._story_bible_error(value, brief),
        )
        bible = _normalise_story_bible(raw, brief)
        self._progress("正在整理作品简介")
        project = NovelProject(
            brief=brief,
            bible=bible,
            status="writing",
            metadata={
                "project_name": name,
                "project_dir": str(project_dir.expanduser().resolve()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "memory": {"summary": "", "character_updates": {}, "foreshadow_updates": {}},
            },
        )
        self._sync_metadata(project)
        project.save(project_dir)
        ProjectStorage(project_dir).save_synopsis(project)
        if self.retriever is not None:
            self._progress("初始化本地向量数据库和 embedding 模型")
            await self.retriever.prepare(project, project_dir, rebuild=False)
            project.save(project_dir)
        return project

    @staticmethod
    def _story_bible_error(value: Any, brief: NovelBrief) -> str | None:
        if not isinstance(value, Mapping):
            return "故事圣经必须是 JSON 对象"
        try:
            bible = _normalise_story_bible(value, brief)
        except Exception as exc:
            return str(exc)
        if not bible.world_setting.strip():
            return "故事圣经缺少 world_setting"
        if not bible.central_conflict.strip():
            return "故事圣经缺少 central_conflict"
        if not bible.characters:
            return "故事圣经至少需要一个角色"
        numbers = sorted(item.number for item in bible.outline)
        expected = list(range(1, brief.target_chapters + 1))
        if numbers != expected:
            return (
                f"章节大纲必须恰好包含 {brief.target_chapters} 章，"
                "并从 1 连续编号"
            )
        return None

    def load_project(self, path: str | Path) -> NovelProject:
        project_file = resolve_project_file(path)
        project = NovelProject.load(project_file)
        project_dir = project_file.parent
        project.metadata.setdefault("project_name", project_dir.name)
        project.metadata["project_dir"] = str(project_dir.expanduser().resolve())
        self._activate(project_dir)
        self._seed_usage(project)
        return project

    def _memory(self, project: NovelProject) -> dict[str, Any]:
        memory = project.metadata.get("memory", {})
        if not isinstance(memory, Mapping):
            return {}
        compact = dict(memory)
        compact["chapter_summaries"] = _compact_records(
            compact.get("chapter_summaries"), max_items=8, max_chars=6000
        )
        compact["timeline_events"] = _compact_records(
            compact.get("timeline_events"), max_items=60, max_chars=10000
        )
        compact["facts"] = _compact_records(
            compact.get("facts"), max_items=80, max_chars=12000
        )
        compact["relationship_updates"] = _compact_records(
            compact.get("relationship_updates"), max_items=40, max_chars=8000
        )
        compact["continuity_warnings"] = _compact_records(
            compact.get("continuity_warnings"), max_items=20, max_chars=4000
        )
        return compact

    def _previous_summary(self, project: NovelProject) -> str:
        if not project.chapters:
            return ""
        return project.chapters[-1].summary

    def _previous_excerpt(self, project: NovelProject, limit: int = 2400) -> str:
        if not project.chapters:
            return ""
        return project.chapters[-1].content[-limit:]

    def _next_plan(self, project: NovelProject) -> ChapterPlan | None:
        done = {chapter.number for chapter in project.chapters}
        for plan in sorted(project.bible.outline, key=lambda item: item.number):
            if plan.number not in done:
                return plan
        return None

    def _context_bible(
        self,
        project: NovelProject,
        plan: ChapterPlan | None = None,
    ) -> dict[str, Any]:
        bible = project.bible.to_dict()
        # Platform-facing copy is useful for metadata and export, but should
        # not be fed back into scene writing where it could bias the prose or
        # leak a marketing spoiler into continuity reasoning.
        bible.pop("synopsis", None)
        bible.pop("short_synopsis", None)
        # Keep the full arc in compact form, but exclude scene-level beats from
        # unrelated chapters.  Full manuscript text is never included.
        bible["outline"] = [
            {
                "number": item.number,
                "title": item.title,
                "summary": item.summary,
                "purpose": item.purpose,
                "foreshadow_ids": item.foreshadow_ids,
                "status": item.status,
            }
            for item in project.bible.outline
        ]
        if len(_text(bible["outline"])) > 12000:
            entries = bible["outline"]
            nearby_numbers = {1, 2, len(entries), len(entries) - 1}
            if plan is not None:
                nearby_numbers.update({plan.number - 1, plan.number, plan.number + 1})
            kept = [
                entry
                for entry in entries
                if entry["number"] in nearby_numbers and entry["number"] > 0
            ]
            kept.sort(key=lambda item: item["number"])
            bible["outline"] = kept + [
                {
                    "number": "…",
                    "title": "中间章节已压缩",
                    "summary": f"省略 {len(entries) - len(kept)} 个章节摘要；以当前章节计划和记忆为准。",
                    "purpose": "",
                    "foreshadow_ids": [],
                    "status": "planned",
                }
            ]
        if plan is not None:
            character_refs = set(plan.characters)
            if plan.pov:
                character_refs.add(plan.pov)
            for hook in project.bible.foreshadows:
                if hook.id in plan.foreshadow_ids:
                    character_refs.update(hook.related_characters)
            if character_refs:
                selected = [
                    item.to_dict()
                    for item in project.bible.characters
                    if item.id in character_refs or item.name in character_refs
                ]
                if selected:
                    bible["characters"] = selected
        return bible

    def _synopsis_context(self, project: NovelProject) -> dict[str, Any]:
        """Build a compact, spoiler-aware source for platform copy."""

        context = self._context_bible(project)
        context["completed_chapters"] = [
            {
                "number": chapter.number,
                "title": chapter.title,
                "summary": chapter.summary,
            }
            for chapter in project.chapters[-20:]
        ]
        return context

    @staticmethod
    def _synopsis_error(value: Any) -> str | None:
        if not isinstance(value, Mapping):
            return "作品简介结果必须是 JSON 对象"
        candidate = value.get("synopsis") or value.get("long_synopsis")
        if isinstance(candidate, Mapping):
            candidate = (
                candidate.get("long")
                or candidate.get("description")
                or candidate.get("long_synopsis")
                or candidate.get("full")
            )
        if not isinstance(candidate, str) or not candidate.strip():
            return "作品简介结果缺少 synopsis"
        return None

    async def generate_synopsis(self, project: NovelProject) -> dict[str, str]:
        """Generate or refresh platform-facing long and short descriptions."""

        storage = self._storage(project)
        self._progress("正在生成作品简介")
        raw = await self.llm.request_json(
            build_synopsis_messages(
                project.brief.to_dict(),
                self._synopsis_context(project),
            ),
            purpose="synopsis",
            schema_hint=SYNOPSIS_SCHEMA,
            validator=self._synopsis_error,
        )
        long_text, short_text = _normalise_synopsis(raw, project.brief)
        previous = (project.bible.synopsis, project.bible.short_synopsis)
        project.bible.synopsis = long_text
        project.bible.short_synopsis = short_text
        self._sync_metadata(project)
        try:
            project.validate()
            storage.save(project)
            storage.save_synopsis(project)
        except Exception:
            project.bible.synopsis, project.bible.short_synopsis = previous
            raise
        return {"synopsis": long_text, "short_synopsis": short_text}

    def _standalone_style_memory(
        self,
        project: NovelProject,
        chapter_number: int,
    ) -> dict[str, Any]:
        """Give a standalone prose review only the context it needs."""

        previous = [
            {
                "number": chapter.number,
                "title": chapter.title,
                "summary": chapter.summary,
            }
            for chapter in project.chapters
            if chapter.number < chapter_number
        ][-8:]
        return {
            "mode": "standalone_naturalness_review",
            "chapter_number": chapter_number,
            "previous_chapters": previous,
            "instruction": "只检查当前章节表达；不要用后续章节内容要求改写当前章节。",
        }

    @staticmethod
    def _style_issue_as_review_item(item: Mapping[str, Any]) -> dict[str, str]:
        return {
            "severity": _text(item.get("severity"), "minor"),
            "category": "style",
            "evidence": _text(item.get("evidence")),
            "problem": _text(item.get("problem")),
            "fix": _text(item.get("fix")),
            "source": "local_naturalness_lint",
        }

    async def polish_chapter(
        self,
        project: NovelProject,
        chapter_number: int,
        *,
        apply: bool = True,
    ) -> dict[str, Any]:
        """Review one saved chapter and optionally apply a safe prose polish.

        This path never creates a scene plan or a new chapter.  Existing
        memory/character/foreshadow state is intentionally left unchanged;
        the rewrite prompt is restricted to surface language.
        """

        storage = self._storage(project)
        chapter = next(
            (item for item in project.chapters if item.number == chapter_number),
            None,
        )
        if chapter is None:
            raise ValueError(f"第 {chapter_number} 章尚未生成，不能单独润色")
        plan = next(
            (item for item in project.bible.outline if item.number == chapter_number),
            None,
        )
        if plan is None:
            raise ValueError(f"第 {chapter_number} 章没有对应的大纲")

        style_report = analyze_prose(chapter.content) if self.ai_style_review else None
        memory = self._standalone_style_memory(project, chapter_number)
        self._progress(f"第 {chapter_number} 章：独立检查文风和模板化表达")
        review_raw = await self.llm.request_json(
            build_chapter_review_messages(
                self._context_bible(project, plan),
                plan.to_dict(),
                chapter.content,
                memory,
                style_report,
                self.ai_style_review,
            ),
            purpose=f"chapter_{chapter_number:03d}_deai_review",
            schema_hint=REVIEW_SCHEMA,
            validator=self._review_error,
        )
        review = _merge_style_report(_normalise_review(review_raw), style_report)
        result: dict[str, Any] = {
            "chapter": chapter_number,
            "changed": False,
            "review": review,
            "naturalness": style_report,
        }
        if not apply:
            return result

        local_issues = (
            style_report.get("issues", [])
            if isinstance(style_report, Mapping)
            else []
        )
        model_style_issue = any(
            isinstance(item, Mapping)
            and str(item.get("category", "")).lower()
            in {"style", "prose", "pacing", "dialogue", "rhythm", "wording"}
            for item in review.get("issues", [])
        )
        should_polish = bool(local_issues) or (
            review.get("decision") == "revise" and model_style_issue
        )
        if not should_polish:
            return result

        # A standalone polish request explicitly asks for surface cleanup. If
        # the reviewer passed but local lint found signals, turn those signals
        # into bounded revision advice; the normal generation path remains
        # reviewer-controlled and does not force a rewrite.
        revision_review = dict(review)
        if revision_review.get("decision") != "revise" and local_issues:
            revision_review["decision"] = "revise"
            revision_review["issues"] = list(revision_review.get("issues", []))
            revision_review["issues"].extend(
                self._style_issue_as_review_item(item)
                for item in local_issues
                if isinstance(item, Mapping)
            )
            instructions = revision_review.setdefault("revision_instructions", [])
            if isinstance(instructions, list):
                instructions.append(
                    "只做去模板化的局部语言调整，不改变事件、人物动机、数字、地点或伏笔。"
                )

        self._progress(f"第 {chapter_number} 章：应用安全的自然化润色")
        candidate = await self.llm.request_text(
            build_chapter_revision_messages(
                self._context_bible(project, plan),
                plan.to_dict(),
                chapter.content,
                revision_review,
                memory,
                True,
            ),
            purpose=f"chapter_{chapter_number:03d}_deai_polish",
            validator=lambda value: None
            if len(value.strip()) >= max(20, min(500, plan.target_words // 20))
            else "润色结果过短，已拒绝本次修改",
        )
        candidate_error = self._revision_candidate_error(
            chapter.content,
            candidate,
            project,
            plan,
        )
        if candidate_error:
            result["rejected"] = candidate_error
            return result
        if candidate.strip() == chapter.content.strip():
            result["unchanged"] = True
            return result

        snapshot = project.to_dict()
        old_content = chapter.content
        backup_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = (
            storage.root
            / "audit"
            / "polish"
            / f"chapter_{chapter_number:03d}_{backup_stamp}.md"
        )
        atomic_write_text(
            backup_path,
            f"# 第 {chapter.number} 章 {chapter.title}\n\n{old_content.strip()}\n",
        )
        chapter.content = candidate.strip()
        chapter.metadata["deai_review"] = review
        chapter.metadata["naturalness_report"] = style_report or {}
        history = chapter.metadata.setdefault("polish_history", [])
        if isinstance(history, list):
            history.append(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "backup": str(backup_path.relative_to(storage.root)),
                }
            )
        chapter.review_notes = list(chapter.review_notes) + self._review_notes(review)
        self._sync_metadata(project)
        try:
            project.validate()
            storage.save_chapter(chapter)
            storage.save(project)
        except Exception:
            restored = NovelProject.from_dict(snapshot)
            for key in ("project_dir", "project_name"):
                if key in project.metadata:
                    restored.metadata[key] = project.metadata[key]
            self._restore_project(project, restored)
            raise

        if self.retriever is not None:
            try:
                await self.retriever.prepare(project, storage.root, rebuild=False)
            except Exception as exc:
                self._progress(f"第 {chapter_number} 章：RAG 索引稍后重建（{exc}）")
        result["changed"] = chapter.content != old_content
        result["backup"] = str(backup_path)
        return result

    async def polish_chapters(
        self,
        project: NovelProject,
        start: int,
        count: int = 1,
        *,
        apply: bool = True,
    ) -> list[dict[str, Any]]:
        if type(start) is not int or start <= 0:
            raise ValueError("start 必须是正整数")
        if type(count) is not int or count < 1:
            raise ValueError("count 必须是正整数")
        available = {chapter.number for chapter in project.chapters}
        requested = range(start, start + count)
        missing = [number for number in requested if number not in available]
        if missing:
            raise ValueError(
                "以下章节尚未生成，不能单独润色：" + "、".join(map(str, missing))
            )
        results: list[dict[str, Any]] = []
        for number in requested:
            results.append(await self.polish_chapter(project, number, apply=apply))
        return results

    async def write_next_chapter(self, project: NovelProject, *, revise: bool | None = None) -> Chapter | None:
        storage = self._storage(project)
        plan = self._next_plan(project)
        if plan is None:
            project.current_chapter = len(project.bible.outline)
            project.status = "completed"
            self._sync_metadata(project)
            storage.save(project)
            return None

        project.status = "writing"
        memory = self._memory(project)
        prompt_memory = dict(memory)
        if self.retriever is not None:
            self._progress(f"第 {plan.number} 章：从向量数据库召回相关旧情节")
            await self.retriever.prepare(project, storage.root, rebuild=False)
            retrieved = await self.retriever.retrieve(project, plan, memory)
            if retrieved:
                prompt_memory["retrieved_history"] = retrieved
        self._progress(f"第 {plan.number} 章：生成场景计划")
        scene_raw = await self.llm.request_json(
            build_chapter_plan_messages(
                self._context_bible(project, plan),
                plan.to_dict(),
                prompt_memory,
                self._previous_summary(project),
            ),
            purpose=f"chapter_{plan.number:03d}_plan",
            schema_hint=CHAPTER_PLAN_SCHEMA,
            validator=lambda value: self._scene_plan_error(value, plan, project),
        )
        scene_plan = _normalise_scene_plan(scene_raw, plan)

        minimum_length = max(20, min(500, plan.target_words // 20))
        self._progress(f"第 {plan.number} 章：生成正文")
        style_constraints: dict[str, Any] = {
            "language": project.brief.language,
            "tone": project.bible.tone,
            "style_guide": project.bible.style_guide,
        }
        if self.ai_style_review:
            style_constraints["natural_prose_rules"] = [
                "减少连续重复的比喻、强刺激转折词和情绪套语",
                "优先用具体动作、物件、停顿和对白表现情绪",
                "保留长短句变化，不让每个段落都承担高潮功能",
            ]
        draft = await self.llm.request_text(
            build_chapter_draft_messages(
                self._context_bible(project, plan),
                scene_plan,
                prompt_memory,
                self._previous_excerpt(project),
                style_constraints,
                self.ai_style_review,
            ),
            purpose=f"chapter_{plan.number:03d}_draft",
            validator=lambda value: None
            if len(value.strip()) >= minimum_length
            else f"章节正文过短，至少需要约 {minimum_length} 个字符",
        )
        style_report = analyze_prose(draft) if self.ai_style_review else None
        self._progress(f"第 {plan.number} 章：检查人物、连续性、伏笔和文风")
        review_raw = await self.llm.request_json(
            build_chapter_review_messages(
                self._context_bible(project, plan),
                scene_plan,
                draft,
                prompt_memory,
                style_report,
                self.ai_style_review,
            ),
            purpose=f"chapter_{plan.number:03d}_review",
            schema_hint=REVIEW_SCHEMA,
            validator=self._review_error,
        )
        review = _merge_style_report(_normalise_review(review_raw), style_report)
        should_revise = (revise if revise is not None else self.revise) and review["decision"] == "revise"
        final_text = draft
        final_status = "reviewed"
        if should_revise:
            self._progress(f"第 {plan.number} 章：根据审稿意见改写")
            final_text = await self.llm.request_text(
                build_chapter_revision_messages(
                    self._context_bible(project, plan),
                    scene_plan,
                    draft,
                    review,
                    prompt_memory,
                    self.ai_style_review,
                ),
                purpose=f"chapter_{plan.number:03d}_revision",
                validator=lambda value: None
                if len(value.strip()) >= minimum_length
                else f"修订正文过短，至少需要约 {minimum_length} 个字符",
            )
            candidate_error = self._revision_candidate_error(
                draft,
                final_text,
                project,
                plan,
            )
            if candidate_error:
                review["revision_rejected"] = candidate_error
                final_text = draft
                final_status = "reviewed"
            else:
                final_status = "revised"

        if self.ai_style_review:
            review["naturalness_after_revision"] = analyze_prose(final_text)

        self._progress(f"第 {plan.number} 章：更新人物状态和伏笔账本")
        memory_raw = await self.llm.request_json(
            build_memory_update_messages(
                self._context_bible(project, plan), scene_plan, final_text, memory
            ),
            purpose=f"chapter_{plan.number:03d}_memory",
            schema_hint=MEMORY_UPDATE_SCHEMA,
            validator=lambda value: self._memory_error(value, project),
        )
        memory_update = _normalise_memory(memory_raw)

        chapter = Chapter(
            number=plan.number,
            title=plan.title,
            content=final_text,
            summary=_text(memory_update.get("summary") or plan.summary),
            status="final" if review["decision"] == "pass" or final_status == "revised" else final_status,
            review_notes=self._review_notes(review),
            character_updates=dict(memory_update.get("character_updates", {})),
            foreshadow_updates=dict(memory_update.get("foreshadow_updates", {})),
            metadata={"scene_plan": scene_plan, "review": review, "memory_update": memory_update},
        )

        # Commit state only after every model call succeeded.  This makes a
        # failed call resumable without leaving a half-written chapter.
        snapshot = project.to_dict()
        self._apply_memory(project, memory_update, chapter.number)
        chapter.foreshadow_updates = {
            hook_id: next(
                (hook.status for hook in project.bible.foreshadows if hook.id == hook_id),
                status,
            )
            for hook_id, status in chapter.foreshadow_updates.items()
        }
        project.chapters.append(chapter)
        project.current_chapter = chapter.number
        for item in project.bible.outline:
            if item.number == chapter.number:
                item.status = "final"
                break
        project.status = "completed" if self._next_plan(project) is None else "writing"
        self._sync_metadata(project)
        try:
            project.validate()
            storage.save_chapter(chapter)
            # Write the replaceable chapter artifact first.  If the subsequent
            # project checkpoint fails, a retry safely overwrites this orphan;
            # the inverse order could advance state while leaving no chapter file.
            storage.save(project)
        except Exception:
            restored = NovelProject.from_dict(snapshot)
            for key in ("project_dir", "project_name"):
                if key in project.metadata:
                    restored.metadata[key] = project.metadata[key]
            self._restore_project(project, restored)
            raise
        if self.retriever is not None:
            self._progress(f"第 {chapter.number} 章：写入本地向量数据库")
            try:
                await self.retriever.index_chapter(chapter, memory_update)
            except Exception as exc:
                # The project checkpoint is authoritative; the vector index
                # is a rebuildable cache and must not roll back finished prose.
                self._progress(f"RAG 索引失败，可稍后重建：{exc}")
        return chapter

    @staticmethod
    def _restore_project(target: NovelProject, source: NovelProject) -> None:
        target.brief = source.brief
        target.bible = source.bible
        target.chapters = source.chapters
        target.current_chapter = source.current_chapter
        target.status = source.status
        target.metadata = source.metadata
        target.version = source.version

    @staticmethod
    def _revision_candidate_error(
        original: str,
        candidate: str,
        project: NovelProject,
        plan: ChapterPlan,
    ) -> str | None:
        """Reject obviously unsafe whole-chapter rewrites before committing.

        This is a small local guard, not a semantic proof.  It catches common
        failure modes of a prose rewrite while leaving the continuity reviewer
        responsible for meaning and plot.
        """

        original = original.strip()
        candidate = candidate.strip()
        if not candidate:
            return "改写结果为空"
        if len(original) >= 200:
            ratio = len(candidate) / len(original)
            if ratio < 0.55 or ratio > 1.8:
                return f"改写长度比例异常：{ratio:.2f}"
        original_numbers = Counter(re.findall(r"\d+(?:\.\d+)?", original))
        candidate_numbers = Counter(re.findall(r"\d+(?:\.\d+)?", candidate))
        missing_numbers = [
            token for token, count in original_numbers.items() if candidate_numbers[token] < count
        ]
        if missing_numbers:
            return "改写结果删掉了原文中的数字信息：" + "、".join(missing_numbers[:5])
        refs = set(plan.characters)
        if plan.pov:
            refs.add(plan.pov)
        for character in project.bible.characters:
            if character.id in refs or character.name in refs:
                if character.name in original and character.name not in candidate:
                    return f"改写结果删掉了参与角色：{character.name}"
        if any(marker in candidate for marker in ("改写后正文：", "以下是改写", "根据审稿意见")):
            return "改写结果包含元话语"
        return None

    @staticmethod
    def _review_notes(review: Mapping[str, Any]) -> list[str]:
        notes: list[str] = []
        for issue in review.get("issues", []):
            if isinstance(issue, Mapping):
                notes.append(
                    "；".join(
                        part
                        for part in (
                            _text(issue.get("severity")),
                            _text(issue.get("problem")),
                            _text(issue.get("fix")),
                        )
                        if part
                    )
                )
            elif _text(issue).strip():
                notes.append(_text(issue))
        for check in review.get("style_checks", []):
            if not isinstance(check, Mapping) or str(check.get("result", "")).lower() != "fail":
                continue
            note = "；".join(
                part
                for part in (
                    "文风",
                    _text(check.get("evidence")),
                    _text(check.get("note")),
                )
                if part
            )
            if note:
                notes.append(note)
        return notes

    @staticmethod
    def _review_error(value: Any) -> str | None:
        if not isinstance(value, Mapping):
            return "审稿结果必须是 JSON 对象"
        if isinstance(value.get("review"), Mapping):
            value = value["review"]
        if not any(key in value for key in ("decision", "status", "pass", "approved")):
            return "审稿结果缺少 decision"
        decision = value.get("decision", value.get("status"))
        if decision is None and "pass" in value:
            decision = "pass" if value.get("pass") else "revise"
        if decision is None and "approved" in value:
            decision = "pass" if value.get("approved") else "revise"
        if decision is not None:
            normalized = str(decision).lower()
            if normalized not in {
                "pass", "revise", "approved", "ok", "accept", "通过", "需要修改",
                "needs_revision", "needs-revision", "fail",
            }:
                return "审稿 decision 必须是 pass 或 revise"
        issues = value.get("issues", [])
        if not isinstance(issues, list):
            return "审稿结果的 issues 必须是数组"
        return None

    @staticmethod
    def _scene_plan_error(value: Any, base: ChapterPlan, project: NovelProject) -> str | None:
        if not isinstance(value, Mapping):
            return "章节场景计划必须是 JSON 对象"
        source = value.get("chapter_plan") if isinstance(value.get("chapter_plan"), Mapping) else value
        beats = source.get("beats")
        if not isinstance(beats, list) or not beats or any(not _text(beat).strip() for beat in beats):
            return "章节场景计划至少需要一个 beat"
        number = source.get("number", source.get("chapter_number", base.number))
        if number != base.number:
            return f"章节场景计划编号必须是 {base.number}"
        known_characters = {item.id for item in project.bible.characters}
        known_character_names = {item.name for item in project.bible.characters}
        known_hooks = {item.id for item in project.bible.foreshadows}
        for character_id in _strings(source.get("characters")):
            if character_id not in known_characters and character_id not in known_character_names:
                return f"章节场景计划引用了未知角色：{character_id}"
        for hook_id in _strings(source.get("foreshadow_ids")):
            if hook_id not in known_hooks:
                return f"章节场景计划引用了未知伏笔：{hook_id}"
        return None

    @staticmethod
    def _memory_error(value: Any, project: NovelProject) -> str | None:
        if not isinstance(value, Mapping):
            return "章节记忆必须是 JSON 对象"
        source = value.get("memory") if isinstance(value.get("memory"), Mapping) else value
        summary = source.get("summary") or source.get("chapter_summary")
        if not isinstance(summary, str) or not summary.strip():
            return "章节记忆缺少 summary"
        character_updates = source.get("character_updates", {})
        if not isinstance(character_updates, (Mapping, list)):
            return "character_updates 必须是对象或数组"
        known_characters = {item.id for item in project.bible.characters}
        if isinstance(character_updates, Mapping):
            unknown = set(str(key) for key in character_updates) - known_characters
            if any(not isinstance(item, Mapping) for item in character_updates.values()):
                return "character_updates 的每个角色值必须是对象"
        else:
            unknown = {
                str(item.get("character_id") or item.get("id"))
                for item in character_updates
                if isinstance(item, Mapping) and (item.get("character_id") or item.get("id"))
            } - known_characters
        if unknown:
            return "character_updates 引用了未知角色：" + "、".join(sorted(unknown))
        raw_hooks = source.get("foreshadow_updates", source.get("foreshadowing_updates", {}))
        if isinstance(raw_hooks, list):
            raw_hooks = {
                str(item.get("clue_id") or item.get("id")): item.get("status")
                for item in raw_hooks
                if isinstance(item, Mapping) and (item.get("clue_id") or item.get("id"))
            }
        if not isinstance(raw_hooks, Mapping):
            return "foreshadow_updates 必须是对象或数组"
        known_hooks = {item.id for item in project.bible.foreshadows}
        unknown_hooks = set(str(key) for key in raw_hooks) - known_hooks
        if unknown_hooks:
            return "foreshadow_updates 引用了未知伏笔：" + "、".join(sorted(unknown_hooks))
        aliases = {
            "planned", "planted", "developing", "resolved", "abandoned",
            "seeded", "developed", "paid_off", "paid-off", "paid off", "open",
        }
        invalid = [str(status) for status in raw_hooks.values() if str(status).lower() not in aliases]
        if invalid:
            return "foreshadow_updates 包含未知状态：" + "、".join(sorted(set(invalid)))
        return None

    @staticmethod
    def _apply_memory(project: NovelProject, update: Mapping[str, Any], chapter_number: int) -> None:
        previous = project.metadata.get("memory", {})
        memory = dict(previous) if isinstance(previous, Mapping) else {}
        summary = _text(update.get("summary") or memory.get("summary"))

        chapter_summaries = list(memory.get("chapter_summaries", []))
        chapter_summaries.append({"number": chapter_number, "summary": summary})
        memory["chapter_summaries"] = chapter_summaries[-12:]

        def extend_bounded(key: str, incoming: Any, limit: int) -> None:
            existing = list(memory.get(key, []))
            values = incoming if isinstance(incoming, list) else []
            for value in values:
                enriched = dict(value) if isinstance(value, Mapping) else value
                if isinstance(enriched, dict):
                    enriched.setdefault("chapter", chapter_number)
                existing.append(enriched)
            deduped: list[Any] = []
            seen: set[str] = set()
            for value in existing:
                marker = _text(value)
                if marker in seen:
                    continue
                seen.add(marker)
                deduped.append(value)
            memory[key] = deduped[-limit:]

        extend_bounded("timeline_events", update.get("timeline_events"), 120)
        extend_bounded("relationship_updates", update.get("relationship_updates"), 100)
        extend_bounded("facts", update.get("new_facts") or update.get("facts"), 200)
        extend_bounded("continuity_warnings", update.get("continuity_warnings"), 50)

        merged_characters = dict(memory.get("character_updates", {}))
        incoming_characters = update.get("character_updates", {})
        if isinstance(incoming_characters, Mapping):
            for character_id, changes in incoming_characters.items():
                prior_state = merged_characters.get(character_id, {})
                combined = dict(prior_state) if isinstance(prior_state, Mapping) else {}
                if isinstance(changes, Mapping):
                    for key, value in changes.items():
                        if key == "knows" and isinstance(value, list):
                            combined[key] = list(
                                dict.fromkeys(_strings(combined.get(key)) + _strings(value))
                            )
                        else:
                            combined[key] = value
                merged_characters[_text(character_id)] = combined

        merged_hooks = dict(memory.get("foreshadow_updates", {}))
        for hook in project.bible.foreshadows:
            merged_hooks.setdefault(hook.id, hook.status)
        incoming_hooks = update.get("foreshadow_updates", {})
        if isinstance(incoming_hooks, Mapping):
            for key, value in incoming_hooks.items():
                hook_id = _text(key)
                merged_hooks[hook_id] = _advance_foreshadow_status(
                    _status(merged_hooks.get(hook_id)),
                    _status(value),
                )

        memory["summary"] = summary
        memory["character_updates"] = merged_characters
        memory["foreshadow_updates"] = merged_hooks
        if "unresolved_threads" in update:
            memory["unresolved_threads"] = _strings(update.get("unresolved_threads"))
        if "next_chapter_context" in update:
            memory["next_chapter_context"] = _text(update.get("next_chapter_context"))
        project.metadata["memory"] = memory

        character_updates = update.get("character_updates", {})
        if isinstance(character_updates, Mapping):
            for character in project.bible.characters:
                changes = character_updates.get(character.id)
                if isinstance(changes, Mapping):
                    for key, value in changes.items():
                        if key == "knows" and isinstance(value, list):
                            character.state[key] = list(
                                dict.fromkeys(_strings(character.state.get(key)) + _strings(value))
                            )
                        else:
                            character.state[key] = value

        hook_updates = update.get("foreshadow_updates", {})
        if isinstance(hook_updates, Mapping):
            for hook in project.bible.foreshadows:
                new_status = hook_updates.get(hook.id)
                if not new_status:
                    continue
                prior_status = hook.status
                incoming_status = _status(new_status)
                effective_status = _advance_foreshadow_status(prior_status, incoming_status)
                hook.status = effective_status
                if hook.status in {"planted", "developing", "resolved"} and hook.plant_chapter is None:
                    hook.plant_chapter = chapter_number
                if hook.status == "resolved" and prior_status != "resolved":
                    if hook.plant_chapter is None:
                        hook.plant_chapter = chapter_number
                    hook.payoff_chapter = chapter_number

    async def write_chapters(self, project: NovelProject, count: int = 1) -> list[Chapter]:
        if type(count) is not int or count < 1:
            raise ValueError("count 必须是正整数")
        result: list[Chapter] = []
        for _ in range(count):
            chapter = await self.write_next_chapter(project)
            if chapter is None:
                break
            result.append(chapter)
        return result


__all__ = ["NovelAgent", "NovelGenerationError"]
