from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping


class ModelValidationError(ValueError):
    """Raised when persisted novel data violates a domain invariant."""


def _mapping(value: Any, model_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelValidationError(f"{model_name} 必须由 JSON 对象构造")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ModelValidationError(f"{field_name} 必须是字符串数组")
    return list(value)


def _string_dict(value: Any, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
    ):
        raise ModelValidationError(f"{field_name} 必须是字符串到字符串的对象")
    return dict(value)


def _object_dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ModelValidationError(f"{field_name} 必须是 JSON 对象")
    return deepcopy(dict(value))


def _require_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError(f"{field_name} 必须是非空字符串")


def _require_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise ModelValidationError(f"{field_name} 必须是字符串")


def _require_positive_int(value: Any, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ModelValidationError(f"{field_name} 必须是正整数")


def _optional_positive_int(value: Any, field_name: str) -> None:
    if value is not None:
        _require_positive_int(value, field_name)


@dataclass(slots=True)
class NovelBrief:
    """The user's high-level request and generation constraints."""

    title: str
    premise: str
    genre: str = ""
    language: str = "zh-CN"
    target_chapters: int = 10
    target_words_per_chapter: int = 2500
    tone: str = ""
    style: str = ""
    themes: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def validate(self) -> None:
        _require_text(self.title, "NovelBrief.title")
        _require_text(self.premise, "NovelBrief.premise")
        for name, value in (
            ("genre", self.genre),
            ("language", self.language),
            ("tone", self.tone),
            ("style", self.style),
        ):
            _require_string(value, f"NovelBrief.{name}")
        _require_positive_int(self.target_chapters, "NovelBrief.target_chapters")
        _require_positive_int(
            self.target_words_per_chapter,
            "NovelBrief.target_words_per_chapter",
        )
        _string_list(self.themes, "NovelBrief.themes")
        _string_list(self.constraints, "NovelBrief.constraints")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "title": self.title,
            "premise": self.premise,
            "genre": self.genre,
            "language": self.language,
            "target_chapters": self.target_chapters,
            "target_words_per_chapter": self.target_words_per_chapter,
            "tone": self.tone,
            "style": self.style,
            "themes": list(self.themes),
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NovelBrief:
        value = _mapping(data, cls.__name__)
        brief = cls(
            title=value.get("title", ""),
            premise=value.get("premise", ""),
            genre=value.get("genre", ""),
            language=value.get("language", "zh-CN"),
            target_chapters=value.get("target_chapters", 10),
            target_words_per_chapter=value.get("target_words_per_chapter", 2500),
            tone=value.get("tone", ""),
            style=value.get("style", ""),
            themes=_string_list(value.get("themes"), "NovelBrief.themes"),
            constraints=_string_list(value.get("constraints"), "NovelBrief.constraints"),
        )
        brief.validate()
        return brief


@dataclass(slots=True)
class Character:
    """A character profile plus mutable state used between chapters."""

    id: str
    name: str
    role: str = ""
    description: str = ""
    age: str = ""
    appearance: str = ""
    personality: list[str] = field(default_factory=list)
    background: str = ""
    goals: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    arc: str = ""
    relationships: dict[str, str] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_text(self.id, "Character.id")
        _require_text(self.name, "Character.name")
        for name, value in (
            ("role", self.role),
            ("description", self.description),
            ("age", self.age),
            ("appearance", self.appearance),
            ("background", self.background),
            ("arc", self.arc),
        ):
            _require_string(value, f"Character.{name}")
        _string_list(self.personality, "Character.personality")
        _string_list(self.goals, "Character.goals")
        _string_list(self.conflicts, "Character.conflicts")
        _string_dict(self.relationships, "Character.relationships")
        _object_dict(self.state, "Character.state")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "age": self.age,
            "appearance": self.appearance,
            "personality": list(self.personality),
            "background": self.background,
            "goals": list(self.goals),
            "conflicts": list(self.conflicts),
            "arc": self.arc,
            "relationships": dict(self.relationships),
            "state": deepcopy(self.state),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Character:
        value = _mapping(data, cls.__name__)
        character = cls(
            id=value.get("id", ""),
            name=value.get("name", ""),
            role=value.get("role", ""),
            description=value.get("description", ""),
            age=value.get("age", ""),
            appearance=value.get("appearance", ""),
            personality=_string_list(value.get("personality"), "Character.personality"),
            background=value.get("background", ""),
            goals=_string_list(value.get("goals"), "Character.goals"),
            conflicts=_string_list(value.get("conflicts"), "Character.conflicts"),
            arc=value.get("arc", ""),
            relationships=_string_dict(value.get("relationships"), "Character.relationships"),
            state=_object_dict(value.get("state"), "Character.state"),
        )
        character.validate()
        return character


@dataclass(slots=True)
class ChapterPlan:
    """A single chapter's intended dramatic work."""

    STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"planned", "drafted", "reviewed", "revised", "final"}
    )

    number: int
    title: str
    summary: str
    purpose: str = ""
    pov: str = ""
    characters: list[str] = field(default_factory=list)
    beats: list[str] = field(default_factory=list)
    foreshadow_ids: list[str] = field(default_factory=list)
    target_words: int = 2500
    status: str = "planned"

    def validate(self) -> None:
        _require_positive_int(self.number, "ChapterPlan.number")
        _require_text(self.title, "ChapterPlan.title")
        _require_text(self.summary, "ChapterPlan.summary")
        _require_string(self.purpose, "ChapterPlan.purpose")
        _require_string(self.pov, "ChapterPlan.pov")
        _string_list(self.characters, "ChapterPlan.characters")
        _string_list(self.beats, "ChapterPlan.beats")
        _string_list(self.foreshadow_ids, "ChapterPlan.foreshadow_ids")
        _require_positive_int(self.target_words, "ChapterPlan.target_words")
        if self.status not in self.STATUSES:
            allowed = "、".join(sorted(self.STATUSES))
            raise ModelValidationError(f"ChapterPlan.status 必须是：{allowed}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "number": self.number,
            "title": self.title,
            "summary": self.summary,
            "purpose": self.purpose,
            "pov": self.pov,
            "characters": list(self.characters),
            "beats": list(self.beats),
            "foreshadow_ids": list(self.foreshadow_ids),
            "target_words": self.target_words,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChapterPlan:
        value = _mapping(data, cls.__name__)
        plan = cls(
            number=value.get("number", 0),
            title=value.get("title", ""),
            summary=value.get("summary", ""),
            purpose=value.get("purpose", ""),
            pov=value.get("pov", ""),
            characters=_string_list(value.get("characters"), "ChapterPlan.characters"),
            beats=_string_list(value.get("beats"), "ChapterPlan.beats"),
            foreshadow_ids=_string_list(
                value.get("foreshadow_ids"),
                "ChapterPlan.foreshadow_ids",
            ),
            target_words=value.get("target_words", 2500),
            status=value.get("status", "planned"),
        )
        plan.validate()
        return plan


@dataclass(slots=True)
class Foreshadow:
    """A tracked setup/payoff item in the novel's continuity ledger."""

    STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"planned", "planted", "developing", "resolved", "abandoned"}
    )

    id: str
    description: str
    status: str = "planned"
    plant_chapter: int | None = None
    payoff_chapter: int | None = None
    related_characters: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        _require_text(self.id, "Foreshadow.id")
        _require_text(self.description, "Foreshadow.description")
        if self.status not in self.STATUSES:
            allowed = "、".join(sorted(self.STATUSES))
            raise ModelValidationError(f"Foreshadow.status 必须是：{allowed}")
        _optional_positive_int(self.plant_chapter, "Foreshadow.plant_chapter")
        _optional_positive_int(self.payoff_chapter, "Foreshadow.payoff_chapter")
        if (
            self.plant_chapter is not None
            and self.payoff_chapter is not None
            and self.payoff_chapter < self.plant_chapter
        ):
            raise ModelValidationError("Foreshadow.payoff_chapter 不能早于 plant_chapter")
        if self.status in {"planted", "developing", "resolved"} and self.plant_chapter is None:
            raise ModelValidationError(f"状态为 {self.status} 时必须设置 plant_chapter")
        if self.status == "resolved" and self.payoff_chapter is None:
            raise ModelValidationError("状态为 resolved 时必须设置 payoff_chapter")
        _string_list(self.related_characters, "Foreshadow.related_characters")
        _string_list(self.notes, "Foreshadow.notes")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "plant_chapter": self.plant_chapter,
            "payoff_chapter": self.payoff_chapter,
            "related_characters": list(self.related_characters),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Foreshadow:
        value = _mapping(data, cls.__name__)
        foreshadow = cls(
            id=value.get("id", ""),
            description=value.get("description", ""),
            status=value.get("status", "planned"),
            plant_chapter=value.get("plant_chapter"),
            payoff_chapter=value.get("payoff_chapter"),
            related_characters=_string_list(
                value.get("related_characters"),
                "Foreshadow.related_characters",
            ),
            notes=_string_list(value.get("notes"), "Foreshadow.notes"),
        )
        foreshadow.validate()
        return foreshadow


@dataclass(slots=True)
class StoryBible:
    """Stable world, character, outline and continuity information."""

    world_setting: str = ""
    central_conflict: str = ""
    themes: list[str] = field(default_factory=list)
    tone: str = ""
    style_guide: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    characters: list[Character] = field(default_factory=list)
    outline: list[ChapterPlan] = field(default_factory=list)
    foreshadows: list[Foreshadow] = field(default_factory=list)
    # Optional marketing copy generated alongside the planning data.  Empty
    # values keep older project.json files fully backwards compatible.
    synopsis: str = ""
    short_synopsis: str = ""

    def validate(self) -> None:
        _require_text(self.world_setting, "StoryBible.world_setting")
        _require_text(self.central_conflict, "StoryBible.central_conflict")
        _require_string(self.tone, "StoryBible.tone")
        _string_list(self.themes, "StoryBible.themes")
        _string_list(self.style_guide, "StoryBible.style_guide")
        _string_list(self.rules, "StoryBible.rules")
        _require_string(self.synopsis, "StoryBible.synopsis")
        _require_string(self.short_synopsis, "StoryBible.short_synopsis")
        for character in self.characters:
            if not isinstance(character, Character):
                raise ModelValidationError("StoryBible.characters 必须包含 Character")
            character.validate()
        for plan in self.outline:
            if not isinstance(plan, ChapterPlan):
                raise ModelValidationError("StoryBible.outline 必须包含 ChapterPlan")
            plan.validate()
        for foreshadow in self.foreshadows:
            if not isinstance(foreshadow, Foreshadow):
                raise ModelValidationError("StoryBible.foreshadows 必须包含 Foreshadow")
            foreshadow.validate()

        self._ensure_unique((item.id for item in self.characters), "角色 ID")
        self._ensure_unique((item.number for item in self.outline), "章节计划编号")
        self._ensure_unique((item.id for item in self.foreshadows), "伏笔 ID")
        outline_numbers = [item.number for item in self.outline]
        if outline_numbers and outline_numbers != list(range(1, len(outline_numbers) + 1)):
            raise ModelValidationError("章节计划编号必须从 1 连续递增")
        character_ids = {item.id for item in self.characters}
        foreshadow_ids = {item.id for item in self.foreshadows}
        for plan in self.outline:
            unknown_characters = set(plan.characters) - character_ids
            if unknown_characters:
                raise ModelValidationError(
                    "章节计划引用了未知角色：" + "、".join(sorted(unknown_characters))
                )
            unknown_foreshadows = set(plan.foreshadow_ids) - foreshadow_ids
            if unknown_foreshadows:
                raise ModelValidationError(
                    "章节计划引用了未知伏笔：" + "、".join(sorted(unknown_foreshadows))
                )
        for foreshadow in self.foreshadows:
            unknown_characters = set(foreshadow.related_characters) - character_ids
            if unknown_characters:
                raise ModelValidationError(
                    "伏笔引用了未知角色：" + "、".join(sorted(unknown_characters))
                )

    @staticmethod
    def _ensure_unique(values, label: str) -> None:
        seen: set[Any] = set()
        duplicates: set[Any] = set()
        for value in values:
            if value in seen:
                duplicates.add(value)
            seen.add(value)
        if duplicates:
            readable = "、".join(str(item) for item in sorted(duplicates, key=str))
            raise ModelValidationError(f"{label} 不能重复：{readable}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = {
            "world_setting": self.world_setting,
            "central_conflict": self.central_conflict,
            "themes": list(self.themes),
            "tone": self.tone,
            "style_guide": list(self.style_guide),
            "rules": list(self.rules),
            "characters": [item.to_dict() for item in self.characters],
            "outline": [item.to_dict() for item in self.outline],
            "foreshadows": [item.to_dict() for item in self.foreshadows],
        }
        # Do not add empty optional fields when saving legacy projects.  New
        # projects normally contain both values and therefore persist them.
        if self.synopsis.strip():
            result["synopsis"] = self.synopsis
        if self.short_synopsis.strip():
            result["short_synopsis"] = self.short_synopsis
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StoryBible:
        value = _mapping(data, cls.__name__)
        raw_characters = value.get("characters", [])
        raw_outline = value.get("outline", [])
        raw_foreshadows = value.get("foreshadows", [])
        for name, items in (
            ("characters", raw_characters),
            ("outline", raw_outline),
            ("foreshadows", raw_foreshadows),
        ):
            if not isinstance(items, list):
                raise ModelValidationError(f"StoryBible.{name} 必须是数组")
        bible = cls(
            world_setting=value.get("world_setting", ""),
            central_conflict=value.get("central_conflict", ""),
            themes=_string_list(value.get("themes"), "StoryBible.themes"),
            tone=value.get("tone", ""),
            style_guide=_string_list(value.get("style_guide"), "StoryBible.style_guide"),
            rules=_string_list(value.get("rules"), "StoryBible.rules"),
            characters=[Character.from_dict(item) for item in raw_characters],
            outline=[ChapterPlan.from_dict(item) for item in raw_outline],
            foreshadows=[Foreshadow.from_dict(item) for item in raw_foreshadows],
            synopsis=value.get("synopsis") or "",
            short_synopsis=value.get("short_synopsis") or "",
        )
        bible.validate()
        return bible


@dataclass(slots=True)
class Chapter:
    """A persisted chapter draft and the state changes extracted from it."""

    STATUSES: ClassVar[frozenset[str]] = ChapterPlan.STATUSES

    number: int
    title: str
    content: str = ""
    summary: str = ""
    status: str = "planned"
    review_notes: list[str] = field(default_factory=list)
    character_updates: dict[str, Any] = field(default_factory=dict)
    foreshadow_updates: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_positive_int(self.number, "Chapter.number")
        _require_text(self.title, "Chapter.title")
        _require_string(self.content, "Chapter.content")
        _require_string(self.summary, "Chapter.summary")
        if self.status not in self.STATUSES:
            allowed = "、".join(sorted(self.STATUSES))
            raise ModelValidationError(f"Chapter.status 必须是：{allowed}")
        if self.status != "planned" and not self.content.strip():
            raise ModelValidationError(f"状态为 {self.status} 的章节必须包含正文")
        _string_list(self.review_notes, "Chapter.review_notes")
        _object_dict(self.character_updates, "Chapter.character_updates")
        updates = _string_dict(self.foreshadow_updates, "Chapter.foreshadow_updates")
        invalid = sorted(set(updates.values()) - Foreshadow.STATUSES)
        if invalid:
            raise ModelValidationError(
                "Chapter.foreshadow_updates 包含未知状态：" + "、".join(invalid)
            )
        _object_dict(self.metadata, "Chapter.metadata")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "status": self.status,
            "review_notes": list(self.review_notes),
            "character_updates": deepcopy(self.character_updates),
            "foreshadow_updates": dict(self.foreshadow_updates),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Chapter:
        value = _mapping(data, cls.__name__)
        chapter = cls(
            number=value.get("number", 0),
            title=value.get("title", ""),
            content=value.get("content", ""),
            summary=value.get("summary", ""),
            status=value.get("status", "planned"),
            review_notes=_string_list(value.get("review_notes"), "Chapter.review_notes"),
            character_updates=_object_dict(
                value.get("character_updates"),
                "Chapter.character_updates",
            ),
            foreshadow_updates=_string_dict(
                value.get("foreshadow_updates"),
                "Chapter.foreshadow_updates",
            ),
            metadata=_object_dict(value.get("metadata"), "Chapter.metadata"),
        )
        chapter.validate()
        return chapter


@dataclass(slots=True)
class NovelProject:
    """The complete resumable state of one novel generation project."""

    STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"planning", "writing", "paused", "completed", "failed"}
    )

    brief: NovelBrief
    bible: StoryBible = field(default_factory=StoryBible)
    chapters: list[Chapter] = field(default_factory=list)
    current_chapter: int = 0
    status: str = "planning"
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def validate(self) -> None:
        if not isinstance(self.brief, NovelBrief):
            raise ModelValidationError("NovelProject.brief 必须是 NovelBrief")
        if not isinstance(self.bible, StoryBible):
            raise ModelValidationError("NovelProject.bible 必须是 StoryBible")
        self.brief.validate()
        self.bible.validate()
        for chapter in self.chapters:
            if not isinstance(chapter, Chapter):
                raise ModelValidationError("NovelProject.chapters 必须包含 Chapter")
            chapter.validate()
        StoryBible._ensure_unique((item.number for item in self.chapters), "已生成章节编号")
        chapter_numbers = [item.number for item in self.chapters]
        if chapter_numbers != list(range(1, len(chapter_numbers) + 1)):
            raise ModelValidationError("已生成章节编号必须从 1 连续递增")
        outline_numbers = {item.number for item in self.bible.outline}
        unknown_chapters = set(chapter_numbers) - outline_numbers
        if unknown_chapters:
            raise ModelValidationError(
                "已生成章节不在章节计划中：" + "、".join(str(item) for item in sorted(unknown_chapters))
            )
        known_foreshadows = {item.id for item in self.bible.foreshadows}
        for chapter in self.chapters:
            unknown_hooks = set(chapter.foreshadow_updates) - known_foreshadows
            if unknown_hooks:
                raise ModelValidationError(
                    "章节伏笔更新引用了未知伏笔：" + "、".join(sorted(unknown_hooks))
                )
        if len(self.bible.outline) != self.brief.target_chapters:
            raise ModelValidationError(
                "章节计划数量必须等于 NovelBrief.target_chapters"
            )
        if type(self.current_chapter) is not int or self.current_chapter < 0:
            raise ModelValidationError("NovelProject.current_chapter 必须是非负整数")
        expected_current = chapter_numbers[-1] if chapter_numbers else 0
        if self.current_chapter != expected_current:
            raise ModelValidationError(
                "current_chapter 必须等于已生成章节中的最大编号（没有章节时为 0）"
            )
        if self.status not in self.STATUSES:
            allowed = "、".join(sorted(self.STATUSES))
            raise ModelValidationError(f"NovelProject.status 必须是：{allowed}")
        _object_dict(self.metadata, "NovelProject.metadata")
        _require_positive_int(self.version, "NovelProject.version")
        if self.status == "completed" and len(self.chapters) != len(self.bible.outline):
            raise ModelValidationError("未生成全部章节的项目不能标记为 completed")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        persisted_metadata = {
            key: value
            for key, value in self.metadata.items()
            # These are runtime bindings, not portable novel state.  Storage
            # rebinds them to the path used for loading.
            if key not in {"project_dir", "project_name"}
        }
        return {
            "version": self.version,
            "status": self.status,
            "current_chapter": self.current_chapter,
            "brief": self.brief.to_dict(),
            "bible": self.bible.to_dict(),
            "chapters": [chapter.to_dict() for chapter in self.chapters],
            "metadata": deepcopy(persisted_metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NovelProject:
        value = _mapping(data, cls.__name__)
        raw_brief = value.get("brief")
        if raw_brief is None:
            raise ModelValidationError("NovelProject.brief 不能为空")
        raw_chapters = value.get("chapters", [])
        if not isinstance(raw_chapters, list):
            raise ModelValidationError("NovelProject.chapters 必须是数组")
        raw_bible = value.get("bible", {})
        project = cls(
            brief=NovelBrief.from_dict(raw_brief),
            bible=StoryBible.from_dict(raw_bible),
            chapters=[Chapter.from_dict(item) for item in raw_chapters],
            current_chapter=value.get("current_chapter", 0),
            status=value.get("status", "planning"),
            metadata=_object_dict(value.get("metadata"), "NovelProject.metadata"),
            version=value.get("version", 1),
        )
        project.validate()
        return project

    def save(self, path: str | Path) -> Path:
        """Atomically save this project to a directory or explicit JSON file."""

        from .storage import save_project

        return save_project(self, path)

    @classmethod
    def load(cls, path: str | Path) -> NovelProject:
        """Load and validate a project from a directory or explicit JSON file."""

        from .storage import load_project

        return load_project(path)
