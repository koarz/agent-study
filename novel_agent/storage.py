from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .models import Chapter, NovelProject


PROJECT_FILENAME = "project.json"
SYNOPSIS_FILENAME = "synopsis.md"


def resolve_project_file(path: str | os.PathLike[str]) -> Path:
    """Resolve either a project directory or an explicit JSON filename."""

    target = Path(path).expanduser()
    if target.suffix.lower() == ".json":
        return target
    return target / PROJECT_FILENAME


def atomic_write_text(path: str | os.PathLike[str], content: str) -> Path:
    """Write UTF-8 text through a sibling temporary file and atomic replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
        return target
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def atomic_write_json(path: str | os.PathLike[str], value: Mapping[str, Any]) -> Path:
    """Serialize a JSON object and atomically replace the target file."""

    if not isinstance(value, Mapping):
        raise TypeError("atomic_write_json 只接受 JSON 对象")
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return atomic_write_text(path, content)


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"项目文件不存在：{target}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"项目文件不是有效 JSON：{target}（{exc}）") from exc
    if not isinstance(value, dict):
        raise ValueError(f"项目文件根节点必须是 JSON 对象：{target}")
    return value


def save_project(project: NovelProject, path: str | os.PathLike[str]) -> Path:
    if not isinstance(project, NovelProject):
        raise TypeError("project 必须是 NovelProject")
    project.validate()
    target = resolve_project_file(path)
    saved = atomic_write_json(target, project.to_dict())
    # Keep the in-memory object bound to the location just written while
    # leaving these runtime-only fields out of portable JSON serialization.
    project.metadata["project_dir"] = str(target.parent.expanduser().resolve())
    project.metadata["project_name"] = target.parent.name
    return saved


def load_project(path: str | os.PathLike[str]) -> NovelProject:
    target = resolve_project_file(path)
    project = NovelProject.from_dict(read_json(target))
    # The filesystem location supplied by the caller is authoritative.  Do
    # not let a copied project retain an absolute path from its old machine or
    # directory when a later ``write`` resumes it.
    project.metadata["project_dir"] = str(target.parent.expanduser().resolve())
    project.metadata.setdefault("project_name", target.parent.name)
    return project


class ProjectStorage:
    """Convenience facade for project state and per-chapter Markdown files."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).expanduser()

    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_FILENAME

    @property
    def synopsis_file(self) -> Path:
        return self.root / SYNOPSIS_FILENAME

    @property
    def chapters_dir(self) -> Path:
        return self.root / "chapters"

    def save(self, project: NovelProject) -> Path:
        return save_project(project, self.project_file)

    def load(self) -> NovelProject:
        return load_project(self.project_file)

    def chapter_file(self, number: int) -> Path:
        if type(number) is not int or number <= 0:
            raise ValueError("章节编号必须是正整数")
        return self.chapters_dir / f"{number:03d}.md"

    def save_chapter(self, chapter: Chapter) -> Path:
        if not isinstance(chapter, Chapter):
            raise TypeError("chapter 必须是 Chapter")
        chapter.validate()
        heading = f"# 第 {chapter.number} 章 {chapter.title}\n\n"
        content = heading + chapter.content.strip() + "\n"
        return atomic_write_text(self.chapter_file(chapter.number), content)

    def save_synopsis(self, project: NovelProject) -> Path:
        """Write copy-ready platform descriptions beside project.json."""

        if not isinstance(project, NovelProject):
            raise TypeError("project 必须是 NovelProject")
        project.validate()
        long_text = project.bible.synopsis.strip()
        short_text = project.bible.short_synopsis.strip()
        if not long_text:
            raise ValueError("项目尚未生成作品简介")
        sections = [
            f"# {project.brief.title}",
            "",
            "## 作品简介",
            "",
            long_text,
        ]
        if short_text:
            sections.extend(["", "## 一句话简介", "", short_text])
        return atomic_write_text(self.synopsis_file, "\n".join(sections) + "\n")
