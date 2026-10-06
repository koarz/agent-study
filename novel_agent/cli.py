from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .llm import FatalLLMError, OpenAICompatibleBackend
from .models import ModelValidationError, NovelBrief, NovelProject
from .retrieval import RAGIndexError
from .storage import resolve_project_file


DEFAULT_RAG_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RAG_DEVICE = "auto"
DEFAULT_RAG_TOP_K = 6
DEFAULT_RAG_MAX_CHARS = 6000


@dataclass(frozen=True, slots=True)
class RAGSettings:
    enabled: bool = False
    model: str = DEFAULT_RAG_MODEL
    device: str = DEFAULT_RAG_DEVICE
    top_k: int = DEFAULT_RAG_TOP_K
    max_chars: int = DEFAULT_RAG_MAX_CHARS
    rebuild: bool = False
    offline: bool = False

    def as_metadata(self) -> dict[str, object]:
        """Return the portable, non-machine-specific project settings."""

        return {
            "enabled": self.enabled,
            "embedding_model": self.model,
            "top_k": self.top_k,
            "max_chars": self.max_chars,
        }


def _store_rag_settings(project: NovelProject, settings: RAGSettings) -> None:
    existing = project.metadata.get("rag", {})
    persisted = dict(existing) if isinstance(existing, dict) else {}
    persisted.update(settings.as_metadata())
    project.metadata["rag"] = persisted


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是整数") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def _non_negative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是数字") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("不能小于 0")
    return number


def _add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="覆盖 LLM_MODEL_ID")
    parser.add_argument("--base-url", default=None, help="覆盖 LLM_BASE_URL")
    parser.add_argument("--api-key", default=None, help="覆盖 LLM_API_KEY；建议使用环境变量")
    parser.add_argument("--timeout", type=_positive_int, default=None, help="覆盖 LLM_TIMEOUT，单位秒")
    parser.add_argument("--temperature", type=_non_negative_float, default=0.7, help="生成温度")


def _add_rag_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用或禁用 ChromaDB + 本地 BGE 历史章节检索",
    )
    parser.add_argument(
        "--rag-model",
        default=None,
        help=f"本地 BGE 模型，默认 {DEFAULT_RAG_MODEL}",
    )
    parser.add_argument(
        "--rag-device",
        default=None,
        help="embedding 设备，例如 auto、cpu、cuda、cuda:0 或 mps",
    )
    parser.add_argument(
        "--rag-top-k",
        type=_positive_int,
        default=None,
        help=f"每章最多检索片段数，默认 {DEFAULT_RAG_TOP_K}",
    )
    parser.add_argument(
        "--rag-max-chars",
        type=_positive_int,
        default=None,
        help=f"注入提示词的检索文本字符上限，默认 {DEFAULT_RAG_MAX_CHARS}",
    )
    parser.add_argument(
        "--rag-rebuild",
        action="store_true",
        default=None,
        help="本次运行强制重建当前项目的 Chroma 索引",
    )
    parser.add_argument(
        "--rag-offline",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="只从本地缓存加载 BGE 模型，不允许联网下载",
    )


def _add_quality_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--deai-review",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用去模板化文风审稿，并在审稿人判断安全时提出自然化改写（默认启用）",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m novel_agent",
        description="自动规划、续写并维护人物与伏笔状态的小说 Agent",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser(
        "new", help="创建小说项目、故事圣经、作品简介与全书大纲"
    )
    _add_llm_arguments(new_parser)
    _add_rag_arguments(new_parser)
    new_parser.add_argument("--title", required=True, help="小说标题")
    new_parser.add_argument("--premise", required=True, help="一句话或一段话的核心创意")
    new_parser.add_argument("--genre", default="", help="题材，例如悬疑、科幻或仙侠")
    new_parser.add_argument("--tone", default="", help="整体氛围，例如克制、黑暗或轻松")
    new_parser.add_argument("--style", default="", help="叙事视角与文风要求")
    new_parser.add_argument("--language", default="zh-CN", help="创作语言，默认 zh-CN")
    new_parser.add_argument(
        "--chapters",
        type=_positive_int,
        default=10,
        help="目标章节数，默认 10",
    )
    new_parser.add_argument(
        "--words-per-chapter",
        type=_positive_int,
        default=2500,
        help="每章目标字数，默认 2500",
    )
    new_parser.add_argument(
        "--theme",
        action="append",
        default=[],
        help="主题，可重复传入，也可用逗号分隔",
    )
    new_parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="额外约束，可重复传入，也可用逗号分隔",
    )
    new_parser.add_argument("--project-name", default=None, help="项目目录名；默认根据标题生成")
    new_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("projects"),
        help="项目输出根目录，默认 projects",
    )
    new_parser.set_defaults(handler=_handle_new)

    write_parser = subparsers.add_parser("write", help="从磁盘项目继续生成章节")
    _add_llm_arguments(write_parser)
    _add_rag_arguments(write_parser)
    _add_quality_arguments(write_parser)
    write_parser.add_argument("project", type=Path, help="项目目录或 project.json 路径")
    write_parser.add_argument("--count", type=_positive_int, default=1, help="本次生成章节数，默认 1")
    write_parser.set_defaults(handler=_handle_write)

    polish_parser = subparsers.add_parser(
        "polish", help="独立检查并润色已经生成的章节，不生成新章节"
    )
    _add_llm_arguments(polish_parser)
    _add_rag_arguments(polish_parser)
    _add_quality_arguments(polish_parser)
    polish_parser.add_argument("project", type=Path, help="项目目录或 project.json 路径")
    polish_parser.add_argument(
        "--chapter",
        type=_positive_int,
        required=True,
        help="从哪一章开始检查",
    )
    polish_parser.add_argument(
        "--count",
        type=_positive_int,
        default=1,
        help="连续处理章节数，默认 1",
    )
    polish_parser.add_argument(
        "--check-only",
        action="store_true",
        help="只输出审稿报告，不修改正文或项目文件",
    )
    polish_parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    polish_parser.set_defaults(handler=_handle_polish)

    synopsis_parser = subparsers.add_parser(
        "synopsis", help="生成或重写已有项目的作品简介"
    )
    _add_llm_arguments(synopsis_parser)
    synopsis_parser.add_argument("project", type=Path, help="项目目录或 project.json 路径")
    synopsis_parser.add_argument("--json", action="store_true", help="以 JSON 输出简介")
    synopsis_parser.set_defaults(handler=_handle_synopsis)

    status_parser = subparsers.add_parser("status", help="查看项目进度，不调用模型")
    status_parser.add_argument("project", type=Path, help="项目目录或 project.json 路径")
    status_parser.add_argument("--json", action="store_true", help="以 JSON 输出状态")
    status_parser.set_defaults(handler=_handle_status)
    return parser


def _split_repeated(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        for item in re.split(r"[,，]", value):
            cleaned = item.strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
    return result


def _project_name(title: str) -> str:
    cleaned = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", title.strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("_-")[:80]
    if cleaned:
        return cleaned
    return datetime.now().strftime("novel_%Y%m%d_%H%M%S")


def _validate_project_name(value: str) -> str:
    name = value.strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("project-name 必须是单个非空目录名，不能包含路径分隔符")
    return name


def _llm_settings(args: argparse.Namespace) -> tuple[str, str, str, int]:
    model = args.model or os.getenv("LLM_MODEL_ID")
    base_url = args.base_url or os.getenv("LLM_BASE_URL")
    api_key = args.api_key or os.getenv("LLM_API_KEY")
    raw_timeout = args.timeout or os.getenv("LLM_TIMEOUT", "120")
    try:
        timeout = int(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM_TIMEOUT 必须是正整数") from exc
    if timeout <= 0:
        raise ValueError("LLM_TIMEOUT 必须是正整数")

    missing = [
        name
        for name, value in (
            ("LLM_MODEL_ID", model),
            ("LLM_BASE_URL", base_url),
            ("LLM_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"缺少配置：{', '.join(missing)}。请写入 .env 或通过参数提供。")
    return str(api_key), str(base_url), str(model), timeout


def _parse_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled", ""}:
        return False
    raise ValueError(f"{label} 必须是 true/false、1/0、yes/no 或 on/off")


def _env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    return None if value is None else _parse_bool(value, name)


def _positive_setting(value: object, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是正整数") from exc
    if number <= 0:
        raise ValueError(f"{label} 必须是正整数")
    return number


def _first_defined(*values):
    return next((value for value in values if value is not None), None)


def _read_rag_status(project_dir: Path, project: NovelProject) -> dict[str, object]:
    """Read the lightweight manifest without importing Chroma or torch."""

    configured = project.metadata.get("rag", {})
    configured = dict(configured) if isinstance(configured, dict) else {}
    manifest_path = project_dir / "rag" / "manifest.json"
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                manifest = raw
        except (OSError, json.JSONDecodeError):
            manifest = {}
    chapters = manifest.get("chapters", {})
    indexed_chapters = len(chapters) if isinstance(chapters, dict) else 0
    indexed_documents = 0
    if isinstance(chapters, dict):
        indexed_documents = sum(
            int(entry.get("chunk_count", len(entry.get("ids", []))))
            for entry in chapters.values()
            if isinstance(entry, dict)
        )
    return {
        "enabled": bool(configured.get("enabled", False)),
        "model": manifest.get("embedding_model", configured.get("embedding_model")),
        "indexed_chapters": indexed_chapters,
        "indexed_documents": indexed_documents,
        "manifest": str(manifest_path) if manifest_path.is_file() else None,
    }


def _rag_settings(args: argparse.Namespace, project: NovelProject | None = None) -> RAGSettings:
    stored: dict[str, object] = {}
    if project is not None:
        raw = project.metadata.get("rag", {})
        if isinstance(raw, dict):
            stored = raw

    stored_enabled = stored.get("enabled")
    if stored_enabled is not None:
        stored_enabled = _parse_bool(stored_enabled, "metadata.rag.enabled")
    cli_enabled = getattr(args, "rag", None)
    if cli_enabled is not None:
        enabled = bool(cli_enabled)
    else:
        env_enabled = _env_bool("NOVEL_RAG_ENABLED")
        enabled = bool(_first_defined(env_enabled, stored_enabled, False))

    # An explicit --no-rag must remain lightweight even if stale optional RAG
    # values exist in the environment or project metadata.
    if not enabled:
        return RAGSettings(enabled=False)

    model = str(
        _first_defined(
            getattr(args, "rag_model", None),
            os.getenv("NOVEL_RAG_EMBEDDING_MODEL_ID"),
            stored.get("embedding_model"),
            stored.get("model"),
            DEFAULT_RAG_MODEL,
        )
    ).strip()
    if not model:
        raise ValueError("RAG embedding 模型不能为空")
    device = str(
        _first_defined(
            getattr(args, "rag_device", None),
            os.getenv("NOVEL_RAG_DEVICE"),
            DEFAULT_RAG_DEVICE,
        )
    ).strip()
    if not device:
        raise ValueError("RAG device 不能为空")
    top_k = _positive_setting(
        _first_defined(
            getattr(args, "rag_top_k", None),
            os.getenv("NOVEL_RAG_TOP_K"),
            stored.get("top_k"),
            DEFAULT_RAG_TOP_K,
        ),
        "NOVEL_RAG_TOP_K",
    )
    max_chars = _positive_setting(
        _first_defined(
            getattr(args, "rag_max_chars", None),
            os.getenv("NOVEL_RAG_MAX_CHARS"),
            stored.get("max_chars"),
            DEFAULT_RAG_MAX_CHARS,
        ),
        "NOVEL_RAG_MAX_CHARS",
    )
    cli_rebuild = getattr(args, "rag_rebuild", None)
    rebuild = bool(
        cli_rebuild
        if cli_rebuild is not None
        else _first_defined(_env_bool("NOVEL_RAG_REBUILD"), False)
    )
    cli_offline = getattr(args, "rag_offline", None)
    offline = bool(
        cli_offline
        if cli_offline is not None
        else _first_defined(_env_bool("NOVEL_RAG_OFFLINE"), False)
    )
    return RAGSettings(
        enabled=True,
        model=model,
        device=device,
        top_k=top_k,
        max_chars=max_chars,
        rebuild=rebuild,
        offline=offline,
    )


def _build_retriever(settings: RAGSettings):
    if not settings.enabled:
        return None
    try:
        from .retrieval import ChromaBGERetriever
        return ChromaBGERetriever(
            model_name=settings.model,
            device=settings.device,
            top_k=settings.top_k,
            max_chars=settings.max_chars,
            rebuild=settings.rebuild,
            offline=settings.offline,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "RAG 依赖"
        raise ValueError(
            f"已启用 RAG，但缺少 {missing}。"
            "请安装可选依赖：python -m pip install -r requirements-rag.txt"
        ) from exc


def _build_agent(
    args: argparse.Namespace,
    output_dir: Path,
    project: NovelProject | None = None,
    *,
    use_rag: bool = True,
    use_style_review: bool | None = None,
):
    # Import lazily so `status` and `--help` remain usable without loading the
    # generation engine or constructing a provider client.
    from .engine import NovelAgent

    api_key, base_url, model, timeout = _llm_settings(args)
    backend = OpenAICompatibleBackend(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
    )
    def report(message: str) -> None:
        print(f"[小说 Agent] {message}", flush=True)

    settings = _rag_settings(args, project) if use_rag else RAGSettings(enabled=False)
    retriever = _build_retriever(settings)
    if use_style_review is None:
        cli_deai = getattr(args, "deai_review", None)
        env_deai = _env_bool("NOVEL_DEAI_REVIEW")
        use_style_review = bool(_first_defined(cli_deai, env_deai, True))
    kwargs = {
        "output_dir": output_dir,
        "temperature": args.temperature,
        "progress_callback": report,
        "ai_style_review": use_style_review,
    }
    # Until RAG is enabled, do not require a newer engine constructor.  This
    # also keeps all legacy CLI invocations dependency-free and compatible.
    if retriever is not None:
        kwargs["retriever"] = retriever
    return NovelAgent(backend, **kwargs)


async def _handle_new(args: argparse.Namespace) -> int:
    project_name = _validate_project_name(args.project_name) if args.project_name else None
    brief = NovelBrief(
        title=args.title,
        premise=args.premise,
        genre=args.genre,
        language=args.language,
        target_chapters=args.chapters,
        target_words_per_chapter=args.words_per_chapter,
        tone=args.tone,
        style=args.style,
        themes=_split_repeated(args.theme),
        constraints=_split_repeated(args.constraint),
    )
    brief.validate()
    output_dir = args.output_dir.expanduser()
    agent = _build_agent(args, output_dir)
    project = await agent.create_project(brief, project_name)
    rag_settings = _rag_settings(args)
    if rag_settings.enabled:
        _store_rag_settings(project, rag_settings)
        project.save(project.metadata.get("project_dir", output_dir))
    fallback_name = project_name or _project_name(args.title)
    project_dir = Path(project.metadata.get("project_dir", output_dir / fallback_name))
    print(f"项目已创建：{project_dir}")
    print(f"标题：{project.brief.title}")
    print(
        f"规划：{len(project.bible.characters)} 名人物，"
        f"{len(project.bible.outline)} 章，{len(project.bible.foreshadows)} 条伏笔"
    )
    if project.bible.synopsis:
        print(f"作品简介：{project.bible.synopsis}")
        if project.bible.short_synopsis:
            print(f"一句话简介：{project.bible.short_synopsis}")
        print(f"简介文件：{project_dir / 'synopsis.md'}")
    print(f"开始写作：python -m novel_agent write {shlex.quote(str(project_dir))}")
    return 0


async def _handle_write(args: argparse.Namespace) -> int:
    project_file = resolve_project_file(args.project).expanduser().resolve()
    project_dir = project_file.parent
    project = NovelProject.load(project_file)
    project.metadata.setdefault("project_name", project_dir.name)
    project.metadata["project_dir"] = str(project_dir)
    rag_settings = _rag_settings(args, project)
    _store_rag_settings(project, rag_settings)
    agent = _build_agent(args, project_dir.parent, project)
    result = await agent.write_chapters(project, args.count)
    if isinstance(result, NovelProject):
        project = result
    print(f"写作完成：{project_dir}")
    print(
        f"当前进度：第 {project.current_chapter}/{project.brief.target_chapters} 章，"
        f"状态：{project.status}"
    )
    return 0


async def _handle_polish(args: argparse.Namespace) -> int:
    project_file = resolve_project_file(args.project).expanduser().resolve()
    project_dir = project_file.parent
    project = NovelProject.load(project_file)
    project.metadata.setdefault("project_name", project_dir.name)
    project.metadata["project_dir"] = str(project_dir)
    # A check-only run never touches the vector index.  Applying a polish uses
    # the project's normal RAG settings so changed chapter text is re-indexed.
    agent = _build_agent(
        args,
        project_dir.parent,
        project,
        use_rag=not args.check_only,
    )
    results = await agent.polish_chapters(
        project,
        args.chapter,
        args.count,
        apply=not args.check_only,
    )
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for item in results:
        chapter_number = item["chapter"]
        report = item.get("naturalness") or {}
        score = report.get("score") if isinstance(report, dict) else None
        if args.check_only:
            state = "检查完成，未修改"
        elif item.get("rejected"):
            state = f"修改已拒绝：{item['rejected']}"
        elif item.get("changed"):
            state = "已润色并保存"
        else:
            state = "无需修改"
        score_text = f"，自然度预检分数 {score}" if score is not None else ""
        print(f"第 {chapter_number} 章：{state}{score_text}")
        issues = report.get("issues", []) if isinstance(report, dict) else []
        for issue in issues:
            if isinstance(issue, dict):
                print(f"  - {issue.get('problem', '文风问题')}：{issue.get('fix', '')}")
    return 0


async def _handle_synopsis(args: argparse.Namespace) -> int:
    project_file = resolve_project_file(args.project).expanduser().resolve()
    project_dir = project_file.parent
    project = NovelProject.load(project_file)
    project.metadata.setdefault("project_name", project_dir.name)
    project.metadata["project_dir"] = str(project_dir)
    # Synopsis generation only needs the story bible.  Avoid initializing the
    # optional vector stack just because the project happens to use RAG.
    agent = _build_agent(
        args,
        project_dir.parent,
        project,
        use_rag=False,
        use_style_review=False,
    )
    result = await agent.generate_synopsis(project)
    payload = {
        "project_file": str(project_file),
        "synopsis": result["synopsis"],
        "short_synopsis": result["short_synopsis"],
        "file": str(project_dir / "synopsis.md"),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"作品简介已保存：{payload['file']}")
        print(f"长简介：{payload['synopsis']}")
        print(f"一句话简介：{payload['short_synopsis']}")
    return 0


async def _handle_status(args: argparse.Namespace) -> int:
    project_file = resolve_project_file(args.project).expanduser().resolve()
    project = NovelProject.load(project_file)
    counts = Counter(item.status for item in project.bible.foreshadows)
    next_chapter = min(
        (plan for plan in project.bible.outline if plan.number > project.current_chapter),
        key=lambda plan: plan.number,
        default=None,
    )
    payload = {
        "project_file": str(project_file),
        "title": project.brief.title,
        "status": project.status,
        "current_chapter": project.current_chapter,
        "target_chapters": project.brief.target_chapters,
        "saved_chapters": len(project.chapters),
        "characters": len(project.bible.characters),
        "foreshadows": dict(sorted(counts.items())),
        "synopsis": (
            {
                "long": project.bible.synopsis,
                "short": project.bible.short_synopsis,
                "file": str(project_file.parent / "synopsis.md"),
            }
            if project.bible.synopsis
            else None
        ),
        "next_chapter": (
            {"number": next_chapter.number, "title": next_chapter.title}
            if next_chapter is not None
            else None
        ),
        "rag": _read_rag_status(project_file.parent, project),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"项目：{payload['title']}")
    print(f"文件：{payload['project_file']}")
    print(f"状态：{payload['status']}")
    print(
        f"进度：第 {payload['current_chapter']}/{payload['target_chapters']} 章"
        f"（已保存 {payload['saved_chapters']} 章）"
    )
    print(f"人物：{payload['characters']} 名")
    foreshadow_text = "，".join(
        f"{status}={count}" for status, count in payload["foreshadows"].items()
    ) or "无"
    print(f"伏笔：{foreshadow_text}")
    if next_chapter is None:
        print("下一章：无")
    else:
        print(f"下一章：第 {next_chapter.number} 章《{next_chapter.title}》")
    rag = payload["rag"]
    if rag["enabled"]:
        print(
            f"RAG：已启用，已索引 {rag['indexed_chapters']} 章 / "
            f"{rag['indexed_documents']} 个片段"
        )
    else:
        print("RAG：未启用")
    if payload["synopsis"]:
        print(f"作品简介：已生成（{payload['synopsis']['file']}）")
    else:
        print("作品简介：未生成，可运行 `python -m novel_agent synopsis <项目>`")
    return 0


async def async_main(argv: Sequence[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        # Process environment variables remain fully supported.  This also
        # keeps offline `status` and help usable before dependencies are set up.
        pass
    else:
        load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    return await handler(args)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    except (FatalLLMError, RAGIndexError, OSError, ModelValidationError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
