"""Local vector retrieval for long-form novel continuity.

The optional stack is deliberately lazy: importing :mod:`novel_agent` or
running the non-RAG CLI commands does not import ChromaDB, PyTorch, or
Sentence Transformers.  When enabled, each project gets an independent
Chroma persistent collection and a small JSON manifest that makes indexes
rebuildable after a model/chunking change.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .models import Chapter, ChapterPlan, NovelProject
from .storage import atomic_write_json


MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
COLLECTION_NAME = "novel_memory"
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


async def _run_sync(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke local vector work behind an async boundary.

    The novel pipeline has a single writer and local model calls are already
    the dominant synchronous operation.  Calling directly avoids executor
    deadlocks seen in restricted Python sandboxes and keeps Chroma's embedded
    client lifecycle deterministic.  A future server deployment can replace
    this helper with a bounded worker pool without changing the retriever API.
    """

    return function(*args, **kwargs)


class RAGDependencyError(ValueError):
    """Raised when the optional local retrieval dependencies are unavailable."""


class RAGIndexError(RuntimeError):
    """Raised for an invalid or unrecoverable local vector index."""


class EmbeddingProvider(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class ContextRetriever(Protocol):
    async def preflight(self, project_dir: str | Path) -> None: ...

    async def prepare(
        self,
        project: NovelProject,
        project_dir: str | Path,
        rebuild: bool = False,
    ) -> None: ...

    async def retrieve(
        self,
        project: NovelProject,
        plan: ChapterPlan,
        memory: Mapping[str, Any],
    ) -> list[dict[str, Any]]: ...

    async def index_chapter(
        self,
        chapter: Chapter,
        memory_update: Mapping[str, Any] | None = None,
    ) -> None: ...

    def status(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """Portable retrieval settings.

    Device and cache paths are intentionally not persisted in the project
    manifest; users can choose them per machine through CLI/environment.
    """

    model_name: str = DEFAULT_MODEL_NAME
    device: str = "auto"
    top_k: int = 6
    max_chars: int = 6000
    chunk_chars: int = 1000
    chunk_overlap: int = 150
    batch_size: int = 16
    offline: bool = False

    def validate(self) -> None:
        if not self.model_name.strip():
            raise ValueError("RAG embedding 模型不能为空")
        if not self.device.strip():
            raise ValueError("RAG device 不能为空")
        for name in ("top_k", "max_chars", "chunk_chars", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"RAG {name} 必须是正整数")
        if type(self.chunk_overlap) is not int or self.chunk_overlap < 0:
            raise ValueError("RAG chunk_overlap 必须是非负整数")
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("RAG chunk_overlap 必须小于 chunk_chars")

    def as_metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            "embedding_model": self.model_name,
            "top_k": self.top_k,
            "max_chars": self.max_chars,
            "chunk_chars": self.chunk_chars,
            "chunk_overlap": self.chunk_overlap,
            "batch_size": self.batch_size,
        }

    def index_metadata(self) -> dict[str, Any]:
        """Only settings that change document text or vector dimensions."""

        self.validate()
        return {
            "embedding_model": self.model_name,
            "chunk_chars": self.chunk_chars,
            "chunk_overlap": self.chunk_overlap,
        }


class LocalBGEEmbedder:
    """Sentence-Transformers wrapper for a local open-source BGE model."""

    QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str = "auto",
        batch_size: int = 16,
        offline: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.offline = offline
        self._model: Any = None
        self._load_lock = asyncio.Lock()

    def _load_model_sync(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RAGDependencyError(
                "启用本地 RAG 需要 sentence-transformers。"
                "请运行：python -m pip install -r requirements-rag.txt"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.device and self.device != "auto":
            kwargs["device"] = self.device
        if self.offline:
            # Recent sentence-transformers forwards this to Hugging Face Hub.
            # Older releases may reject it; the fallback environment flag is
            # handled below.
            kwargs["local_files_only"] = True
        try:
            self._model = SentenceTransformer(self.model_name, **kwargs)
        except TypeError:
            if not self.offline:
                raise
            old_value = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                fallback_kwargs = {key: value for key, value in kwargs.items() if key != "local_files_only"}
                self._model = SentenceTransformer(self.model_name, **fallback_kwargs)
            finally:
                if old_value is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = old_value
        except Exception as exc:
            if self.offline:
                raise RAGDependencyError(
                    f"无法从本地缓存加载 embedding 模型 {self.model_name}。"
                    "请先下载模型，或去掉 --rag-offline 允许首次下载。"
                ) from exc
            raise RAGDependencyError(
                f"无法加载 embedding 模型 {self.model_name}：{exc}"
            ) from exc
        return self._model

    async def _model_ready(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                await _run_sync(self._load_model_sync)
        return self._model

    @staticmethod
    def _normalise_vectors(value: Any) -> list[list[float]]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            raise RAGIndexError("embedding 模型返回了无法识别的向量")
        vectors: list[list[float]] = []
        for row in value:
            if hasattr(row, "tolist"):
                row = row.tolist()
            if not isinstance(row, (list, tuple)):
                raise RAGIndexError("embedding 模型返回了非二维向量")
            vector = [float(item) for item in row]
            if not vector:
                raise RAGIndexError("embedding 向量维度不能为 0")
            vectors.append(vector)
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = [str(text) for text in texts]
        if not values:
            return []
        model = await self._model_ready()
        result = await _run_sync(
            model.encode,
            values,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = self._normalise_vectors(result)
        if len(vectors) != len(values):
            raise RAGIndexError("embedding 返回数量与文档数量不一致")
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([self.QUERY_PREFIX + str(text)])
        return vectors[0]


def chunk_text(text: str, chunk_chars: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into deterministic overlapping chunks without dropping text."""

    if not text or chunk_chars <= 0:
        return []
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("overlap 必须是非负数且小于 chunk_chars")
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(start + chunk_chars, length)
        if end < length:
            boundary_start = start + max(1, int(chunk_chars * 0.65))
            candidates = [
                cleaned.rfind(separator, boundary_start, end)
                for separator in ("\n", "。", "！", "？", "；")
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks


def _hash_payload(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ChromaBGERetriever:
    """Persistent per-project Chroma index backed by a local BGE embedder."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str = "auto",
        top_k: int = 6,
        max_chars: int = 6000,
        chunk_chars: int = 1000,
        chunk_overlap: int = 150,
        batch_size: int = 16,
        rebuild: bool = False,
        offline: bool = False,
        embedder: EmbeddingProvider | None = None,
        client_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self.config = RAGConfig(
            model_name=model_name,
            device=device,
            top_k=top_k,
            max_chars=max_chars,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            offline=offline,
        )
        self.config.validate()
        self._requested_rebuild = rebuild
        self._embedder = embedder
        self._client_factory = client_factory
        self._client: Any = None
        self._collection: Any = None
        self._project_dir: Path | None = None
        self._project_id: str | None = None
        self._manifest: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    @property
    def rag_dir(self) -> Path | None:
        return self._project_dir / "rag" if self._project_dir else None

    @property
    def manifest_path(self) -> Path | None:
        return self.rag_dir / MANIFEST_FILENAME if self.rag_dir else None

    def _default_client(self, path: Path) -> Any:
        try:
            import chromadb
        except Exception as exc:
            raise RAGDependencyError(
                "启用本地 RAG 需要 chromadb。"
                "请运行：python -m pip install -r requirements-rag.txt"
            ) from exc
        try:
            from chromadb.config import Settings

            return chromadb.PersistentClient(
                path=str(path),
                settings=Settings(anonymized_telemetry=False),
            )
        except (TypeError, AttributeError):
            return chromadb.PersistentClient(path=str(path))

    def _new_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "collection": COLLECTION_NAME,
            "project_id": self._project_id,
            **self.config.as_metadata(),
            "chapters": {},
        }

    def _read_manifest(self) -> dict[str, Any] | None:
        path = self.manifest_path
        if path is None or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _write_manifest(self) -> None:
        path = self.manifest_path
        if path is None:
            raise RAGIndexError("RAG 项目路径尚未初始化")
        atomic_write_json(path, self._manifest)

    def _manifest_compatible(self, manifest: Mapping[str, Any] | None) -> bool:
        if not manifest or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            return False
        expected = self.config.index_metadata()
        return all(manifest.get(key) == value for key, value in expected.items()) and manifest.get(
            "project_id"
        ) == self._project_id

    def _collection_count(self) -> int:
        if self._collection is None:
            return 0
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise RAGIndexError("无法读取 Chroma 索引数量，请使用 --rag-rebuild") from exc

    def _reset_collection_sync(self) -> None:
        if self._client is None:
            raise RAGIndexError("Chroma 客户端尚未初始化")
        delete_error: Exception | None = None
        try:
            self._client.delete_collection(name=COLLECTION_NAME)
        except Exception as exc:
            delete_error = exc
            try:
                self._client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        if self._collection_count() != 0:
            error = RAGIndexError(
                "无法清空旧的 Chroma 索引，请删除项目 rag/ 目录后重试"
            )
            if delete_error is not None:
                raise error from delete_error
            raise error

    def _ensure_collection_sync(self) -> None:
        if self._project_dir is None:
            raise RAGIndexError("RAG 项目路径尚未初始化")
        chroma_path = self._project_dir / "rag" / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        if self._client is None:
            self._client = (
                self._client_factory(chroma_path)
                if self._client_factory is not None
                else self._default_client(chroma_path)
            )
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

    def _set_project(self, project: NovelProject, project_dir: str | Path) -> None:
        resolved = Path(project_dir).expanduser().resolve()
        if self._project_dir != resolved:
            self._project_dir = resolved
            self._client = None
            self._collection = None
            self._manifest = {}
        rag_metadata = project.metadata.get("rag", {})
        if not isinstance(rag_metadata, dict):
            rag_metadata = {}
        self._project_id = str(
            rag_metadata.get("project_id")
            or _hash_payload(
                {
                    "title": project.brief.title,
                    "created_at": project.metadata.get("created_at", ""),
                    "project_name": resolved.name,
                }
            )[:20]
        )
        rag_metadata.update(
            {
                "enabled": True,
                "project_id": self._project_id,
                "embedding_model": self.config.model_name,
                "top_k": self.config.top_k,
                "max_chars": self.config.max_chars,
                "chunk_chars": self.config.chunk_chars,
                "chunk_overlap": self.config.chunk_overlap,
            }
        )
        project.metadata["rag"] = rag_metadata

    async def prepare(
        self,
        project: NovelProject,
        project_dir: str | Path,
        rebuild: bool = False,
    ) -> None:
        async with self._lock:
            self._set_project(project, project_dir)
            await _run_sync(self._ensure_collection_sync)
            await self._ensure_embedder_ready()
            force_rebuild = bool(rebuild or self._requested_rebuild)
            self._requested_rebuild = False
            manifest = self._read_manifest()
            if not self._manifest_compatible(manifest) or force_rebuild:
                await _run_sync(self._reset_collection_sync)
                self._manifest = self._new_manifest()
            else:
                self._manifest = dict(manifest or self._new_manifest())
            # Retrieval-only knobs can change without invalidating vectors.
            # Keep the manifest current for status/debugging, while the
            # compatibility check above only keys on index-affecting values.
            self._manifest.update(self.config.as_metadata())

            expected_ids = [
                doc_id
                for entry in self._manifest.get("chapters", {}).values()
                for doc_id in entry.get("ids", [])
            ]
            if expected_ids:
                missing_ids = False
                try:
                    existing = await _run_sync(self._collection.get, ids=expected_ids, include=["metadatas"])
                    found_ids = set(existing.get("ids", [])) if isinstance(existing, Mapping) else set()
                    missing_ids = not set(expected_ids).issubset(found_ids)
                except (AttributeError, TypeError):
                    missing_ids = self._collection_count() < len(expected_ids)
                except Exception as exc:
                    raise RAGIndexError(
                        "无法验证 Chroma 索引完整性，请使用 --rag-rebuild 重建"
                    ) from exc
                if missing_ids:
                    await _run_sync(self._reset_collection_sync)
                    self._manifest = self._new_manifest()

            for chapter in project.chapters:
                await self._index_chapter_unlocked(chapter, None, save_manifest=False)
            self._write_manifest()

    async def preflight(self, project_dir: str | Path) -> None:
        """Validate Chroma and load the local model before paid LLM calls."""

        async with self._lock:
            self._project_dir = Path(project_dir).expanduser().resolve()
            await _run_sync(self._ensure_collection_sync)
            await self._ensure_embedder_ready()

    def _chapter_payload(
        self,
        chapter: Chapter,
        memory_update: Mapping[str, Any] | None,
    ) -> tuple[str, list[str], list[str], list[dict[str, Any]]]:
        memory_update = memory_update or {}
        content_hash = _hash_payload(
            {
                "chapter": chapter.to_dict(),
                "memory": memory_update,
            }
        )
        prefix = f"ch{chapter.number:04d}"
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        def add(kind: str, index: int, text: str) -> None:
            text = str(text).strip()
            if not text:
                return
            doc_id = f"{prefix}:{kind}:{index:04d}:{content_hash[:12]}"
            ids.append(doc_id)
            documents.append(text)
            metadatas.append(
                {
                    "project_id": str(self._project_id),
                    "chapter": chapter.number,
                    "chapter_title": chapter.title,
                    "kind": kind,
                    "chunk_index": index,
                    "content_hash": content_hash,
                }
            )

        add("summary", 0, f"第{chapter.number}章《{chapter.title}》摘要：{chapter.summary}")
        for index, chunk in enumerate(
            chunk_text(chapter.content, self.config.chunk_chars, self.config.chunk_overlap),
            start=0,
        ):
            add("chapter_chunk", index, f"第{chapter.number}章《{chapter.title}》正文片段：{chunk}")
        facts = memory_update.get("new_facts") if isinstance(memory_update, Mapping) else None
        if isinstance(facts, list):
            for index, fact in enumerate(facts):
                add("fact", index, f"第{chapter.number}章新增事实：{json.dumps(fact, ensure_ascii=False)}")
        return content_hash, ids, documents, metadatas

    async def _index_chapter_unlocked(
        self,
        chapter: Chapter,
        memory_update: Mapping[str, Any] | None,
        *,
        save_manifest: bool,
    ) -> None:
        if self._collection is None:
            raise RAGIndexError("RAG collection 尚未初始化")
        if memory_update is None:
            stored_memory = chapter.metadata.get("memory_update") if isinstance(chapter.metadata, Mapping) else None
            if isinstance(stored_memory, Mapping):
                memory_update = stored_memory
        content_hash, ids, documents, metadatas = self._chapter_payload(chapter, memory_update)
        chapters = self._manifest.setdefault("chapters", {})
        old_entry = chapters.get(str(chapter.number), {})
        if old_entry.get("content_hash") == content_hash and old_entry.get("ids"):
            return
        embeddings = await self._embed_documents(documents)
        if len(embeddings) != len(documents):
            raise RAGIndexError("embedding 数量与文档数量不一致")
        old_ids = list(old_entry.get("ids", []))
        # Upsert first; only delete old versions after embeddings are ready.
        await _run_sync(
            self._collection.upsert,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        stale_ids = [item for item in old_ids if item not in ids]
        if stale_ids:
            await _run_sync(self._collection.delete, ids=stale_ids)
        chapters[str(chapter.number)] = {
            "content_hash": content_hash,
            "ids": ids,
            "chunk_count": len(ids),
        }
        if save_manifest:
            self._write_manifest()

    async def _embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self._embedder is None:
            self._make_default_embedder()
        result = self._embedder.embed_documents(texts)
        if inspect.isawaitable(result):
            result = await result
        return [[float(item) for item in vector] for vector in result]

    async def _embed_query(self, text: str) -> list[float]:
        if self._embedder is None:
            self._make_default_embedder()
        result = self._embedder.embed_query(text)
        if inspect.isawaitable(result):
            result = await result
        return [float(item) for item in result]

    def _make_default_embedder(self) -> None:
        self._embedder = LocalBGEEmbedder(
            self.config.model_name,
            device=self.config.device,
            batch_size=self.config.batch_size,
            offline=self.config.offline,
        )

    async def _ensure_embedder_ready(self) -> None:
        if self._embedder is None:
            self._make_default_embedder()
        if isinstance(self._embedder, LocalBGEEmbedder):
            await self._embedder._model_ready()

    async def index_chapter(
        self,
        chapter: Chapter,
        memory_update: Mapping[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            if self._collection is None:
                raise RAGIndexError("请先调用 prepare 初始化 RAG 项目")
            await self._index_chapter_unlocked(chapter, memory_update, save_manifest=True)

    @staticmethod
    def _query_text(project: NovelProject, plan: ChapterPlan, memory: Mapping[str, Any]) -> str:
        character_ids = set(plan.characters)
        characters = [
            character
            for character in project.bible.characters
            if character.id in character_ids or character.name in character_ids or character.id == plan.pov
        ]
        hooks = [hook for hook in project.bible.foreshadows if hook.id in set(plan.foreshadow_ids)]
        parts = [plan.title, plan.summary, plan.purpose, *plan.beats]
        parts.extend(character.name + " " + character.description for character in characters)
        parts.extend(hook.description for hook in hooks)
        parts.extend(str(item) for item in memory.get("unresolved_threads", []) if item)
        parts.append(str(memory.get("next_chapter_context", "")))
        return "\n".join(part for part in parts if part).strip()

    async def retrieve(
        self,
        project: NovelProject,
        plan: ChapterPlan,
        memory: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        async with self._lock:
            if self._collection is None:
                raise RAGIndexError("请先调用 prepare 初始化 RAG 项目")
            if self._collection_count() <= 0:
                return []
            query_text = self._query_text(project, plan, memory)
            if not query_text:
                return []
            query_embedding = await self._embed_query(query_text)
            n_results = min(max(self.config.top_k * 4, self.config.top_k), self._collection_count())
            query_where = {
                "$and": [
                    {"project_id": {"$eq": str(self._project_id)}},
                    {"chapter": {"$lt": plan.number}},
                ]
            }
            try:
                result = await _run_sync(
                    self._collection.query,
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=query_where,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                result = {"ids": [[]]}
            # Tiny fake/test collections and older Chroma releases may not
            # understand compound filters; retry with project isolation and
            # retain the post-query chapter guard below.
            if not (result.get("ids") or [[]])[0]:
                try:
                    result = await _run_sync(
                        self._collection.query,
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        where={"project_id": str(self._project_id)},
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception as exc:
                    raise RAGIndexError(
                        "Chroma 检索失败，请使用 --rag-rebuild 重建索引"
                    ) from exc
            ids_rows = (result.get("ids") or [[]])[0]
            docs_rows = (result.get("documents") or [[]])[0]
            metadata_rows = (result.get("metadatas") or [[]])[0]
            distance_rows = (result.get("distances") or [[]])[0]
            candidates: list[dict[str, Any]] = []
            for index, doc_id in enumerate(ids_rows):
                metadata = metadata_rows[index] if index < len(metadata_rows) else {}
                if not isinstance(metadata, Mapping):
                    metadata = {}
                chapter_number = metadata.get("chapter")
                try:
                    chapter_number = int(chapter_number)
                except (TypeError, ValueError):
                    chapter_number = 0
                # Never expose a future/current uncommitted chapter to the
                # chapter being generated.
                if chapter_number >= plan.number:
                    continue
                text = str(docs_rows[index]) if index < len(docs_rows) else ""
                distance = float(distance_rows[index]) if index < len(distance_rows) else 1.0
                score = max(-1.0, min(1.0, 1.0 - distance))
                candidates.append(
                    {
                        "id": str(doc_id),
                        "chapter": chapter_number,
                        "kind": str(metadata.get("kind", "memory")),
                        "title": str(metadata.get("chapter_title", "")),
                        "score": round(score, 6),
                        "text": text,
                    }
                )
            candidates.sort(key=lambda item: item["score"], reverse=True)
            selected: list[dict[str, Any]] = []
            used_chars = 0
            for item in candidates:
                text_length = len(item["text"])
                remaining = self.config.max_chars - used_chars
                if remaining <= 0:
                    break
                if text_length > remaining:
                    if selected:
                        continue
                    item = dict(item)
                    item["text"] = item["text"][:remaining]
                    text_length = len(item["text"])
                selected.append(item)
                used_chars += text_length
                if len(selected) >= self.config.top_k or used_chars >= self.config.max_chars:
                    break
            return selected

    def status(self) -> dict[str, Any]:
        manifest = self._read_manifest() or {}
        chapters = manifest.get("chapters", {})
        return {
            "enabled": True,
            "model": manifest.get("embedding_model", self.config.model_name),
            "collection": manifest.get("collection", COLLECTION_NAME),
            "indexed_chapters": len(chapters) if isinstance(chapters, Mapping) else 0,
            "indexed_documents": sum(
                int(entry.get("chunk_count", len(entry.get("ids", []))))
                for entry in chapters.values()
                if isinstance(entry, Mapping)
            )
            if isinstance(chapters, Mapping)
            else 0,
            "manifest": str(self.manifest_path) if self.manifest_path else None,
        }


__all__ = [
    "RAGConfig",
    "RAGDependencyError",
    "RAGIndexError",
    "EmbeddingProvider",
    "ContextRetriever",
    "LocalBGEEmbedder",
    "ChromaBGERetriever",
    "chunk_text",
]
