from __future__ import annotations

import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from novel_agent.models import Chapter
from novel_agent.retrieval import ChromaBGERetriever, chunk_text
from tests.test_novel_agent import sample_project


class FakeEmbedder:
    """Deterministic semantic vectors; no model download or torch import."""

    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if "钥匙" in text or "开门" in text:
            return [1.0, 0.0, 0.0]
        if "钟楼" in text or "楼梯" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    async def embed_documents(self, texts):
        self.document_calls += len(texts)
        return [self._vector(str(text)) for text in texts]

    async def embed_query(self, text):
        self.query_calls += 1
        return self._vector(str(text))


def _cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(a * a for a in right))
    return 1.0 - dot / (norm_left * norm_right)


class FakeCollection:
    def __init__(self):
        self.rows: dict[str, dict[str, Any]] = {}

    def count(self):
        return len(self.rows)

    def upsert(self, *, ids, documents, embeddings, metadatas):
        for row_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            self.rows[row_id] = {
                "document": document,
                "embedding": list(embedding),
                "metadata": dict(metadata),
            }

    def delete(self, *, ids):
        for row_id in ids:
            self.rows.pop(row_id, None)

    def query(self, *, query_embeddings, n_results, where, include):
        query = query_embeddings[0]
        rows = [
            (row_id, row)
            for row_id, row in self.rows.items()
            if row["metadata"].get("project_id") == where.get("project_id")
        ]
        rows.sort(key=lambda pair: _cosine_distance(query, pair[1]["embedding"]))
        rows = rows[:n_results]
        return {
            "ids": [[row_id for row_id, _ in rows]],
            "documents": [[row["document"] for _, row in rows]],
            "metadatas": [[row["metadata"] for _, row in rows]],
            "distances": [[_cosine_distance(query, row["embedding"]) for _, row in rows]],
        }


class FakeClient:
    def __init__(self):
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, *, name, metadata=None):
        return self.collections.setdefault(name, FakeCollection())

    def delete_collection(self, name):
        self.collections.pop(name, None)


class RagRetrievalTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clients: dict[str, FakeClient] = {}
        async def direct(function, *args, **kwargs):
            return function(*args, **kwargs)
        self._sync_patch = patch("novel_agent.retrieval._run_sync", direct)
        self._sync_patch.start()

    def tearDown(self):
        self._sync_patch.stop()

    def client_factory(self, path: Path):
        return self.clients.setdefault(str(path), FakeClient())

    async def test_chunking_is_deterministic_and_overlapping(self):
        text = "第一段。" * 20 + "\n第二段。" * 20
        first = chunk_text(text, chunk_chars=30, overlap=5)
        second = chunk_text(text, chunk_chars=30, overlap=5)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)
        self.assertTrue(all(item for item in first))
        self.assertTrue(any(set(first[0][-5:]) & set(first[1][:5]) for _ in [0]))

    async def test_prepare_index_query_and_manifest_survive_new_retriever(self):
        project = sample_project()
        chapter = Chapter(
            number=1,
            title="没有邮戳的信",
            content="林岚在旧邮局找到铜钥匙，钥匙指向钟楼。",
            summary="林岚获得铜钥匙。",
            status="final",
            metadata={
                "memory_update": {
                    "new_facts": [{"id": "fact_1", "fact": "钥匙来自信封"}]
                }
            },
        )
        project.chapters = [chapter]
        project.current_chapter = 1
        embedder = FakeEmbedder()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "project"
            retriever = ChromaBGERetriever(
                model_name="fake-bge",
                embedder=embedder,
                client_factory=self.client_factory,
                top_k=4,
                max_chars=120,
                chunk_chars=20,
                chunk_overlap=5,
            )
            await retriever.prepare(project, path)
            await retriever.index_chapter(chapter, chapter.metadata["memory_update"])
            before = embedder.document_calls
            hits = await retriever.retrieve(project, project.bible.outline[1], {})
            self.assertTrue(hits)
            self.assertLessEqual(sum(len(item["text"]) for item in hits), 120)
            self.assertEqual(hits[0]["chapter"], 1)
            self.assertTrue((path / "rag" / "manifest.json").is_file())

            reopened = ChromaBGERetriever(
                model_name="fake-bge",
                embedder=embedder,
                client_factory=self.client_factory,
                top_k=4,
                max_chars=120,
                chunk_chars=20,
                chunk_overlap=5,
            )
            await reopened.prepare(project, path)
            self.assertEqual(embedder.document_calls, before)
            self.assertTrue(await reopened.retrieve(project, project.bible.outline[1], {}))

    async def test_project_indexes_are_isolated_by_persistent_path(self):
        first_project = sample_project()
        second_project = sample_project()
        first_chapter = Chapter(
            number=1,
            title="甲",
            content="甲项目的秘密线索。",
            summary="甲项目摘要",
            status="final",
        )
        second_chapter = Chapter(
            number=1,
            title="乙",
            content="乙项目的秘密线索。",
            summary="乙项目摘要",
            status="final",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = ChromaBGERetriever(
                model_name="fake-bge", embedder=FakeEmbedder(), client_factory=self.client_factory
            )
            second = ChromaBGERetriever(
                model_name="fake-bge", embedder=FakeEmbedder(), client_factory=self.client_factory
            )
            await first.prepare(first_project, root / "a")
            await second.prepare(second_project, root / "b")
            await first.index_chapter(first_chapter)
            await second.index_chapter(second_chapter)
            first_project.chapters = [first_chapter]
            second_project.chapters = [second_chapter]
            first_hits = await first.retrieve(first_project, first_project.bible.outline[1], {})
            second_hits = await second.retrieve(second_project, second_project.bible.outline[1], {})
            self.assertTrue(all("乙项目" not in item["text"] for item in first_hits))
            self.assertTrue(all("甲项目" not in item["text"] for item in second_hits))

    async def test_real_chroma_persistent_client_when_optional_dependency_is_installed(self):
        try:
            import chromadb  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("optional chromadb dependency is not installed")
        project = sample_project()
        chapter = Chapter(
            number=1,
            title="真实 Chroma",
            content="铜钥匙打开钟楼暗门。",
            summary="钥匙与钟楼产生关联。",
            status="final",
        )
        project.chapters = [chapter]
        project.current_chapter = 1
        with tempfile.TemporaryDirectory() as temp_dir:
            first = ChromaBGERetriever(
                model_name="fake-bge-real-chroma",
                embedder=FakeEmbedder(),
                top_k=3,
                max_chars=200,
            )
            await first.prepare(project, Path(temp_dir))
            await first.index_chapter(chapter)
            second = ChromaBGERetriever(
                model_name="fake-bge-real-chroma",
                embedder=FakeEmbedder(),
                top_k=3,
                max_chars=200,
            )
            await second.prepare(project, Path(temp_dir))
            hits = await second.retrieve(project, project.bible.outline[1], {})
            self.assertTrue(hits)
            self.assertEqual(hits[0]["chapter"], 1)


if __name__ == "__main__":
    unittest.main()
