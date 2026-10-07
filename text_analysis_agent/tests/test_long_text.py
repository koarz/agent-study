from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reader_agent.corpus import Corpus, split_stream
from reader_agent.graph import EvidenceGraph, graph_draft, graph_batch_draft
from reader_agent.scan import FullScanner, scan_plan


class EvidenceBackend:
    """按原文内容生成确定性响应，用于扫描和恢复测试。"""
    def __init__(self, *, failure_after=None):
        self.calls = 0
        self.failure_after = failure_after

    async def complete(self, messages):
        if self.failure_after is not None and self.calls >= self.failure_after:
            raise RuntimeError("模拟中断")
        self.calls += 1
        payload = json.loads(messages[1]["content"])
        if "claims" in payload:
            return json.dumps({"verdicts": [{"claim_id": c["id"], "supported": True} for c in payload["claims"]],
                               "question_fully_answered": False})
        if "facts" in messages[0]["content"]:
            return json.dumps({"status": "unclear", "facts": []})
        claims = []
        for chunk in payload["evidence"]:
            phrase = "林舟用铜钥匙打开了铁门。"
            if phrase in chunk["text"]:
                claims.append({"id": "c" + str(len(claims)), "text": phrase,
                               "citations": [{"chunk_id": chunk["id"], "quote": phrase}]})
        return json.dumps({"status": "unclear" if claims else "not_found", "claims": claims}, ensure_ascii=False)


class StreamingTest(unittest.TestCase):
    def test_suspended_chunk_iterators_allow_other_index_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "书.txt"
            path.write_text("甲乙丙丁。" * 2500, encoding="utf-8")
            first = Corpus(Path(temporary) / "corpus", create=True)
            second = None
            try:
                first.ingest(path, chunk_chars=100, overlap=0)
                with first.db:
                    first.db.execute("CREATE TABLE concurrent_progress (id INTEGER PRIMARY KEY)")
                second = Corpus(first.directory)
                ids = first.select_sources()
                one, two = first.iter_chunks(ids), second.iter_chunks(ids)
                start_one, start_two = next(one), next(two)
                # 两个扫描均停在首批，交替写入不能等待对方扫描结束。
                with first.db:
                    first.db.execute("INSERT INTO concurrent_progress VALUES (1)")
                with second.db:
                    second.db.execute("INSERT INTO concurrent_progress VALUES (2)")
                all_one, all_two = [start_one, *one], [start_two, *two]
                self.assertGreater(len(all_one), 64)
                self.assertEqual([c.id for c in all_one], [c.id for c in all_two])
                self.assertEqual(len({c.id for c in all_one}), first.stats(ids)["chunks"])
                self.assertEqual(first.db.execute("SELECT count(*) FROM concurrent_progress").fetchone()[0], 2)
            finally:
                if second is not None:
                    second.close()
                first.close()

    def test_no_newlines_and_heading_boundaries_preserve_all_offsets(self):
        text = "开场白。" * 30 + "\r\n第一章 明灯\r\n" + "甲乙丙丁。" * 70 + "\r\n第二章 后街\r\n末尾线索。"
        blocks = list(split_stream(io.StringIO(text), 100, 20))
        covered = set()
        for block in blocks:
            self.assertEqual(text[block.start:block.end], block.text)
            self.assertEqual(text.encode()[block.byte_start:block.byte_end].decode(), block.text)
            self.assertEqual(block.line_start, text.count("\n", 0, block.start) + 1)
            self.assertEqual(block.line_end, text.count("\n", 0, block.end - 1) + 1)
            covered.update(range(block.start, block.end))
        self.assertTrue(all(i in covered for i, char in enumerate(text) if not char.isspace()))
        self.assertTrue(any(block.chapter == "第二章 后街" for block in blocks))

    def test_ingest_never_reads_whole_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "书.txt"
            path.write_text("第一章\n" + "无换行的超长段落。" * 3000, encoding="utf-8")
            corpus = Corpus(Path(temporary) / "corpus", create=True)
            try:
                with patch.object(Path, "read_bytes", side_effect=AssertionError("不能整本读取")), \
                     patch.object(Path, "read_text", side_effect=AssertionError("不能整本读取")):
                    result = corpus.ingest(path)
                    self.assertGreater(result["chunks"], 5)
                    self.assertTrue(corpus.search("超长段落", corpus.select_sources()))
                    self.assertFalse(corpus._verified)
            finally:
                corpus.close()

    def test_new_snapshot_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "书.txt"
            path.write_text("第一章\n明确的原文内容。", encoding="utf-8")
            corpus = Corpus(Path(temporary) / "corpus", create=True)
            try:
                corpus.ingest(path)
                ids = corpus.select_sources()
                chunks = corpus.all_chunks(ids)
                snapshot = next((corpus.directory / "snapshots").glob("*.txt"))
                snapshot.write_text("被改写的内容。", encoding="utf-8")
                with self.assertRaises(ValueError):
                    corpus.citation(chunks[0], "明确的原文内容。")
            finally:
                corpus.close()


class ScanTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        path = self.root / "长篇.txt"
        path.write_text(("林舟用铜钥匙打开了铁门。\n" + "小雨下了一夜。" * 30 + "\n") * 12, encoding="utf-8")
        self.corpus = Corpus(self.root / "corpus", create=True)
        self.corpus.ingest(path, chunk_chars=500, overlap=100)

    def tearDown(self):
        self.corpus.close()
        self.temp.cleanup()

    async def test_full_scan_is_bounded_and_resumes_without_repeating_model_calls(self):
        backend = EvidenceBackend()
        result = await FullScanner(self.corpus, backend, batch_chars=1000).scan("林舟怎样开门？")
        self.assertEqual(result["status"], "scan_complete")
        self.assertEqual(result["coverage"]["selected_chunks"], result["coverage"]["total_chunks"])
        records = [json.loads(line) for line in Path(result["evidence_file"]).read_text().splitlines()]
        self.assertTrue(records)
        first_count = backend.calls
        again = await FullScanner(self.corpus, backend, batch_chars=1000).scan("林舟怎样开门？")
        self.assertEqual(backend.calls, first_count)
        self.assertEqual(again["resumed_batches"], again["coverage"]["total_batches"])

    async def test_interrupted_scan_saves_report_and_restores_completed_batches(self):
        backend = EvidenceBackend(failure_after=2)
        with self.assertRaises(RuntimeError):
            await FullScanner(self.corpus, backend, batch_chars=1000).scan("林舟怎样开门？")
        summary = next((self.corpus.directory / "analyses").glob("*/summary.json"))
        partial = json.loads(summary.read_text())
        self.assertEqual(partial["status"], "scan_partial")
        backend.failure_after = None
        result = await FullScanner(self.corpus, backend, batch_chars=1000).scan("林舟怎样开门？")
        self.assertEqual(result["status"], "scan_complete")
        self.assertGreater(result["resumed_batches"], 0)

    async def test_verifier_change_requires_new_review(self):
        backend, verifier = EvidenceBackend(), EvidenceBackend()
        verifier.model = "复核模型甲"
        first = await FullScanner(self.corpus, backend, verifier=verifier, batch_chars=1000).scan("林舟怎样开门？")
        verifier.model = "复核模型乙"
        next_run = await FullScanner(self.corpus, backend, verifier=verifier, batch_chars=1000).scan("林舟怎样开门？")
        self.assertNotEqual(first["report"], next_run["report"])
        self.assertEqual(next_run["resumed_batches"], 0)

    async def test_graph_only_stores_reviewed_relations_with_original_citations(self):
        class GraphBackend(EvidenceBackend):
            async def complete(self, messages):
                payload = json.loads(messages[1]["content"])
                if "claims" in payload:
                    return await super().complete(messages)
                facts = []
                for chunk in payload["evidence"]:
                    phrase = "林舟用铜钥匙打开了铁门。"
                    if phrase in chunk["text"]:
                        facts.append({"id": "f" + str(len(facts)), "subject": "林舟", "predicate": "打开",
                                      "object": "铁门", "time_text": "", "attribution": "narration", "certainty": "explicit",
                                      "citations": [{"chunk_id": chunk["id"], "quote": phrase}]})
                return json.dumps({"status": "unclear", "facts": facts}, ensure_ascii=False)
        graph = EvidenceGraph(self.corpus)
        result = await graph.build(GraphBackend(), batch_chars=1000)
        self.assertGreater(result["graph_facts"], 0)
        ids = self.corpus.select_sources()
        facts = graph.facts("林舟", ids)
        self.assertTrue(facts)
        self.assertEqual(facts[0]["fact"]["object"], "铁门")
        self.assertEqual(facts[0]["citations"][0]["quote"], "林舟用铜钥匙打开了铁门。")
        self.assertTrue(graph.expand("林舟的事情", ids))
        other_path = self.root / "另一本.txt"
        other_path.write_text("林舟没有出现。", encoding="utf-8")
        other = self.corpus.ingest(other_path)["id"]
        self.assertEqual(graph.facts("林舟", [other]), [])
        self.assertEqual(graph.expand("林舟", [other]), [])

    def test_dry_run_has_no_model_calls_and_rejects_too_small_budget(self):
        plan = scan_plan(self.corpus, self.corpus.select_sources(), 1000)
        self.assertGreater(plan["total_batches"], 1)
        self.assertEqual(plan["model_calls_max"], plan["total_batches"] * 20)
        self.assertEqual(plan["model_calls_without_api_retry_max"], plan["total_batches"] * 4)
        with self.assertRaises(ValueError):
            scan_plan(self.corpus, self.corpus.select_sources(), 100)

    def test_graph_fields_cannot_bypass_quote_checks(self):
        chunk = self.corpus.all_chunks(self.corpus.select_sources())[0]
        raw = {"status": "unclear", "facts": [{"id": "f1", "subject": "林舟", "predicate": "打开",
               "object": "铁门", "time_text": "", "attribution": "narration", "certainty": "explicit",
               "citations": [{"chunk_id": chunk.id, "quote": "林舟用铜钥匙打开了铁门。"}]}]}
        draft = graph_draft(raw)
        self.assertIn('"subject": "林舟"', draft["claims"][0]["text"])
        raw["facts"][0]["subject"] = "不存在的人物"
        with self.assertRaises(ValueError):
            graph_draft(raw)

    def test_bad_graph_record_does_not_discard_valid_original_evidence(self):
        chunk = self.corpus.all_chunks(self.corpus.select_sources())[0]
        good = {"id": "f1", "subject": "林舟", "predicate": "打开", "object": "铁门", "time_text": "",
                "attribution": "narration", "certainty": "explicit", "citations": [{"chunk_id": chunk.id, "quote": "林舟用铜钥匙打开了铁门。"}]}
        bad = {**good, "id": "f2", "subject": "虚构人物"}
        draft = graph_batch_draft({"status": "answered", "facts": [good, bad]}, {chunk.id: chunk})
        self.assertEqual(len(draft["claims"]), 1)
        self.assertEqual(draft["pre_discarded"], 1)
        self.assertEqual(draft["status"], "unclear")

    def test_context_expansion_keeps_exact_original_words_and_does_not_repair_fake_quotes(self):
        chunk = self.corpus.all_chunks(self.corpus.select_sources())[0]
        fact = {"id": "f1", "subject": "林舟", "predicate": "打开", "object": "铁门", "time_text": "",
                "attribution": "narration", "certainty": "explicit", "citations": [{"chunk_id": chunk.id, "quote": "铜钥匙打开了铁门。"}]}
        draft = graph_batch_draft({"status": "answered", "facts": [fact]}, {chunk.id: chunk})
        quote = draft["claims"][0]["citations"][0]["quote"]
        self.assertIn("林舟", quote)
        self.assertIn(quote, chunk.text)
        fact["citations"][0]["quote"] = "不存在的引文。"
        draft = graph_batch_draft({"status": "answered", "facts": [fact]}, {chunk.id: chunk})
        self.assertEqual(draft["claims"], [])
