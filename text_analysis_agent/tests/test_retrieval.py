from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from reader_agent.corpus import Corpus, digest
from reader_agent.retrieval import APIEmbedder, APIReranker, HybridRetriever, QdrantStore, SemanticIndex, normalize_vectors, keyword_pairs


class FakeEmbedder:
    """确定性模拟嵌入，不使用网络或模型密钥。"""
    identity = "测试嵌入模型"

    def __init__(self):
        self.calls = 0

    async def embed(self, texts, *, query=False):
        self.calls += 1
        return [[1.0, 0.0] if "铜" in text else [0.0, 1.0] for text in texts]

    async def close(self):
        pass


class SemanticTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.corpus = Corpus(root / "corpus", create=True)
        self.ids = []
        for name, text in [("甲", "铜钥匙打开了铁门。"), ("乙", "银钥匙打开了木门。")]:
            path = root / (name + ".txt")
            path.write_text(text, encoding="utf-8")
            self.ids.append(self.corpus.ingest(path)["id"])
        self.embedder = FakeEmbedder()
        self.store = QdrantStore(self.corpus, self.embedder.identity)
        self.index = SemanticIndex(self.corpus, self.embedder, self.store)

    async def asyncTearDown(self):
        await self.store.close()
        self.corpus.close()
        self.temporary.cleanup()

    async def test_local_all_candidates_multiple_views_batches_cache_and_progress(self):
        from reader_agent.api_calls import api_context
        prototype = self.corpus.all_chunks(self.ids)[0]
        chunks = [replace(prototype, id=f'候选_{i}', text=f'原文_{i}：事件记录。') for i in range(140)]
        class Rank:
            local = True
            identity = '本地多遍测试'
            progress_callback = None
            def __init__(self): self.calls = []
            def window_count(self, text): return 2
            async def score(self, question, documents):
                self.calls.append((question, list(documents)))
                for _ in documents:
                    self.progress_callback(1, 0)
                    self.progress_callback(1, 0)
                return [0.5] * len(documents)
        rank = Rank()
        retriever = HybridRetriever(self.corpus, None, rank, rerank_max_documents=0,
                                    rerank_max_chars=0, local_passes=2, local_batch_size=8)
        question = '某人物在哪里发生某事件'
        groups = {question: chunks, '人物 事件': chunks[:70], '人物 发生': chunks[70:]}
        async def recall(query, source_ids): return groups[query]
        retriever.candidates_for = recall
        notices = []
        with api_context(notices.append, lambda: False):
            await retriever.search_many(question, list(groups), self.ids, limit=6)
        self.assertEqual(retriever.rerank_stats['scored_candidates'], 140)
        self.assertTrue(retriever.rerank_stats['candidate_coverage_complete'])
        self.assertEqual(retriever.rerank_stats['documents'], 280)
        self.assertEqual(retriever.rerank_stats['windows'], 560)
        self.assertEqual(retriever.rerank_stats['passes'], 2)
        self.assertTrue(all(len(documents) <= 8 for _, documents in rank.calls))
        self.assertTrue(all(query.startswith(question) for query, _ in rank.calls))
        for chunk in chunks:
            scored_for = [query for query, documents in rank.calls if chunk.text in documents]
            self.assertEqual(len(scored_for), 2)
            self.assertEqual(len(set(scored_for)), 2)
        detail = notices[-1].detail
        self.assertEqual((detail['current'], detail['total']), (560, 560))
        self.assertEqual(detail['remaining_seconds'], 0)
        self.assertTrue(any(item.detail['eta_at'] for item in notices if item.detail['current']))
        count = len(rank.calls)
        await retriever.search_many(question, [], self.ids, limit=6)
        self.assertEqual(len(rank.calls), count)
        # 新检索器也应从磁盘复用各表述评分，而不是重复推理。
        again = HybridRetriever(self.corpus, None, rank, local_passes=2)
        again.candidates_for = recall
        await again.search_many(question, list(groups), self.ids, limit=6)
        self.assertEqual(len(rank.calls), count)

    async def test_local_later_round_scores_all_new_candidates_without_repeating_old(self):
        chunks = self.corpus.all_chunks(self.ids)
        class Rank:
            local = True
            def __init__(self): self.calls = []
            async def score(self, question, documents):
                self.calls.extend((question, text) for text in documents)
                return [0.2] * len(documents)
        rank = Rank()
        retriever = HybridRetriever(self.corpus, None, rank, local_passes=2, rerank_max_documents=1)
        async def recall(query, source_ids): return chunks[:1] if query == '第一查询' else chunks
        retriever.candidates_for = recall
        retriever.begin_round(1, 2)
        await retriever.search_many('当前问题', ['第一查询'], self.ids, limit=2)
        retriever.begin_round(2, 2)
        await retriever.search_many('当前问题', ['新查询'], self.ids, limit=2)
        self.assertEqual(len(rank.calls), 4)
        self.assertEqual(len(set(rank.calls)), 4)
        self.assertEqual(retriever.rerank_stats['scored_candidates'], 2)

    async def test_incremental_build_hybrid_scope_and_resume(self):
        with self.assertRaises(ValueError):
            await self.index.ready(self.ids)
        first = await self.index.build(self.ids, batch_size=1)
        self.assertEqual(first["new_chunks"], 2)
        calls = self.embedder.calls
        again = await self.index.build(self.ids)
        self.assertEqual(again["reused_chunks"], 2)
        self.assertEqual(self.embedder.calls, calls)
        retriever = HybridRetriever(self.corpus, self.index)
        await retriever.prepare([self.ids[0]])
        hits = await retriever.search("钥匙", [self.ids[0]], limit=4)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source_id, self.ids[0])
        self.assertIn("铜钥匙", hits[0].text)

    async def test_required_phrases_keep_source_scope(self):
        self.assertEqual(self.corpus.search('钥匙', [self.ids[0]], must_contain=('银钥匙',)), [])
        self.assertEqual(self.corpus.search('钥匙', self.ids, must_contain=('铜钥匙', '木门')), [])
        hits = self.corpus.search('钥匙', self.ids, must_contain=('铜钥匙', '铁门'))
        self.assertEqual([c.source_id for c in hits], [self.ids[0]])

    async def test_single_channel_evidence_survives_fusion_until_rerank(self):
        from types import SimpleNamespace
        distractor = self.corpus.all_chunks([self.ids[0]])[0]
        evidence = self.corpus.all_chunks([self.ids[1]])[0]
        class Dense:
            async def search(self, question, source_ids, limit):
                return [evidence]
        class Rank:
            async def score(self, question, documents):
                return [1.0 if '银钥匙' in text else 0.0 for text in documents]
        corpus = SimpleNamespace(search=lambda *args, **kwargs: [distractor])
        graph = SimpleNamespace(expand=lambda *args, **kwargs: [distractor])
        retriever = HybridRetriever(corpus, Dense(), Rank(), candidates=1, graph=graph)
        hits = await retriever.search('钥匙', self.ids, limit=1)
        self.assertEqual(hits[0].id, evidence.id)

    async def test_merged_queries_share_scores_and_persistent_cache(self):
        chunks = self.corpus.all_chunks(self.ids)
        class Dense:
            async def search(self, question, ids, limit): return chunks
        class Rank:
            identity = '测试重排版本一'
            calls = []
            async def score(self, question, documents):
                self.calls.append((question, documents))
                return [1.0 if '银钥匙' in text else 0.0 for text in documents]
        rank = Rank()
        first = HybridRetriever(self.corpus, Dense(), rank)
        results = await first.search_many('哪把钥匙', ['钥匙', '钥匙 铁门'], self.ids, limit=2)
        self.assertEqual(len(rank.calls), 1)
        self.assertEqual(len(rank.calls[0][1]), 2)
        self.assertEqual(first.rerank_stats['documents'], 2)
        await first.search_many('哪把钥匙', ['钥匙'], self.ids, limit=2)
        self.assertEqual(len(rank.calls), 1)
        second = HybridRetriever(self.corpus, Dense(), rank)
        await second.search_many('哪把钥匙', ['钥匙'], self.ids, limit=2)
        self.assertEqual(len(rank.calls), 1)
        self.assertEqual(second.rerank_stats['documents'], 0)
        self.assertTrue(all(group[0].id == results[0][0].id for group in results))
        await second.search_many('换个问题', ['钥匙'], self.ids, limit=2)
        self.assertEqual(len(rank.calls), 2)
        rank.identity = '测试重排版本二'
        third = HybridRetriever(self.corpus, Dense(), rank)
        await third.search_many('哪把钥匙', ['钥匙'], self.ids, limit=2)
        self.assertEqual(len(rank.calls), 3)

    async def test_budget_gives_rewritten_query_a_scoring_slot(self):
        template = self.corpus.all_chunks(self.ids)[0]
        broad = [replace(template, id=f'broad-{index}', text=f'主角的一般经历{index}') for index in range(20)]
        event = replace(template, id='event', text='林舟在石室完成突破。')
        class Rank:
            calls = []
            async def score(self, question, documents):
                self.calls.extend(documents)
                return [0.9 if text == event.text else 0.5 for text in documents]
        rank = Rank()
        retriever = HybridRetriever(self.corpus, self.index, rank, rerank_max_documents=2)
        retriever.candidates_for = AsyncMock(side_effect=[broad, [event]])
        results = await retriever.search_many('林舟在哪里突破', ['宽泛查询', '事件改写'], self.ids, limit=2)
        self.assertEqual(rank.calls, [broad[0].text, event.text])
        self.assertEqual(results[1][0].id, event.id)
        self.assertEqual(retriever.rerank_stats['documents'], 2)

    async def test_round_budget_keeps_old_candidates_and_reserves_later_scoring(self):
        template = self.corpus.all_chunks(self.ids)[0]
        first = [replace(template, id=f'first-{i}', text=f'第一路候选{i}') for i in range(6)]
        second = replace(template, id='second', text='第二路新事件原文')
        class Rank:
            calls = []
            async def score(self, question, documents):
                self.calls.extend(documents)
                return [0.5] * len(documents)
        rank = Rank()
        retriever = HybridRetriever(self.corpus, self.index, rank, rerank_max_documents=4)
        retriever.candidates_for = AsyncMock(side_effect=[first, [second]])
        retriever.begin_round(1, 2)
        await retriever.search_many('同一问题', ['第一轮查询'], self.ids, limit=2)
        self.assertEqual(len(rank.calls), 2)
        retriever.begin_round(2, 2)
        results = await retriever.search_many('同一问题', ['第二轮查询'], self.ids, limit=2)
        self.assertEqual(len(rank.calls), 4)
        self.assertIn(second.text, rank.calls)
        self.assertEqual(len(results), 2)
        self.assertEqual(retriever.retrieval_trace[-1]['groups'][0]['candidate_ids'], [c.id for c in first])

    def test_best_reranked_evidence_precedes_protected_recall_candidates(self):
        template = self.corpus.all_chunks(self.ids)[0]
        noise = [replace(template, id=f'noise-{i}', text=f'无关片段{i}') for i in range(6)]
        evidence = replace(template, id='evidence', text='当前问题的事件直接依据')
        candidates = [*noise, evidence]
        scores = {digest(chunk.text): 1.0 if chunk.id == 'evidence' else 0.1 for chunk in candidates}
        hits = HybridRetriever.select_ranked(candidates, scores, 6)
        self.assertEqual(hits[0].id, evidence.id)
        self.assertIn(noise[0].id, [chunk.id for chunk in hits])
        self.assertIn(noise[1].id, [chunk.id for chunk in hits])

    def test_weak_reranker_cannot_remove_top_recall_evidence(self):
        template = self.corpus.all_chunks(self.ids)[0]
        target = replace(template, id='target', text='直接写明事件地点的原文')
        noise = [replace(template, id=f'noise-{i}', text=f'没有事件依据的背景{i}') for i in range(8)]
        scores = {digest(c.text): 0.99 if c.id != target.id else 0.7 for c in [target, *noise]}
        results = HybridRetriever.select_ranked([target, *noise], scores, 6)
        self.assertIn(target.id, [c.id for c in results])
        self.assertEqual(len(results), 6)

    async def test_rerank_budget_is_shared_across_rounds_and_keeps_unscored_evidence(self):
        chunks = self.corpus.all_chunks(self.ids)
        class Dense:
            async def search(self, question, ids, limit): return chunks
        class Rank:
            calls = []
            async def score(self, question, documents):
                self.calls.extend(documents)
                return [0.5] * len(documents)
        rank = Rank()
        retriever = HybridRetriever(self.corpus, Dense(), rank, rerank_max_documents=1)
        for query in ('钥匙', '门'):
            groups = await retriever.search_many('钥匙的用途', [query], self.ids, limit=2)
            self.assertEqual(len(groups[0]), 2)
        self.assertEqual(len(rank.calls), 1)
        self.assertEqual(retriever.rerank_stats['documents'], 1)
        blocked = HybridRetriever(self.corpus, Dense(), rank, rerank_max_chars=1)
        groups = await blocked.search_many('钥匙的用途', ['钥匙'], self.ids, limit=2)
        self.assertEqual(len(rank.calls), 1)
        self.assertEqual(len(groups[0]), 2)

    async def test_invalid_rerank_result_is_never_cached(self):
        chunks = self.corpus.all_chunks(self.ids)
        class Rank:
            identity = '不合格重排测试'
            async def score(self, question, documents): return [float('nan')] * len(documents)
        retriever = HybridRetriever(self.corpus, self.index, Rank())
        with self.assertRaises(ValueError):
            await retriever.score_pool('问题', chunks)
        self.assertEqual(retriever.cache.get(Rank.identity, digest('问题'), [digest(c.text) for c in chunks]), {})

    async def test_event_candidates_survive_fusion_and_bounded_rerank(self):
        path = Path(self.temporary.name) / '升仙.txt'
        path.write_text('第一章 往事\n林舟曾问别人六转蛊仙的修为。\n第二章 碎窍\n林舟在石室中升仙，最终成功。', encoding='utf-8')
        source = self.corpus.ingest(path)['id']
        chunks = self.corpus.all_chunks([source])
        class Dense:
            async def search(self, question, ids, limit):
                return [chunks[0]]
        class Rank:
            calls = []
            async def score(self, question, documents):
                self.calls.append(documents)
                return [1.0 if '最终成功' in text else 0.0 for text in documents]
        rank = Rank()
        retriever = HybridRetriever(self.corpus, Dense(), rank, candidates=1)
        hits = await retriever.search('林舟 升仙', [source], limit=1)
        self.assertIn('最终成功', hits[0].text)
        self.assertTrue(all(len(batch) <= 32 for batch in rank.calls))
        self.assertEqual(keyword_pairs('林舟 升仙')[0], ('林舟', '升仙'))
        self.assertEqual(keyword_pairs('铜钥匙 打开 铁门'), [('铜钥匙', '打开'), ('铜钥匙', '铁门'), ('打开', '铁门')])
        self.assertEqual(keyword_pairs('林舟成为掌门了吗'), [])

    async def test_missing_remote_collection_rebuilds_saved_progress(self):
        await self.index.build(self.ids)
        await self.store.client.delete_collection(self.store.collection)
        rebuilt = await self.index.build(self.ids)
        self.assertEqual(rebuilt["new_chunks"], 2)
        await self.index.ready(self.ids)

    async def test_interruption_reuses_committed_batches(self):
        def stop(message):
            if "新增" in message:
                raise RuntimeError("模拟索引中断")
        with self.assertRaises(RuntimeError):
            await self.index.build(self.ids, batch_size=1, progress=stop)
        resumed = await self.index.build(self.ids, batch_size=1)
        self.assertEqual(resumed["reused_chunks"], 1)
        self.assertEqual(resumed["new_chunks"], 1)

    async def test_forged_payload_and_missing_vectors_are_rejected(self):
        await self.index.build(self.ids)
        chunk = self.corpus.all_chunks([self.ids[0]])[0]
        await self.store.client.set_payload(self.store.collection, payload={"text_sha256": "伪造哈希"}, points=[self.store.point_id(chunk.id)])
        with self.assertRaises(ValueError):
            await self.index.search("铜钥匙", [self.ids[0]], 3)
        await self.store.client.delete(self.store.collection, points_selector=[self.store.point_id(chunk.id)])
        with self.assertRaises(ValueError):
            await self.index.ready(self.ids)
        self.index.repair()
        await self.index.build(self.ids)
        await self.index.ready(self.ids)


class APIProtocolTest(unittest.IsolatedAsyncioTestCase):
    async def test_qianwen_native_rerank_endpoint_body_and_response(self):
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={'output': {'results': [{'index': 1, 'relevance_score': 0.3}, {'index': 0, 'relevance_score': 0.8}]}})
        reranker = APIReranker(api_key='fake', model='qwen3.7-text-rerank', url='https://maas.qianwenaiapi.com/compatible-mode/v1')
        await reranker.close()
        reranker.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            self.assertEqual(await reranker.score('问题', ['甲', '乙']), [0.8, 0.3])
            self.assertEqual(str(requests[0].url), 'https://maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank')
            body = json.loads(requests[0].content)
            self.assertEqual(body['input'], {'query': '问题', 'documents': ['甲', '乙']})
            self.assertEqual(body['parameters']['top_n'], 2)
            unknown = APIReranker(api_key='fake', model='qwen3.7-text-rerank', url='https://another.test/compatible-mode/v1')
            self.assertFalse(unknown.native)
            self.assertEqual(unknown.url, 'https://another.test/compatible-mode/v1')
            await unknown.close()
        finally:
            await reranker.close()

    async def test_rerank_http_error_has_safe_actionable_message(self):
        reranker = APIReranker(api_key='fake', model='测试模型', url='https://model.test/rerank')
        await reranker.close()
        reranker.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404, text='原文和密钥不能暴露')))
        try:
            with self.assertRaises(ValueError) as caught:
                await reranker.score('问题', ['甲'])
            self.assertIn('原文重排接口返回 HTTP 404', str(caught.exception))
            self.assertNotIn('原文和密钥', str(caught.exception))
            self.assertIsInstance(caught.exception.__cause__, httpx.HTTPStatusError)
        finally:
            await reranker.close()

    async def test_embedding_reorders_indices_and_applies_query_prefix(self):
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"object": "list", "model": "模拟模型", "data": [
                {"object": "embedding", "index": 1, "embedding": [0, 2]},
                {"object": "embedding", "index": 0, "embedding": [3, 0]}], "usage": {"prompt_tokens": 2, "total_tokens": 2}})
        embedder = APIEmbedder(api_key="fake", model="模拟模型", base_url="https://model.test/v1", query_prefix="查询：")
        await embedder.close()
        embedder.client = AsyncOpenAI(api_key="fake", base_url="https://model.test/v1", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        try:
            vectors = await embedder.embed(["甲", "乙"], query=True)
            self.assertEqual(vectors, [[1, 0], [0, 1]])
            self.assertEqual(requests[0]["input"], ["查询：甲", "查询：乙"])
        finally:
            await embedder.close()

    async def test_reranker_rejects_missing_or_duplicate_indices(self):
        rows = [{"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.8}]
        reranker = APIReranker(api_key="fake", model="模拟模型", url="https://model.test/rerank")
        await reranker.close()
        reranker.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"results": rows})))
        try:
            with self.assertRaises(ValueError):
                await reranker.score("问题", ["甲", "乙"])
            rows[:] = [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.9}]
            self.assertEqual(await reranker.score("问题", ["甲", "乙"]), [0.9, 0.8])
        finally:
            await reranker.close()

    async def test_invalid_vectors_are_rejected(self):
        for vectors in ([[0, 0]], [[float("nan"), 1]], [[True, 1]], [[1, 2], [1]], []):
            with self.assertRaises(ValueError):
                normalize_vectors(vectors, max(1, len(vectors)))
