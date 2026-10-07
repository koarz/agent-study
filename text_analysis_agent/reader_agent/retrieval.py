"""混合检索：BM25、语义向量、证据图谱与交叉编码器重排。

各检索器仅提供原文定位信息，真正的证据必须经过快照校验。
向量载荷和图谱描述不能直接作为回答依据。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import uuid
from typing import Protocol

from .corpus import Chunk, Corpus, digest
from .logging_utils import emit, model_operation
from .progress import ProgressTracker
from .api_calls import api_call, notify, check_stopping


class Embedder(Protocol):
    identity: str
    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]: ...


class Reranker(Protocol):
    async def score(self, question: str, documents: list[str]) -> list[float]: ...


def embedding_identity(model: str, base_url: str, prefix: str, dimensions: int | None, *, local=False) -> str:
    metadata = [model, prefix, "local", 8192] if local else [model, base_url, prefix, dimensions]
    return hashlib.sha256(json.dumps(metadata).encode()).hexdigest()


def configured_embedding_identity(config: dict) -> str:
    dimensions = config.get("EMBEDDING_DIMENSIONS", "").strip()
    return embedding_identity(config.get("EMBEDDING_MODEL_ID", ""),
                              config.get("EMBEDDING_BASE_URL") or config.get("LLM_BASE_URL", ""),
                              config.get("EMBEDDING_QUERY_PREFIX", ""), int(dimensions) if dimensions else None,
                              local=config.get("EMBEDDING_PROVIDER") == "local")


def normalize_vectors(vectors, count: int) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != count or not vectors:
        raise ValueError("嵌入服务未返回全部向量")
    dimension = len(vectors[0])
    if not dimension:
        raise ValueError("嵌入向量不能为空")
    normalized = []
    for vector in vectors:
        if len(vector) != dimension or any(type(v) not in {float, int} or not math.isfinite(v) for v in vector):
            raise ValueError("嵌入向量维度不一致或包含非有限值")
        norm = math.sqrt(sum(v * v for v in vector))
        if not norm:
            raise ValueError("嵌入服务返回了零向量")
        normalized.append([v / norm for v in vector])
    return normalized


class APIEmbedder:
    def __init__(self, *, api_key: str, model: str, base_url: str, timeout: float = 120,
                 query_prefix: str = "", dimensions: int | None = None):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.model = model
        self.query_prefix = query_prefix
        self.dimensions = dimensions
        self.identity = embedding_identity(model, base_url, query_prefix, dimensions)

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        with model_operation("查询嵌入" if query else "原文嵌入", self.model, documents=len(texts)) as metadata:
            result = await self._embed(texts, query=query)
            metadata["dimensions"] = len(result[0])
            return result

    async def _embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        kwargs = {"model": self.model, "input": [self.query_prefix + t if query else t for t in texts],
                  "encoding_format": "float"}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = await api_call(lambda: self.client.embeddings.create(**kwargs), operation="查询嵌入" if query else "原文嵌入",
                                  api_key=self.client.api_key, endpoint=self.client.base_url)
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise ValueError("嵌入服务返回了重复或缺失的输入编号")
        return normalize_vectors([item.embedding for item in ordered], len(texts))

    async def close(self):
        await self.client.close()


class APIReranker:
    """接入 Cohere/Jina 兼容或千问原生重排接口。

    服务返回的分数只表示相关性，不表示结论真实。
    """
    def __init__(self, *, api_key: str, model: str, url: str, timeout: float = 120):
        import httpx
        from urllib.parse import urlsplit, urlunsplit

        self.client = httpx.AsyncClient(timeout=timeout)
        endpoint = urlsplit(url)
        path = endpoint.path.rstrip("/")
        native_path = "/api/v1/services/rerank/text-rerank/text-rerank"
        official_host = endpoint.hostname in {"maas.qianwenaiapi.com", "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com"} or (endpoint.hostname or "").endswith(".maas.aliyuncs.com")
        # 仅按已确认的千问官方地址及模型修正根路径，不猜测其他服务商的接口。
        if official_host and path in {"", "/v1", "/compatible-mode/v1", "/compatible-api/v1"} and model in {"qwen3.7-text-rerank", "qwen3-vl-rerank", "gte-rerank-v2"}:
            url = urlunsplit(endpoint._replace(path=native_path))
            path = native_path
            emit("使用千问原生重排接口", model=model)
        self.native = path.endswith(native_path)
        self.api_key, self.model, self.url = api_key, model, url
        self.identity = digest(json.dumps(['api', model, url, digest(api_key), 'full-text-v1']))

    async def score(self, question: str, documents: list[str]) -> list[float]:
        self.last_usage = {}
        with model_operation("原文重排", self.model, documents=len(documents),
                             document_chars=sum(map(len, documents)), query_chars=len(question)) as metadata:
            result = await self._score(question, documents)
            metadata.update(self.last_usage)
            return result

    async def _score(self, question: str, documents: list[str]) -> list[float]:
        import httpx
        inputs = {"query": question, "documents": documents}
        parameters = {"top_n": len(documents), "return_documents": False}
        body = {"model": self.model, "input": inputs, "parameters": parameters} if self.native else {
            "model": self.model, **inputs, **parameters}
        async def request():
            response = await self.client.post(self.url, headers={"Authorization": "Bearer " + self.api_key}, json=body)
            if response.status_code == 429:
                response.raise_for_status()
            return response
        response = await api_call(request, operation="原文重排", api_key=self.api_key, endpoint=self.url)
        if response.is_error:
            # 外部错误响应可能含提示词或密钥，只回传状态码及本项目定义的排错说明。
            descriptions = {400: "请求格式或输入长度不符合要求", 401: "密钥无效", 403: "密钥没有模型访问权限",
                            404: "接口地址或模型不存在，请核对重排完整地址及模型名称", 429: "请求受到限流或账户额度不足"}
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ValueError(f"原文重排接口返回 HTTP {response.status_code}：{descriptions.get(response.status_code, '服务暂时不可用')}。") from exc
        result = response.json()
        usage = result.get('usage', {}) if isinstance(result, dict) else {}
        if isinstance(usage, dict):
            self.last_usage = {key: value for key, value in usage.items()
                               if key in {'input_tokens', 'output_tokens', 'total_tokens'}
                               and type(value) is int and value >= 0}
        output = result.get("output", {}) if self.native and isinstance(result, dict) else result
        rows = output.get("results") if isinstance(output, dict) else None
        if not isinstance(rows, list):
            raise ValueError("重排服务响应格式无效")
        scores = {}
        for row in rows:
            if not isinstance(row, dict) or type(row.get("index")) is not int:
                raise ValueError("重排服务未返回合法文档编号")
            index, value = row["index"], row.get("relevance_score")
            if index in scores or not 0 <= index < len(documents) or type(value) not in {float, int} or not math.isfinite(value):
                raise ValueError("重排服务返回了重复编号或无效分数")
            scores[index] = float(value)
        if len(scores) != len(documents):
            raise ValueError("重排服务未返回全部候选文档")
        return [scores[i] for i in range(len(documents))]

    async def close(self):
        await self.client.aclose()


class LocalEmbedder:
    def __init__(self, model: str, *, device: str = "cpu", query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device)
        self.model.max_seq_length = 8192
        self.query_prefix = query_prefix
        self.identity = embedding_identity(model, "", query_prefix, None, local=True)

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        inputs = [self.query_prefix + t if query else t for t in texts]
        result = await asyncio.to_thread(self.model.encode, inputs, normalize_embeddings=True,
                                         batch_size=4, show_progress_bar=False)
        return normalize_vectors(result.tolist(), len(texts))

    async def close(self):
        pass


class LocalReranker:
    local = True

    def __init__(self, model: str, *, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, device=device, max_length=8192)
        self.identity = digest(json.dumps(['local', model, 'full-text-v1']))

    async def score(self, question: str, documents: list[str]) -> list[float]:
        values = await asyncio.to_thread(self.model.predict, [(question, document) for document in documents],
                                         batch_size=4, show_progress_bar=False)
        return [float(value) for value in values]

    async def close(self):
        pass


class QdrantStore:
    def __init__(self, corpus: Corpus, embedding_id: str, *, url: str | None = None, api_key: str | None = None):
        from qdrant_client import AsyncQdrantClient

        # 本地模式无需额外服务，适合个人书库。
        # 服务模式使用 HNSW 与磁盘向量，检索接口保持一致。
        kwargs = {"url": url, "api_key": api_key} if url else {"path": str(corpus.directory / "vectors")}
        self.client = AsyncQdrantClient(**kwargs)
        self.collection = "reader_" + hashlib.sha256((str(corpus.directory) + embedding_id).encode()).hexdigest()[:24]
        self.models = __import__("qdrant_client", fromlist=["models"]).models

    async def exists(self) -> bool:
        return await self.client.collection_exists(self.collection)

    async def initialize(self, dimension: int):
        if not await self.exists():
            await self.client.create_collection(self.collection,
                vectors_config=self.models.VectorParams(size=dimension, distance=self.models.Distance.COSINE,
                                                        on_disk=True),
                hnsw_config=self.models.HnswConfigDiff(on_disk=True))
        else:
            info = await self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            if vectors.size != dimension:
                raise ValueError("向量索引维度与嵌入模型不一致")

    def scope(self, source_ids: list[str]):
        return self.models.Filter(must=[self.models.FieldCondition(key="source_id",
                                                                   match=self.models.MatchAny(any=source_ids))])

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]):
        points = [self.models.PointStruct(id=self.point_id(chunk.id), vector=vector,
                  payload={"chunk_id": chunk.id, "source_id": chunk.source_id, "text_sha256": digest(chunk.text)})
                  for chunk, vector in zip(chunks, vectors)]
        await self.client.upsert(self.collection, points=points, wait=True)

    async def count(self, source_ids: list[str]) -> int:
        return (await self.client.count(self.collection, count_filter=self.scope(source_ids), exact=True)).count

    async def search(self, vector: list[float], source_ids: list[str], limit: int) -> list[dict]:
        response = await self.client.query_points(self.collection, query=vector, query_filter=self.scope(source_ids),
                                                   limit=limit, with_payload=True, with_vectors=False)
        return [point.payload for point in response.points]

    async def close(self):
        await self.client.close()


class SemanticIndex:
    def __init__(self, corpus: Corpus, embedder: Embedder, store):
        self.corpus, self.embedder, self.store = corpus, embedder, store
        self.identity = embedder.identity
        self.corpus.db.execute("""CREATE TABLE IF NOT EXISTS vector_progress (
            model TEXT NOT NULL, chunk_id TEXT NOT NULL, text_sha256 TEXT NOT NULL,
            PRIMARY KEY(model,chunk_id))""")
        self.corpus.db.commit()

    async def build(self, source_ids: list[str], *, batch_size: int = 16, progress=None):
        if not 1 <= batch_size <= 128:
            raise ValueError("嵌入批次大小须为 1 到 128")
        progress = progress or (lambda message: None)
        tracker = ProgressTracker(self.corpus.stats(source_ids)["chunks"])
        progress(tracker.update("准备语义索引", current=0, stage="准备原文与向量存储"))
        self.corpus.begin_read()
        exists = await self.store.exists()
        if not exists:
            with self.corpus.db:
                self.corpus.db.execute("DELETE FROM vector_progress WHERE model=?", (self.identity,))
        new = reused = 0
        pending = []

        async def commit():
            nonlocal new
            vectors = normalize_vectors(await self.embedder.embed([chunk.text for chunk in pending]), len(pending))
            await self.store.initialize(len(vectors[0]))
            # 先确认向量写入成功，再保存本地进度。
            # 两步之间中断时，恢复后可用相同 ID 安全重复写入。
            await self.store.upsert(pending, vectors)
            with self.corpus.db:
                self.corpus.db.executemany("INSERT OR REPLACE INTO vector_progress VALUES (?,?,?)",
                    [(self.identity, chunk.id, digest(chunk.text)) for chunk in pending])
            new += len(pending)
            progress(tracker.update(f"语义索引：新增 {new} 段，已复用 {reused} 段", current=new + reused, reused=reused, stage="建立语义索引"))
            pending.clear()

        for chunk in self.corpus.iter_chunks(source_ids):
            cached = self.corpus.db.execute("SELECT text_sha256 FROM vector_progress WHERE model=? AND chunk_id=?",
                                             (self.identity, chunk.id)).fetchone()
            if cached and cached["text_sha256"] == digest(chunk.text):
                reused += 1
                if reused % 128 == 0:
                    progress(tracker.update(f"复用语义索引：{reused} 段", current=new + reused, reused=reused, stage="恢复索引进度"))
            else:
                pending.append(chunk)
                if len(pending) == batch_size:
                    await commit()
        if pending:
            await commit()
        await self.ready(source_ids)
        progress(tracker.update("语义索引已完整核对", current=new + reused, reused=reused, stage="索引完成"))
        return {"new_chunks": new, "reused_chunks": reused, "embedding_id": self.identity}

    async def ready(self, source_ids: list[str]):
        missing = self.corpus.db.execute(
            f"""SELECT count(*) FROM chunks c LEFT JOIN vector_progress v ON v.chunk_id=c.id AND v.model=?
                WHERE c.source_id IN ({self.corpus._where(source_ids)}) AND v.chunk_id IS NULL""",
            [self.identity, *source_ids]).fetchone()[0]
        if missing or not await self.store.exists():
            raise ValueError("原文尚未建立完整的语义索引，请先运行 index；不会自动退回仅关键词检索")
        if await self.store.count(source_ids) != self.corpus.stats(source_ids)["chunks"]:
            raise ValueError("向量存储与原文段落数量不一致，请使用 index --repair 修复索引")

    async def search(self, question: str, source_ids: list[str], limit: int) -> list[Chunk]:
        vector = normalize_vectors(await self.embedder.embed([question], query=True), 1)[0]
        payloads = await self.store.search(vector, source_ids, limit)
        chunks = []
        for payload in payloads:
            if not isinstance(payload, dict) or payload.get("source_id") not in source_ids:
                raise ValueError("向量检索返回了范围之外的原文")
            chunk = self.corpus.get_chunk(payload.get("chunk_id"))
            if chunk.source_id != payload["source_id"] or digest(chunk.text) != payload.get("text_sha256"):
                raise ValueError("向量索引与原文内容不一致，停止回答")
            chunks.append(chunk)
        return chunks

    def repair(self):
        with self.corpus.db:
            self.corpus.db.execute("DELETE FROM vector_progress WHERE model=?", (self.identity,))


def rrf(rankings: list[list[Chunk]], k: int = 60) -> list[Chunk]:
    scores, chunks = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, chunk in enumerate(ranking, 1):
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            chunks[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
    return [chunks[key] for key in sorted(chunks, key=lambda key: (-scores[key], key))]


def keyword_pairs(question: str) -> list[tuple[str, str]]:
    """从精简查询中组合关键词，不识别特定人物、作品或事件类型。"""
    from itertools import combinations
    words = list(dict.fromkeys(word.strip('，,。？?：:；;“”\"') for word in question.split()))
    words = [word for word in words if 2 <= len(word) <= 24]
    if not 2 <= len(words) <= 6:
        return []
    return list(combinations(words, 2))[:3]


def phrase_distance(text: str, first: str, second: str) -> int:
    """用最近词距挑选事件候选，词距仅表示相关性，不能作为事实依据。"""
    left = [m.start() for m in re.finditer(re.escape(first), text)]
    right = [m.start() for m in re.finditer(re.escape(second), text)]
    i = j = 0
    distance = len(text)
    while i < len(left) and j < len(right):
        distance = min(distance, abs(left[i] - right[j]))
        if left[i] < right[j]:
            i += 1
        else:
            j += 1
    return distance


class HybridRetriever:
    def __init__(self, corpus: Corpus, semantic: SemanticIndex, reranker: Reranker | None = None,
                 *, candidates: int = 40, graph=None, rerank_max_documents=128, rerank_max_chars=200000,
                 local_passes=2, local_batch_size=8):
        if not 1 <= candidates <= 256 or not 0 <= rerank_max_documents <= 2048 or not 0 <= rerank_max_chars <= 4000000:
            raise ValueError('重排预算超出范围')
        if not 1 <= local_passes <= 3 or not 1 <= local_batch_size <= 32:
            raise ValueError('本地重排遍数须为 1 到 3，批大小须为 1 到 32')
        self.local_passes, self.local_batch_size = local_passes, local_batch_size
        self.corpus, self.semantic, self.reranker = corpus, semantic, reranker
        self.candidates, self.graph = candidates, graph
        self.rerank_max_documents, self.rerank_max_chars = rerank_max_documents, rerank_max_chars
        self.cache = None
        if reranker is not None and hasattr(reranker, 'identity') and hasattr(corpus, 'directory'):
            from .rerank_cache import RerankCache
            self.cache = RerankCache(corpus.directory / 'rerank.sqlite3')
        self._reset_rerank()

    def _reset_rerank(self):
        self.scores = {}
        self.query_groups = {}
        self.retrieval_trace = []
        local = getattr(self.reranker, 'local', False)
        self.active_max_documents = None if local else self.rerank_max_documents
        self.active_max_chars = None if local else self.rerank_max_chars
        self.rerank_stats = {'requests': 0, 'documents': 0, 'document_chars': 0, 'cache_hits': 0,
                             'unscored': 0, 'max_documents': self.active_max_documents,
                             'max_chars': self.active_max_chars, 'enabled': self.reranker is not None,
                             'provider': 'local' if getattr(self.reranker, 'local', False) else 'api' if self.reranker else 'none'}

    def begin_round(self, round_number, total_rounds):
        """第一轮只使用其预算份额，后续轮次可以使用此前剩余预算。"""
        if getattr(self.reranker, 'local', False):
            return
        self.active_max_documents = self.rerank_max_documents * round_number // total_rounds
        self.active_max_chars = self.rerank_max_chars * round_number // total_rounds

    async def prepare(self, source_ids: list[str]):
        self._reset_rerank()
        await self.semantic.ready(source_ids)

    async def candidates_for(self, question: str, source_ids: list[str]) -> list[Chunk]:
        lexical = self.corpus.search(question, source_ids, limit=self.candidates)
        dense = await self.semantic.search(question, source_ids, self.candidates)
        rankings = [lexical, dense]
        if self.graph is not None:
            graph_hits = self.graph.expand(question, source_ids, limit=self.candidates)
            if graph_hits:
                rankings.append(graph_hits)
        # 各路已经各自限量，保留合并后的候选再重排。
        # 提前按融合名次截断会丢掉仅被单一路径找到的真正证据。
        candidates = rrf(rankings)
        base_candidates = list(candidates)
        # 在有界本地候选中寻找“主体＋事件”，并保留到重排阶段。
        # 不把新增候选再次截掉，否则稀有事件仍会被通用语义匹配挤走。
        seen = {chunk.id for chunk in candidates}
        additions = []
        for subject, event in keyword_pairs(question):
            hits = self.corpus.search(subject + ' ' + event, source_ids, limit=256,
                                      must_contain=(subject, event))
            hits.sort(key=lambda c: (phrase_distance(c.text, subject, event), c.id))
            selected = hits[:16]
            additions.append(selected)
            for chunk in selected:
                if chunk.id not in seen:
                    candidates.append(chunk)
                    seen.add(chunk.id)
        if self.reranker is None and additions:
            candidates = rrf([base_candidates, *additions])
        return candidates

    async def score_pool(self, question, candidates):
        """一次问答使用同一问题评分，重复段落只发送一次，预算耗尽保留免费排序。"""
        if self.reranker is None:
            return {}
        if getattr(self.reranker, 'local', False):
            return await self.score_local_pool(question, candidates)
        qhash = digest(question)
        by_hash = {digest(c.text): c for c in candidates}
        hashes = list(by_hash)
        cached = self.cache.get(self.reranker.identity, qhash, hashes) if self.cache else {}
        scores = {key: self.scores.get((qhash, key), cached.get(key)) for key in hashes}
        self.rerank_stats['cache_hits'] += sum(value is not None for value in scores.values())
        for key, value in scores.items():
            if value is not None:
                self.scores[qhash, key] = value
        selected, chars = [], 0
        for key in hashes:
            if scores[key] is not None:
                continue
            size = len(by_hash[key].text)
            if (self.rerank_stats['documents'] + len(selected) < self.active_max_documents
                    and self.rerank_stats['document_chars'] + chars + size <= self.active_max_chars):
                selected.append(key)
                chars += size
        for offset in range(0, len(selected), min(self.candidates, 32)):
            batch = selected[offset:offset + min(self.candidates, 32)]
            texts = [by_hash[key].text for key in batch]
            # 发送前扣减预算，失败或重试不能悄悄启动额外候选批次。
            self.rerank_stats['requests'] += 1
            self.rerank_stats['documents'] += len(batch)
            self.rerank_stats['document_chars'] += sum(map(len, texts))
            values = await self.reranker.score(question, texts)
            if len(values) != len(batch) or any(type(v) not in {int, float} or not math.isfinite(v) for v in values):
                raise ValueError('重排结果不完整或无效')
            for key, value in zip(batch, values):
                scores[key] = value
                self.scores[qhash, key] = value
            if self.cache:
                self.cache.put(self.reranker.identity, qhash, {key: scores[key] for key in batch})
        self.rerank_stats['unscored'] += sum(value is None for value in scores.values())
        emit('重排预算与缓存统计', **self.rerank_stats)
        return {key: value for key, value in scores.items() if value is not None}

    async def score_local_pool(self, question, candidates):
        """覆盖全部召回候选，按当前问题的不同检索表述多遍分批评分。"""
        by_hash = {digest(c.text): c for c in candidates}
        views = {key: [question] for key in by_hash}
        # 每段最多参与配置的遍数；补充表述始终保留当前问题及其约束。
        for query, group in self.query_groups.items():
            if query == question:
                continue
            view = question + "\n检索关键词（仅辅助定位，不是事实）：" + query
            for chunk in group:
                key = digest(chunk.text)
                if key in views and len(views[key]) < self.local_passes and view not in views[key]:
                    views[key].append(view)
        window_count = getattr(self.reranker, 'window_count', lambda text: 1)
        batches, aggregate, reused = [], {}, 0
        passes = max(map(len, views.values()), default=0)
        for pass_index in range(passes):
            groups = {}
            for key, questions in views.items():
                if pass_index < len(questions):
                    groups.setdefault(questions[pass_index], []).append(key)
            for view, keys in groups.items():
                qhash = digest(view)
                cached = self.cache.get(self.reranker.identity, qhash, keys) if self.cache else {}
                fresh = []
                for key in keys:
                    value = self.scores.get((qhash, key), cached.get(key))
                    if value is None:
                        fresh.append(key)
                    else:
                        self.scores[qhash, key] = value
                        aggregate[key] = max(aggregate.get(key, value), value)
                        reused += 1
                for offset in range(0, len(fresh), self.local_batch_size):
                    batches.append((pass_index + 1, view, fresh[offset:offset + self.local_batch_size]))
        total = sum(window_count(by_hash[key].text) for _, _, keys in batches for key in keys)
        tracker = ProgressTracker(total, unit="窗口")
        completed = 0
        stats = self.rerank_stats
        stats.update(max_documents=None, max_chars=None, device=getattr(self.reranker, 'device', 'cpu'), configured_passes=self.local_passes,
                     passes=passes, candidates=len(by_hash), unscored=0)
        stats['cache_hits'] += reused
        for batch_index, (pass_number, view, keys) in enumerate(batches, 1):
            check_stopping()
            texts = [by_hash[key].text for key in keys]
            def update_windows(count=0, extra=0):
                nonlocal completed
                completed += count
                tracker.total += extra
                stats['windows'] = stats.get('windows', 0) + count
                notify(tracker.update(
                    f"本地重排（{stats['device'].upper()}）· 第 {pass_number}/{passes} 遍 · 批次 {batch_index}/{len(batches)} · 窗口 {completed}/{tracker.total}",
                    current=completed, stage="本地分批分窗重排"))
            update_windows()
            stats['requests'] += 1
            stats['documents'] += len(keys)
            stats['document_chars'] += sum(map(len, texts))
            has_window_callback = hasattr(self.reranker, 'progress_callback')
            if has_window_callback:
                self.reranker.progress_callback = update_windows
            try:
                values = await self.reranker.score(view, texts)
            finally:
                if has_window_callback:
                    self.reranker.progress_callback = None
            if len(values) != len(keys) or any(type(v) not in {int, float} or not math.isfinite(v) for v in values):
                raise ValueError('本地重排结果不完整或无效')
            if not has_window_callback:
                update_windows(sum(window_count(text) for text in texts))
            qhash = digest(view)
            for key, value in zip(keys, values):
                self.scores[qhash, key] = value
                aggregate[key] = max(aggregate.get(key, value), value)
            if self.cache:
                self.cache.put(self.reranker.identity, qhash, dict(zip(keys, values)))
        stats.update(scored_candidates=len(aggregate), unscored=len(by_hash) - len(aggregate),
                     candidate_coverage_complete=len(aggregate) == len(by_hash))
        emit('本地多遍重排覆盖统计', **stats)
        return aggregate

    @staticmethod
    def select_ranked(candidates, scores, limit):
        if not scores:
            return candidates[:limit]
        ranked = sorted((c for c in candidates if digest(c.text) in scores),
                        key=lambda c: (-scores[digest(c.text)], c.id))
        unscored = [c for c in candidates if digest(c.text) not in scores]
        if limit == 1:
            return ranked[:1] or candidates[:1]
        # 每条查询先交出重排首位，让上下文轮流选取时优先获得真正评分结果。
        # 随后保留召回靠前的候选，防止小模型完全替换关键词与向量证据。
        protected = candidates[:min(2, max(1, limit // 3))]
        reserve = min(len(unscored), max(1, limit // 3)) if unscored else 0
        selected, seen = [], set()
        for chunk in [*ranked[:1], *protected, *ranked[1:max(1, limit - len(protected) - reserve)], *unscored, *ranked]:
            if chunk.id not in seen:
                selected.append(chunk)
                seen.add(chunk.id)
                if len(selected) == limit:
                    break
        return selected

    async def search(self, question: str, source_ids: list[str], *, limit: int) -> list[Chunk]:
        candidates = await self.candidates_for(question, source_ids)
        scores = await self.score_pool(question, candidates)
        return self.select_ranked(candidates, scores, limit)

    async def search_many(self, question, queries, source_ids, *, limit):
        """各查询先召回，合并去重后统一评分，评分结果供所有查询及下一轮复用。"""
        for query in queries:
            self.query_groups[query] = await self.candidates_for(query, source_ids)
        groups = list(self.query_groups.values())
        # 各查询按名次轮流获得评分机会，长候选列表不能耗尽其他查询的预算。
        # 保留旧轮次召回，后续预算仍可评分此前没来得及处理的段落。
        pool, seen = [], set()
        for rank in range(max((len(group) for group in groups), default=0)):
            for group in groups:
                if rank < len(group) and group[rank].id not in seen:
                    chunk = group[rank]
                    pool.append(chunk)
                    seen.add(chunk.id)
        scores = await self.score_pool(question, pool)
        results = [self.select_ranked(group, scores, limit) for group in groups]
        self.retrieval_trace.append({'groups': [
            {'query_index': index, 'candidate_ids': [c.id for c in group],
             'selected_ids': [c.id for c in hits]} for index, (group, hits) in enumerate(zip(groups, results))],
            'scored_ids': [c.id for c in pool if digest(c.text) in scores],
            'scores': {c.id: scores[digest(c.text)] for c in pool if digest(c.text) in scores},
            'active_max_documents': self.active_max_documents, 'active_max_chars': self.active_max_chars})
        emit('检索候选评分覆盖', groups=len(groups), candidates=len(pool),
             scored=len(self.retrieval_trace[-1]['scored_ids']), selected=sum(map(len, results)))
        return results

    async def close(self):
        await self.semantic.store.close()
        await self.semantic.embedder.close()
        if self.reranker is not None:
            await self.reranker.close()


def required(config: dict, name: str, fallback: str = "") -> str:
    value = (config.get(name) or fallback).strip()
    if not value or value.startswith("YOUR_"):
        raise ValueError(f"请在本项目 .env 配置 {name}")
    return value


def make_retriever(corpus: Corpus, *, for_index: bool = False, mode: str | None = None):
    from .config import get_config
    config = get_config()
    mode = mode or config.get("RETRIEVAL_MODE", "hybrid")
    if mode not in {"hybrid", "lexical"}:
        raise ValueError("RETRIEVAL_MODE 必须为 hybrid 或 lexical")
    if not for_index and mode == "lexical":
        return None
    model = required(config, "EMBEDDING_MODEL_ID")
    prefix = config.get("EMBEDDING_QUERY_PREFIX", "")
    if config.get("EMBEDDING_PROVIDER", "api") == "local":
        from .local_reranker import resolve_device
        embedder = LocalEmbedder(model, device=resolve_device(config.get("LOCAL_DEVICE") or "auto"), query_prefix=prefix)
    else:
        dimensions = config.get("EMBEDDING_DIMENSIONS", "").strip()
        embedder = APIEmbedder(api_key=required(config, "EMBEDDING_API_KEY", config.get("LLM_API_KEY", "")),
                              model=model, base_url=required(config, "EMBEDDING_BASE_URL", config.get("LLM_BASE_URL", "")),
                              query_prefix=prefix, dimensions=int(dimensions) if dimensions else None)
    reranker = None
    if not for_index:
        provider = config.get("RERANK_PROVIDER", "api")
        if provider == "local":
            from .config import PROJECT
            from .local_models import MODEL_ID
            local_model = config.get("LOCAL_RERANK_MODEL_ID") or config.get("RERANK_MODEL_ID") or MODEL_ID
            if local_model == MODEL_ID:
                from .local_reranker import QwenLocalReranker
                reranker = QwenLocalReranker(PROJECT / "models", device=config.get("LOCAL_DEVICE") or "auto",
                    window_chars=int(config.get("LOCAL_RERANK_WINDOW_CHARS", "").strip() or "1200"))
            else:
                from .local_reranker import resolve_device
                reranker = LocalReranker(local_model, device=resolve_device(config.get("LOCAL_DEVICE") or "auto"))
        elif provider == "api":
            reranker = APIReranker(api_key=required(config, "RERANK_API_KEY", config.get("LLM_API_KEY", "")),
                                  model=required(config, "RERANK_MODEL_ID"), url=required(config, "RERANK_URL"))
        elif provider != "none":
            raise ValueError("RERANK_PROVIDER 必须为 api、local 或 none")
    store = QdrantStore(corpus, embedder.identity, url=config.get("QDRANT_URL") or None,
                        api_key=config.get("QDRANT_API_KEY") or None)
    semantic = SemanticIndex(corpus, embedder, store)
    from .graph import EvidenceGraph
    return HybridRetriever(corpus, semantic, reranker, graph=EvidenceGraph(corpus),
                           local_passes=int(config.get('LOCAL_RERANK_PASSES', '').strip() or '2'),
                           local_batch_size=int(config.get('LOCAL_RERANK_BATCH_SIZE', '').strip() or '8'),
                           rerank_max_documents=int(config.get('RERANK_MAX_DOCUMENTS', '').strip() or '128'),
                           rerank_max_chars=int(config.get('RERANK_MAX_CHARS', '').strip() or '200000'))
