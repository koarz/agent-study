"""所有关系绑定原文引用；图谱用于定位原文，不能替代原文作为证据。"""

from __future__ import annotations

import hashlib
import json

from . import prompts, schemas
from .corpus import Corpus
from .llm import ModelOutputError
from .logging_utils import emit
import logging
from .scan import FullScanner


GRAPH_EXTRACT = prompts.BOUNDARY + """
逐批从原文提取直接有依据的实体关系和事件，建立证据图谱。
返回 {"status":"answered|unclear|not_found","facts":[
 {"id":"f1","subject":"原文里的实体名称","predicate":"直接有依据的关系或行为",
  "object":"原文里的另一个实体名称","time_text":"原文明确的时间措辞或空字符串",
  "attribution":"narration|dialogue|rumor|dream|hypothesis|unclear",
  "certainty":"explicit|unclear",
  "citations":[{"chunk_id":"段落id","quote":"逐字连续原文"}]}
]}
每批最多 12 条，不能完整提取时 status=unclear。没有关系事实则 not_found 且 facts=[]。
subject/object 必须在引用中按原文写出完整名称，不使用代词，不推测别名对应，不捏造关系。
引用必须同时覆盖实体名称和关系依据。只有“他”“她”等代词的句子不够，须按原文连续引用给出姓名的上下文。
“我、你、他、她、我们、他们、自己、本人、主角”等不能作为图谱实体名。多个别名分别记录，不能拼接成原文不存在的名称。
只在原文明示身份对应时提取“别名”关系，不能自动把同姓人物、称呼、职务合并成同一个人。
时间不明确时 time_text=""；保留“昨日”“三年前”等原文说法，不自行换算日期或事件时间。
台词、传闻、梦境、假设必须保留 attribution 与 certainty，不得当成叙述事实。
一条关系发生变化或双方说法冲突时分别记录，不覆盖先前状态，也不推测原因。
这是局部证据提取，不能断言全文所有关系已经覆盖。
"""


def graph_draft(raw: dict) -> dict:
    facts = raw.get("facts")
    if not isinstance(facts, list) or len(facts) > 12:
        raise ModelOutputError("图谱提取结果格式无效")
    claims = []
    for fact in facts:
        if not isinstance(fact, dict):
            raise ModelOutputError("图谱关系格式无效")
        fields = {}
        for key in ("subject", "predicate", "object", "time_text", "attribution", "certainty"):
            value = fact.get(key)
            if not isinstance(value, str) or len(value) > 120 or (key != "time_text" and not value.strip()):
                raise ModelOutputError("图谱关系字段无效")
            fields[key] = value
        if fields["attribution"] not in {"narration", "dialogue", "rumor", "dream", "hypothesis", "unclear"}:
            raise ModelOutputError("图谱叙述归属无效")
        if fields["certainty"] not in {"explicit", "unclear"}:
            raise ModelOutputError("图谱确定性标注无效")
        if fields["subject"] in {"我", "你", "他", "她", "它", "我们", "他们", "她们", "你们", "自己", "本人", "主角"} or fields["object"] in {"我", "你", "他", "她", "它", "我们", "他们", "她们", "你们", "自己", "本人", "主角"}:
            raise ModelOutputError("图谱实体必须使用原文明示的名称，不能将代词合并为人物")
        citations = fact.get("citations")
        if not isinstance(citations, list) or any(not isinstance(c, dict) or not isinstance(c.get("quote"), str) for c in citations):
            raise ModelOutputError("图谱关系没有有效引用")
        quotes = "\n".join(c["quote"] for c in citations)
        if fields["subject"] not in quotes or fields["object"] not in quotes:
            raise ModelOutputError("图谱实体必须在原文引用中出现")
        if fields["time_text"] and fields["time_text"] not in quotes:
            raise ModelOutputError("图谱时间必须保留原文措辞")
        # 将所有结构化字段放入待复核结论，避免未校验的元数据绕过复核。
        claims.append({"id": fact.get("id"), "text": json.dumps(fields, ensure_ascii=False),
                       "citations": citations})
    return {"status": raw.get("status"), "claims": claims}


def graph_batch_draft(raw: dict, chosen=None) -> dict:
    """逐条丢弃格式或实体引用不合格的候选，避免一条坏记录终止整本书。"""
    if not isinstance(raw.get("facts"), list) or len(raw["facts"]) > 12:
        raise ModelOutputError("图谱提取结果格式无效")
    claims, rejected = [], 0
    for fact in raw["facts"]:
        try:
            if chosen is not None:
                fact = expand_quote_context(fact, chosen)
            claims.extend(graph_draft({"status": "unclear", "facts": [fact]})["claims"])
        except ModelOutputError:
            rejected += 1
    if rejected:
        emit("图谱候选实体或字段未通过校验", level=logging.WARNING, rejected_claims=rejected)
    return {"status": "unclear" if rejected else raw.get("status"), "claims": claims, "pre_discarded": rejected}


def expand_quote_context(fact, chosen):
    """从同一原文段落补齐姓名所在的上下文，不改写候选关系或任何原文字符。"""
    citations = []
    for citation in fact["citations"]:
        chunk = chosen.get(citation["chunk_id"])
        quote = citation["quote"]
        if chunk is None or not quote.strip() or quote not in chunk.text:
            citations.append(citation)
            continue
        left = chunk.text.index(quote)
        right = left + len(quote)
        for name in (fact["subject"], fact["object"]):
            if name in quote or name not in chunk.text:
                continue
            # 只定位候选已给出的原文字面名称；名称是否属于该事件仍由复核判断。
            positions, offset = [], 0
            while (position := chunk.text.find(name, offset)) >= 0:
                positions.append(position)
                offset = position + max(1, len(name))
            position = min(positions, key=lambda value: abs(value - left))
            begin, end = min(left, position), max(right, position + len(name))
            if end - begin <= 800:
                left, right = begin, end
        citations.append({**citation, "quote": chunk.text[left:right]})
    return {**fact, "citations": citations}


class EvidenceGraph:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.corpus.db.executescript("""
            CREATE TABLE IF NOT EXISTS graph_facts (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
                start INTEGER NOT NULL, claim_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS graph_subject ON graph_facts(source_id,subject);
            CREATE INDEX IF NOT EXISTS graph_object ON graph_facts(source_id,object);
        """)

    def record(self, claims: list[dict]):
        with self.corpus.db:
            for claim in claims:
                fields = json.loads(claim["text"])
                key = [fields, [(c["source_id"], c["start"], c["end"], c["quote"]) for c in claim["citations"]]]
                fact_id = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                for citation in claim["citations"]:
                    self.corpus.db.execute("INSERT OR IGNORE INTO graph_facts VALUES (?,?,?,?,?,?,?)",
                        (fact_id, citation["source_id"], fields["subject"], fields["predicate"], fields["object"],
                         citation["start"], json.dumps(claim, ensure_ascii=False)))

    def facts(self, entity: str, source_ids: list[str], *, limit: int = 100) -> list[dict]:
        if not entity.strip() or not 1 <= limit <= 1000:
            raise ValueError("实体名称非空，结果上限为 1 到 1000")
        rows = self.corpus.db.execute(
            f"SELECT * FROM graph_facts WHERE source_id IN ({self.corpus._where(source_ids)}) AND (subject=? OR object=?) ORDER BY source_id,start,id LIMIT ?",
            [*source_ids, entity, entity, limit])
        results = []
        for row in rows:
            claim = json.loads(row["claim_json"])
            for citation in claim["citations"]:
                chunk = self.corpus.get_chunk(citation["chunk_id"])
                if self.corpus.citation(chunk, citation["quote"]) != citation:
                    raise ValueError("图谱引用与原文不一致")
            results.append({"fact": json.loads(claim["text"]), "citations": claim["citations"]})
        return results

    def expand(self, question: str, source_ids: list[str], *, limit: int = 40):
        rows = self.corpus.db.execute(
            f"""SELECT * FROM graph_facts WHERE source_id IN ({self.corpus._where(source_ids)})
                AND (instr(?,subject)>0 OR instr(?,object)>0) ORDER BY source_id,start,id LIMIT ?""",
            [*source_ids, question, question, limit])
        chunks, entities = {}, set()

        def collect(row):
            entities.update((row["subject"], row["object"]))
            for citation in json.loads(row["claim_json"])["citations"]:
                if citation["source_id"] in source_ids:
                    chunk = self.corpus.get_chunk(citation["chunk_id"])
                    if self.corpus.citation(chunk, citation["quote"]) != citation:
                        raise ValueError("图谱引用与原文不一致")
                    chunks[chunk.id] = chunk

        for row in rows:
            collect(row)
        # 再扩展一层，用于寻找原文明示的别名或相关实体。
        # 限制扩展规模，并始终只返回对应的原文段落。
        for entity in sorted(entities)[:20]:
            if len(chunks) >= limit:
                break
            related = self.corpus.db.execute(
                f"SELECT * FROM graph_facts WHERE source_id IN ({self.corpus._where(source_ids)}) AND (subject=? OR object=?) ORDER BY start,id LIMIT ?",
                [*source_ids, entity, entity, limit])
            for row in related:
                collect(row)
                if len(chunks) >= limit:
                    break
        return list(chunks.values())[:limit]

    async def build(self, backend, *, sources=None, verifier=None, batch_chars=6000, progress=None):
        scanner = FullScanner(self.corpus, backend, verifier=verifier, batch_chars=batch_chars,
                              instruction=GRAPH_EXTRACT, transform=graph_batch_draft, on_batch=self.record, progress=progress, schema=schemas.GRAPH)
        result = await scanner.scan("提取原文明示的实体关系、别名和事件，保留时间措辞、叙述归属及不确定性。", sources=sources)
        source_ids = self.corpus.select_sources(sources)
        result["graph_facts"] = self.corpus.db.execute(
            f"SELECT count(*) FROM graph_facts WHERE source_id IN ({self.corpus._where(source_ids)})", source_ids).fetchone()[0]
        return result
