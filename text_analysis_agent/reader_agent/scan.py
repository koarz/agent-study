"""支持恢复进度的全文遍历；报告保留所有通过复核的局部发现。

    记录实际遍历覆盖率，但不声称模型提取绝无遗漏。
    模型生成的摘要不能替代原文证据。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import logging
from pathlib import Path

from . import prompts, schemas
from .agent import ReadingAgent
from .corpus import Corpus
from .llm import Backend, ModelOutputError, request_json
from .logging_utils import emit
from .progress import ProgressTracker


EXTRACT = prompts.ANSWER + """
当前任务是逐批扫描长篇小说。你只看到其中一批原文，不能据此给出全文结论。
按原文出现顺序提取与 question 直接有关的局部事实、事件、明确的关系、明确的矛盾，
每条保留原文引用，不依赖其他批次，不从用户问题推断事实。
先为局部事实建立证据目录，不作全文总结。没有相关证据可以返回 not_found。
本批相关事实多于 12 条或无法完整提取时，status 必须为 unclear，不能声称全部。
"""


def batches(corpus: Corpus, source_ids: list[str], budget: int):
    current, used = [], 0
    for chunk in corpus.iter_chunks(source_ids):
        if len(chunk.text) > budget:
            raise ValueError("扫描预算小于已有原文分块，请增大 --batch-chars")
        if current and used + len(chunk.text) > budget:
            yield current
            current, used = [], 0
        current.append(chunk)
        used += len(chunk.text)
    if current:
        yield current


def scan_plan(corpus: Corpus, source_ids: list[str], budget: int) -> dict:
    if budget < 1000:
        raise ValueError("扫描批次预算至少 1000 字")
    # 规划阶段只查询段落长度，不将原文内容载入内存。
    rows = corpus.db.execute(
        f"SELECT length(c.text) AS size FROM chunks c JOIN sources s ON s.id=c.source_id WHERE source_id IN ({corpus._where(source_ids)}) ORDER BY s.name,c.ordinal",
        source_ids,
    )
    count = chars = batch_count = used = 0
    for row in rows:
        size = row["size"]
        if size > budget:
            raise ValueError("扫描预算小于已有原文分块，请增大 --batch-chars")
        if used and used + size > budget:
            batch_count += 1
            used = 0
        used += size
        chars += size
        count += 1
    if used:
        batch_count += 1
    return {"total_chunks": count, "total_batches": batch_count, "input_chars": chars,
            "batch_chars": budget, "model_calls_min": batch_count, "model_calls_max": batch_count * 20,
            "model_calls_without_api_retry_max": batch_count * 4,
            "model_calls_without_format_retry_max": batch_count * 2}


class FullScanner:
    def __init__(self, corpus: Corpus, backend: Backend, *, batch_chars: int = 12000,
                 progress=None, verifier: Backend | None = None, instruction: str = EXTRACT,
                 transform=None, on_batch=None, schema=None):
        self.corpus = corpus
        self.backend = backend
        self.batch_chars = batch_chars
        self.progress = progress or (lambda message: None)
        self.verifier = verifier or backend
        self.instruction = instruction
        self.schema = schema or schemas.ANSWER
        self.transform = transform or (lambda draft, chosen: draft)
        self.on_batch = on_batch or (lambda claims: None)
        self.validator = ReadingAgent(corpus, backend)

    def _validate_batch(self, draft, chosen):
        claims, seen = [], set()
        rejected = draft.get("pre_discarded", 0)
        for candidate in draft["claims"]:
            # 引文不连续、段落不属于本批或 ID 重复时，仅拒绝该候选。
            # 快照损坏等存储错误仍直接抛出，不能用跳过候选掩盖。
            references = candidate.get("citations", [])
            bad_quote = any(not isinstance(c.get("quote"), str) or not c["quote"].strip() or
                            c.get("chunk_id") not in chosen or c["quote"] not in chosen[c["chunk_id"]].text
                            for c in references)
            if bad_quote or candidate.get("id") in seen:
                rejected += 1
                continue
            try:
                checked = self.validator._validate_draft({"status": "unclear", "claims": [candidate]}, chosen)
            except ModelOutputError:
                rejected += 1
                continue
            seen.add(candidate["id"])
            claims.extend(checked)
        if rejected:
            draft["status"] = "unclear"
            emit("扫描候选结论未通过引用校验", level=logging.WARNING, rejected_claims=rejected)
        else:
            self.validator._validate_draft(draft, chosen)
        return claims, rejected

    async def scan(self, question: str, *, sources: list[str] | None = None) -> dict:
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("问题须为 1 到 4000 字的非空文本")
        self.corpus.begin_read()
        source_ids = self.corpus.select_sources(sources)
        source_meta = [s for s in self.corpus.sources() if s["id"] in source_ids]
        plan = scan_plan(self.corpus, source_ids, self.batch_chars)
        tracker = ProgressTracker(plan["total_batches"], unit="批")
        def fingerprint(backend):
            # 回答或复核模型变化后重新核对，避免沿用另一模型的历史结论。
            metadata = {"model": getattr(backend, "model", None),
                        "temperature": getattr(backend, "temperature", None),
                        "endpoint": str(getattr(getattr(backend, "client", None), "base_url", "")),
                        "adapter": type(backend).__qualname__, "json_mode": getattr(backend, "json_mode", None)}
            return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        config = {"question": question, "sources": source_meta, "batch_chars": self.batch_chars,
                  "prompts_sha256": hashlib.sha256((self.instruction + prompts.VERIFY + json.dumps(self.schema, sort_keys=True) + "原文上下文补齐-v1").encode()).hexdigest(),
                  "backend_fingerprint": fingerprint(self.backend),
                  "verifier_fingerprint": fingerprint(self.verifier)}
        encoded_config = json.dumps(config, ensure_ascii=False, sort_keys=True)
        run_id = hashlib.sha256(encoded_config.encode()).hexdigest()
        directory = self.corpus.directory / "analyses" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = sqlite3.connect(directory / "progress.sqlite3")
        checkpoint.row_factory = sqlite3.Row
        checkpoint.executescript("""
            CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS batches (
                ordinal INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL,
                chunks INTEGER NOT NULL, status TEXT NOT NULL, discarded INTEGER NOT NULL,
                claims_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS findings (
                id TEXT PRIMARY KEY, batch INTEGER NOT NULL, claim_json TEXT NOT NULL
            );
        """)
        existing = checkpoint.execute("SELECT json FROM config WHERE id=1").fetchone()
        if existing and existing["json"] != encoded_config:
            checkpoint.close()
            raise ValueError("扫描配置与进度记录不一致，停止恢复")
        with checkpoint:
            checkpoint.execute("INSERT OR IGNORE INTO config VALUES (1,?)", (encoded_config,))
        complete = False
        reused = 0
        try:
            for index, batch in enumerate(batches(self.corpus, source_ids, self.batch_chars)):
                chosen = {chunk.id: chunk for chunk in batch}
                evidence = [chunk.to_dict() for chunk in batch]
                fingerprint = hashlib.sha256(json.dumps(evidence, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                cached = checkpoint.execute("SELECT * FROM batches WHERE ordinal=?", (index,)).fetchone()
                if cached:
                    if cached["fingerprint"] != fingerprint:
                        raise ValueError("已扫描批次与现有原文不一致，停止恢复")
                    # 恢复进度时仍重新检查已保存引用是否与原文一致。
                    restored = self.validator._validate_draft(
                        {"status": "unclear", "claims": json.loads(cached["claims_json"])}, chosen)
                    self.on_batch(restored)
                    reused += 1
                    self.progress(tracker.update(f"扫描 {index + 1}/{plan['total_batches']} 批：已恢复，跳过模型调用",
                                                current=index + 1, reused=reused, stage="恢复扫描进度"))
                    continue
                coverage = {"mode": "retrieved", "scan_batch": index + 1,
                            "total_batches": plan["total_batches"], "sources": source_meta,
                            "total_chunks": plan["total_chunks"], "selected_chunks": len(batch)}
                payload = {"question": question, "coverage": coverage, "evidence": evidence}
                self.progress(tracker.update(f"扫描 {index + 1}/{plan['total_batches']} 批：提取并核对原文",
                                            current=index, reused=reused, stage="提取与核对原文"))
                draft = self.transform(await request_json(self.backend, self.instruction, payload, schema=self.schema), chosen)
                claims, invalid = self._validate_batch(draft, chosen)
                accepted = []
                if claims:
                    review = await request_json(self.verifier, prompts.VERIFY, {**payload, "claims": claims}, schema=schemas.VERIFY)
                    accepted, _ = self.validator._validate_verification(review, claims)
                emit("全文扫描批次复核完成", batch=index + 1, total_batches=plan["total_batches"],
                     accepted_claims=len(accepted), rejected_claims=invalid + len(claims) - len(accepted))
                with checkpoint:
                    checkpoint.execute("INSERT INTO batches VALUES (?,?,?,?,?,?)",
                                       (index, fingerprint, len(batch), draft["status"], invalid + len(claims) - len(accepted),
                                        json.dumps(accepted, ensure_ascii=False)))
                    for claim in accepted:
                        # 根据原文位置去除重叠分块产生的重复记录。
                        # 不同断言分别保留，不自动消解原文中的矛盾。
                        key = [claim["text"], sorted((c["source_id"], c["start"], c["end"], c["quote"])
                                                    for c in claim["citations"])]
                        finding_id = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()
                        checkpoint.execute("INSERT OR IGNORE INTO findings VALUES (?,?,?)",
                                           (finding_id, index, json.dumps(claim, ensure_ascii=False)))
                self.on_batch(accepted)
                self.progress(tracker.update(f"已完成 {index + 1}/{plan['total_batches']} 批：保留 {len(accepted)} 条，拒绝 {invalid + len(claims) - len(accepted)} 条",
                                            current=index + 1, reused=reused, stage="保存证据与更新进度"))
            processed = checkpoint.execute("SELECT coalesce(sum(chunks),0) FROM batches").fetchone()[0]
            completed_batches = checkpoint.execute("SELECT count(*) FROM batches").fetchone()[0]
            if processed != plan["total_chunks"] or completed_batches != plan["total_batches"]:
                raise ValueError("扫描覆盖数量不一致，不能标记全文扫描完成")
            complete = True
        finally:
            try:
                result = self._export(checkpoint, directory, config, plan, complete, reused)
                self.progress(f"原文证据报告：{result['report']}")
            finally:
                checkpoint.close()
        return result

    @staticmethod
    def _export(checkpoint, directory: Path, config: dict, plan: dict, complete: bool, reused: int) -> dict:
        count = checkpoint.execute("SELECT count(*) FROM findings").fetchone()[0]
        processed = checkpoint.execute("SELECT coalesce(sum(chunks),0) FROM batches").fetchone()[0]
        issues = checkpoint.execute("SELECT count(*) FROM batches WHERE status='unclear' OR discarded>0").fetchone()[0]
        discarded = checkpoint.execute("SELECT coalesce(sum(discarded),0) FROM batches").fetchone()[0]
        result = {"question": config["question"], "status": "scan_complete" if complete else "scan_partial",
                  "claims": [], "notice": "全文逐批扫描完成；报告保留通过核对的局部证据，模型提取仍可能漏检。" if complete else
                  "扫描尚未完成，报告仅覆盖已保存的批次；再次执行相同命令可继续。",
                  "coverage": {"mode": "full_scan" if complete else "partial_scan",
                               "sources": config["sources"], "total_chunks": plan["total_chunks"],
                               "selected_chunks": processed, "total_batches": plan["total_batches"],
                               "semantic_completeness": "not_guaranteed"},
                  "finding_count": count, "unclear_batches": issues, "discarded_claims": discarded,
                  "resumed_batches": reused, "report": str(directory / "report.md"),
                  "evidence_file": str(directory / "evidence.jsonl"), "plan": plan}
        report_tmp = directory / "report.md.tmp"
        evidence_tmp = directory / "evidence.jsonl.tmp"
        with report_tmp.open("w", encoding="utf-8") as report, evidence_tmp.open("w", encoding="utf-8") as evidence:
            report.write(f"# 原文证据扫描报告\n\n问题：{config['question']}\n\n{result['notice']}\n\n")
            report.write(f"扫描覆盖：{processed}/{plan['total_chunks']} 个段落；通过核对的记录：{count} 条。\n\n")
            report.write("下列记录按扫描顺序列出，保留每批的局部发现。跨批矛盾不自动裁决，记录不等同于完整人物档案或全文总结。\n\n")
            report.write(f"存在不明确或拒绝结论的批次：{issues}；被拒绝的结论：{discarded}。\n\n")
            for number, row in enumerate(checkpoint.execute("SELECT * FROM findings ORDER BY batch,rowid"), 1):
                claim = json.loads(row["claim_json"])
                evidence.write(json.dumps({"record": number, "batch": row["batch"] + 1, **claim}, ensure_ascii=False) + "\n")
                report.write(f"## {number}. 第 {row['batch'] + 1} 批\n\n{claim['text']}\n\n")
                for citation in claim["citations"]:
                    report.write(f"[{citation['source']} · {citation['chapter']} · 第 {citation['line_start']}—{citation['line_end']} 行]\n\n")
                    for line in citation["quote"].splitlines():
                        report.write("> " + line + "\n")
                    report.write("\n")
        report_tmp.replace(directory / "report.md")
        evidence_tmp.replace(directory / "evidence.jsonl")
        (directory / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
