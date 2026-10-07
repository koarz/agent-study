"""为前端提供持久化个人书库和后台分析任务。"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

from .agent import ReadingAgent
from .config import get_config
from .corpus import Corpus
from .graph import EvidenceGraph
from .llm import CompatibleBackend, ModelOutputError
from .retrieval import make_retriever, configured_embedding_identity
from .scan import FullScanner
from .logging_utils import Observability, request_id as current_request_id
from .progress import ProgressUpdate
from .api_calls import APIServiceError, api_context, quota_exhausted


def now():
    return datetime.now(timezone.utc).isoformat()


class ConversationBusyError(ValueError):
    """会话仍有未结束的问答，暂时不能删除其任务记录。"""


class Library:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        corpus = Corpus(self.directory, create=True)
        corpus.close()
        self.path = self.directory / "library.sqlite3"
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS books (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, book_id TEXT, kind TEXT NOT NULL,
                    payload TEXT NOT NULL, status TEXT NOT NULL, progress TEXT NOT NULL,
                    result TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS book_jobs ON jobs(book_id,created_at);
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, book_id TEXT NOT NULL, title TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS book_conversations ON conversations(book_id,updated_at);
            """)
            if "progress_detail" not in {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}:
                db.execute("ALTER TABLE jobs ADD COLUMN progress_detail TEXT NOT NULL DEFAULT '{}'")
            if "conversation_id" not in {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}:
                db.execute("ALTER TABLE jobs ADD COLUMN conversation_id TEXT")
            if "auto_recovery" not in {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}:
                db.execute("ALTER TABLE jobs ADD COLUMN auto_recovery TEXT NOT NULL DEFAULT '{}'")
            db.execute("CREATE INDEX IF NOT EXISTS conversation_jobs ON jobs(conversation_id,created_at)")
            # 每本书的旧问答迁入独立历史会话，保留任务 ID、状态及全部结果。
            for old in db.execute("SELECT book_id,min(created_at) AS first,max(updated_at) AS last FROM jobs WHERE kind='ask' AND conversation_id IS NULL AND book_id IS NOT NULL GROUP BY book_id").fetchall():
                conversation_id = uuid.uuid4().hex
                db.execute("INSERT INTO conversations VALUES (?,?,?,?,?)", (conversation_id, old['book_id'], '历史会话', old['first'], old['last']))
                db.execute("UPDATE jobs SET conversation_id=? WHERE kind='ask' AND book_id=? AND conversation_id IS NULL", (conversation_id, old['book_id']))
        self.sync()

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def sync(self):
        corpus = Corpus(self.directory)
        try:
            sources = corpus.sources()
        finally:
            corpus.close()
        with self.connection() as db:
            db.executemany("INSERT OR IGNORE INTO books(id,title,created_at) VALUES (?,?,?)",
                           [(s["id"], Path(s["name"]).stem, now()) for s in sources])

    def books(self, *, archived: bool = False) -> list[dict]:
        self.sync()
        corpus = Corpus(self.directory)
        try:
            sources = {s["id"]: s for s in corpus.sources()}
            indexes = self.index_metadata(corpus, sources)
        finally:
            corpus.close()
        with self.connection() as db:
            rows = db.execute("SELECT * FROM books WHERE archived=? ORDER BY created_at DESC,id", (int(archived),))
            return [{**dict(row), **sources[row["id"]], "title": row["title"],
                     **indexes[row["id"]]}
                    for row in rows if row["id"] in sources]

    def index_metadata(self, corpus, sources):
        """书库和书籍详情共用索引状态，不能把局部关系记录当成全书处理完成。"""
        if not sources:
            return {}
        ids = list(sources)
        placeholders = corpus._where(ids)
        tables = {row[0] for row in corpus.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        try:
            embedding_id = configured_embedding_identity(get_config())
        except ValueError:
            embedding_id = ""
        # 更换嵌入模型后，旧模型的索引不能继续显示为已建立。
        vector_counts = dict(corpus.db.execute(
            f"SELECT c.source_id,count(DISTINCT v.chunk_id) FROM vector_progress v JOIN chunks c ON c.id=v.chunk_id WHERE v.model=? AND c.source_id IN ({placeholders}) GROUP BY c.source_id",
            [embedding_id, *ids])) if "vector_progress" in tables else {}
        graph_counts = dict(corpus.db.execute(
            f"SELECT source_id,count(*) FROM graph_facts WHERE source_id IN ({placeholders}) GROUP BY source_id", ids)) if "graph_facts" in tables else {}
        active, graph_ready = set(), set()
        with self.connection() as db:
            for job in db.execute(
                f"SELECT book_id,kind,status,result FROM jobs WHERE book_id IN ({placeholders}) AND kind IN ('index','graph') AND status IN ('queued','running','completed')", ids):
                if job["status"] in {"queued", "running"}:
                    active.add((job["book_id"], job["kind"]))
                elif job["kind"] == "graph" and job["result"]:
                    result = json.loads(job["result"])
                    scope, source = result.get("coverage", {}), sources[job["book_id"]]
                    if (result.get("status") == "scan_complete" and scope.get("mode") == "full_scan"
                            and scope.get("selected_chunks") == scope.get("total_chunks") == source["chunks"]
                            and any(s.get("id") == source["id"] and s.get("sha256") == source["sha256"]
                                    for s in scope.get("sources", []))):
                        graph_ready.add(job["book_id"])
        metadata = {}
        for source_id, source in sources.items():
            vector_count, graph_count = vector_counts.get(source_id, 0), graph_counts.get(source_id, 0)
            def status(kind, ready, partial):
                return "ready" if ready else "building" if (source_id, kind) in active else "partial" if partial else "missing"
            metadata[source_id] = {"vector_chunks": vector_count, "graph_facts": graph_count, "index_status": {
                "index": status("index", source["chunks"] > 0 and vector_count >= source["chunks"], vector_count > 0),
                "graph": status("graph", source_id in graph_ready, graph_count > 0)}}
        return metadata

    def book(self, book_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
        if row is None:
            raise KeyError("书籍不存在")
        corpus = Corpus(self.directory)
        try:
            source = next((s for s in corpus.sources() if s["id"] == book_id), None)
            indexes = self.index_metadata(corpus, {book_id: source}) if source is not None else {}
        finally:
            corpus.close()
        if source is None:
            raise KeyError("原文不存在")
        return {**dict(row), **source, "title": row["title"], **indexes[book_id]}

    def update_book(self, book_id: str, *, title: str | None = None, archived: bool | None = None):
        self.book(book_id)
        with self.connection() as db:
            if title is not None:
                if not title.strip() or len(title) > 200:
                    raise ValueError("书名须为 1 到 200 字")
                db.execute("UPDATE books SET title=? WHERE id=?", (title.strip(), book_id))
            if archived is not None:
                db.execute("UPDATE books SET archived=? WHERE id=?", (int(archived), book_id))
        return self.book(book_id)

    def create_job(self, kind: str, payload: dict, book_id: str | None = None):
        payload = dict(payload)
        conversation_id = None
        if kind == "ask" and book_id is not None:
            conversation_id = payload.get("conversation_id") or self.default_conversation(book_id)["id"]
            payload["conversation_id"] = conversation_id
        job_id = uuid.uuid4().hex
        stamp = now()
        with self.connection() as db:
            # 会话检查与任务写入使用同一事务，避免删除后又写入孤立问答。
            db.execute("BEGIN IMMEDIATE")
            if conversation_id is not None:
                conversation = db.execute("SELECT book_id FROM conversations WHERE id=?", (conversation_id,)).fetchone()
                if conversation is None:
                    raise KeyError("会话不存在")
                if conversation['book_id'] != book_id:
                    raise ValueError("会话不属于当前书籍")
            db.execute("INSERT INTO jobs(id,book_id,kind,payload,status,progress,result,error,created_at,updated_at,conversation_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (job_id, book_id, kind, json.dumps(payload, ensure_ascii=False), "queued", "等待执行", None, None, stamp, stamp, conversation_id))
            if conversation_id is not None:
                db.execute("UPDATE conversations SET updated_at=?, title=CASE WHEN title='新会话' AND (SELECT count(*) FROM jobs WHERE conversation_id=?)=1 THEN ? ELSE title END WHERE id=?",
                           (stamp, conversation_id, payload.get('question', '').strip()[:32] or '新会话', conversation_id))
        return self.job(job_id)

    def conversations(self, book_id):
        self.book(book_id)
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT c.*,count(j.id) AS message_count FROM conversations c LEFT JOIN jobs j ON j.conversation_id=c.id AND j.kind='ask' WHERE c.book_id=? GROUP BY c.id ORDER BY c.updated_at DESC,c.id", (book_id,))]

    def conversation(self, conversation_id, *, book_id=None):
        with self.connection() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        if row is None:
            raise KeyError("会话不存在")
        if book_id is not None and row['book_id'] != book_id:
            raise ValueError("会话不属于当前书籍")
        return dict(row)

    def create_conversation(self, book_id, *, title='新会话'):
        self.book(book_id)
        if not isinstance(title, str) or not title.strip() or len(title) > 100:
            raise ValueError("会话名称须为 1 到 100 字")
        conversation_id, stamp = uuid.uuid4().hex, now()
        with self.connection() as db:
            db.execute("INSERT INTO conversations VALUES (?,?,?,?,?)", (conversation_id, book_id, title.strip(), stamp, stamp))
        return self.conversation(conversation_id)

    def default_conversation(self, book_id):
        self.book(book_id)
        with self.connection() as db:
            # 保证旧客户端同时提交时只创建一个默认会话，不混入其他书籍。
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM conversations WHERE book_id=? ORDER BY created_at,id LIMIT 1", (book_id,)).fetchone()
            if row is not None:
                return dict(row)
            conversation_id, stamp = uuid.uuid4().hex, now()
            db.execute("INSERT INTO conversations VALUES (?,?,?,?,?)", (conversation_id, book_id, '新会话', stamp, stamp))
        return self.conversation(conversation_id)

    def rename_conversation(self, conversation_id, title):
        self.conversation(conversation_id)
        if not isinstance(title, str) or not title.strip() or len(title) > 100:
            raise ValueError("会话名称须为 1 到 100 字")
        with self.connection() as db:
            db.execute("UPDATE conversations SET title=?,updated_at=? WHERE id=?", (title.strip(), now(), conversation_id))
        return self.conversation(conversation_id)

    def delete_conversation(self, conversation_id, *, active_job_ids=()):
        """原子删除会话及问答记录，保留书籍原文、索引和独立的用量统计。"""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT id FROM conversations WHERE id=?", (conversation_id,)).fetchone() is None:
                raise KeyError("会话不存在")
            jobs = db.execute("SELECT id,status FROM jobs WHERE conversation_id=?", (conversation_id,)).fetchall()
            if any(row['status'] in {'queued', 'running'} or row['id'] in active_job_ids for row in jobs):
                raise ConversationBusyError("此会话还有问答正在执行或排队，请等任务结束后再删除")
            db.execute("DELETE FROM jobs WHERE conversation_id=?", (conversation_id,))
            db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        return {'deleted': True, 'conversation_id': conversation_id,
                'deleted_jobs': len(jobs), 'deleted_job_ids': [row['id'] for row in jobs]}

    @staticmethod
    def decode_job(row) -> dict:
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        if result.get('conversation_id') is not None:
            result['payload']['conversation_id'] = result['conversation_id']
        result["result"] = json.loads(result["result"]) if result["result"] else None
        result["progress_detail"] = json.loads(result["progress_detail"])
        result["auto_recovery"] = json.loads(result["auto_recovery"])
        return result

    def job(self, job_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("任务不存在")
            return self.decode_job(row)

    def jobs(self, book_id: str | None = None, *, kind: str | None = None, conversation_id=None, limit: int = 100) -> list[dict]:
        conditions, params = [], []
        if book_id:
            conditions.append("book_id=?")
            params.append(book_id)
        if kind:
            conditions.append("kind=?")
            params.append(kind)
        if conversation_id:
            self.conversation(conversation_id, book_id=book_id)
            conditions.append("conversation_id=?")
            params.append(conversation_id)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self.connection() as db:
            return [self.decode_job(row) for row in db.execute(
                "SELECT * FROM jobs" + where + " ORDER BY created_at DESC LIMIT ?", [*params, limit])]

    def update_job(self, job_id: str, **fields):
        allowed = {"status", "progress", "result", "error", "book_id", "progress_detail", "auto_recovery"}
        if set(fields) - allowed:
            raise ValueError("未知任务字段")
        if "result" in fields and fields["result"] is not None:
            fields["result"] = json.dumps(fields["result"], ensure_ascii=False)
        for field in ('progress_detail', 'auto_recovery'):
            if field in fields:
                fields[field] = json.dumps(fields[field], ensure_ascii=False)
        fields["updated_at"] = now()
        with self.connection() as db:
            db.execute("UPDATE jobs SET " + ",".join(key + "=?" for key in fields) + " WHERE id=?",
                       [*fields.values(), job_id])


class JobService:
    def __init__(self, library: Library, *, workers: int = 2, logs=None,
                 retry_delays=(30, 60, 120, 300), recovery_poll_interval=1):
        self.library = library
        self.logs = logs or Observability(library.directory / "logs")
        self.owns_logs = logs is None
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="reader-job")
        self.stopping = threading.Event()
        self.vector_lock = threading.RLock()
        self.submit_lock = threading.RLock()
        self.scheduled = set()
        self.pause_events = {}
        self.retry_delays = retry_delays
        self.recovery_poll_interval = recovery_poll_interval
        with library.connection() as db:
            recovered = db.execute("UPDATE jobs SET status='interrupted',progress='服务已重启，可继续执行' WHERE status IN ('queued','running')").rowcount
        self.logs.event("任务服务启动", recovered_jobs=recovered, workers=workers)
        # 每本书只恢复最新的关系任务；旧任务、已完成任务和主动暂停任务不复活。
        with library.connection() as db:
            graph_jobs = [library.decode_job(row) for row in db.execute("""
                SELECT j.* FROM jobs j WHERE kind='graph' AND status='interrupted'
                AND NOT EXISTS (SELECT 1 FROM jobs n WHERE n.kind='graph' AND n.book_id IS j.book_id
                    AND (n.created_at>j.created_at OR (n.created_at=j.created_at AND n.id>j.id)))
            """)]
        for job in graph_jobs:
            recovery = {**job['auto_recovery'], 'automatic': True,
                        'restarts': job['auto_recovery'].get('restarts', 0) + 1}
            library.update_job(job['id'], status='queued', error=None,
                               progress='服务已恢复，关系索引将从检查点自动继续', auto_recovery=recovery)
            if not recovery.get('waiting'):
                self._schedule(job['id'])
        self.recovery_thread = threading.Thread(target=self._recovery_loop, name='reader-recovery', daemon=True)
        self.recovery_thread.start()

    def _schedule(self, job_id, request_id=None):
        """统一去重调度，手动恢复与自动恢复不会重复启动同一个任务。"""
        with self.submit_lock:
            if self.stopping.is_set() or job_id in self.scheduled:
                return
            self.scheduled.add(job_id)
            self.pool.submit(self._run, job_id, request_id)

    def _recovery_loop(self):
        # 等待时不占分析线程；其他书籍和问答仍可执行。
        while not self.stopping.wait(self.recovery_poll_interval):
            try:
                with self.library.connection() as db:
                    waiting = [self.library.decode_job(row) for row in db.execute(
                        "SELECT * FROM jobs WHERE kind='graph' AND status='queued'")]
                for job in waiting:
                    recovery = job['auto_recovery']
                    if recovery.get('waiting') and recovery.get('retry_at', 0) <= time.time():
                        self._schedule(job['id'])
            except Exception as exc:
                self.logs.error('检查关系索引自动恢复失败', exc)

    def pause(self, job_id):
        with self.submit_lock:
            job = self.library.job(job_id)
            if job['kind'] != 'graph' or job['status'] not in {'queued', 'running'}:
                raise ValueError('仅可暂停正在执行或等待恢复的关系索引')
            self.pause_events.setdefault(job_id, threading.Event()).set()
            self.library.update_job(job_id, status='paused', error=None,
                progress='已暂停；已完成的检查点保留，重启服务也不会自动继续',
                auto_recovery={**job['auto_recovery'], 'waiting': False, 'resume_requested': False})
            return self.library.job(job_id)

    def resume(self, job_id):
        with self.submit_lock:
            if self.stopping.is_set():
                raise ValueError('服务正在停止，请稍后继续')
            job = self.library.job(job_id)
            if job['kind'] != 'graph' or job['status'] not in {'paused', 'failed', 'interrupted', 'queued', 'running'}:
                raise ValueError('该关系索引无需继续')
            if job['status'] == 'running':
                return {**job, 'reused_job': True}
            active = job_id in self.scheduled
            recovery = {**job['auto_recovery'], 'waiting': False, 'attempt': 0, 'format_attempt': 0,
                        'resume_requested': active, 'automatic': True}
            self.library.update_job(job_id, status='queued', error=None,
                progress='等待当前批次结束后继续' if active else '正在从检查点继续关系索引', auto_recovery=recovery)
            if not active:
                self.pause_events.setdefault(job_id, threading.Event()).clear()
                self._schedule(job_id)
            return self.library.job(job_id)

    @staticmethod
    def recovery_kind(exc):
        """临时接口故障持续恢复；认证、额度和本地配置错误不反复请求。"""
        seen = set()
        candidate = None
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            status = getattr(exc, 'status_code', None) or getattr(getattr(exc, 'response', None), 'status_code', None)
            if status in {401, 403} or quota_exhausted(exc):
                return None
            if type(exc).__name__ in {'APIConnectionError', 'APITimeoutError', 'ConnectError', 'ReadError',
                    'ReadTimeout', 'ConnectTimeout', 'PoolTimeout', 'WriteTimeout', 'TimeoutError', 'ConnectionError'}:
                candidate = 'temporary'
            if status in {408, 409, 429} or isinstance(status, int) and 500 <= status <= 599:
                candidate = 'temporary'
            if isinstance(exc, ModelOutputError):
                candidate = 'format'
            exc = exc.__cause__ or exc.__context__
        return candidate

    def submit(self, kind: str, payload: dict, book_id: str | None = None):
        with self.submit_lock:
            if self.stopping.is_set():
                raise ValueError("服务正在停止")
            payload = dict(payload)
            if kind == 'ask' and book_id is not None:
                conversation = self.library.conversation(payload['conversation_id'], book_id=book_id) if payload.get('conversation_id') else self.library.default_conversation(book_id)
                payload['conversation_id'] = conversation['id']
            elif payload.get('conversation_id'):
                raise ValueError("只有原文问答可以关联会话")
            with self.library.connection() as db:
                pending = [self.library.decode_job(row) for row in db.execute(
                    "SELECT * FROM jobs WHERE book_id IS ? AND kind=? AND status IN ('queued','running') ORDER BY created_at", (book_id, kind))]
            for current in pending:
                if kind in {"index", "graph"} or current["payload"] == payload:
                    self.logs.event("复用正在执行的任务", job_id=current["id"], kind=kind, book_id=book_id)
                    return {**current, "reused_job": True}
            job = self.library.create_job(kind, payload, book_id)
            request_id = current_request_id()
            self.logs.event("任务加入队列", job_id=job["id"], book_id=book_id, kind=kind, request_id=request_id)
            self._schedule(job["id"], request_id)
            return job

    def progress(self, job_id: str, message: str):
        if self.stopping.is_set() or self.pause_events.get(job_id, threading.Event()).is_set():
            raise RuntimeError("任务已中断，可继续")
        fields = {"progress": str(message)}
        if isinstance(message, ProgressUpdate):
            fields["progress_detail"] = message.detail
        self.library.update_job(job_id, **fields)
        self.logs.event("任务进度", job_id=job_id, progress=message)

    def _run(self, job_id: str, request_id=None):
        job = self.library.job(job_id)
        try:
            with self.logs.bind(job_id=job_id, book_id=job["book_id"], kind=job["kind"], request_id=request_id):
                self._run_bound(job)
        finally:
            with self.submit_lock:
                self.scheduled.discard(job_id)
                pending = self.library.job(job_id)
                if pending['status'] == 'queued' and pending['auto_recovery'].get('resume_requested'):
                    self.pause_events.setdefault(job_id, threading.Event()).clear()
                    self.library.update_job(job_id, auto_recovery={**pending['auto_recovery'], 'resume_requested': False})
                    self._schedule(job_id)

    def _run_bound(self, job):
        job_id = job["id"]
        started = time.perf_counter()
        pause_signal = self.pause_events.setdefault(job_id, threading.Event())
        if job['status'] == 'paused' or pause_signal.is_set():
            return
        self.logs.event("任务开始")
        try:
            with self.submit_lock:
                self.progress(job_id, "开始执行")
                self.library.update_job(job_id, status="running", error=None,
                    auto_recovery={**job['auto_recovery'], 'waiting': False})
            # Qdrant 本地存储使用独占文件锁，因此相关客户端依次运行。
            # 连接独立 Qdrant 服务后可并行执行相关任务。
            vector_job = job["kind"] in {"ask", "index"} and not get_config().get("QDRANT_URL")
            saved_progress = None
            def api_notice(message):
                nonlocal saved_progress
                if message is None:
                    if saved_progress is not None and not self.stopping.is_set():
                        self.progress(job_id, saved_progress)
                    saved_progress = None
                else:
                    if saved_progress is None:
                        saved_progress = self.library.job(job_id)["progress"]
                    self.progress(job_id, message)
            with api_context(api_notice, lambda: self.stopping.is_set() or pause_signal.is_set(), priority=0 if job["kind"] == "ask" else 10):
                with self.vector_lock if vector_job else nullcontext():
                    result = asyncio.run(self._execute(job))
            detail = self.library.job(job_id)["progress_detail"]
            if detail:
                detail.update(elapsed_seconds=round(time.perf_counter() - started, 1), remaining_seconds=0, eta_at=now(), stage="已完成")
            with self.submit_lock:
                if pause_signal.is_set():
                    return
                self.library.update_job(job_id, status="completed", progress="已完成", result=result, progress_detail=detail,
                                        auto_recovery={**job['auto_recovery'], 'waiting': False, 'attempt': 0})
                self.logs.event("任务完成", status=result.get("status"), elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
        except Exception as exc:
            with self.submit_lock:
                if pause_signal.is_set():
                    # 立即恢复时，由调度收尾重新入队；否则保持主动暂停。
                    return
                kind = self.recovery_kind(exc) if job['kind'] == 'graph' else None
                if not self.stopping.is_set() and kind:
                    current = self.library.job(job_id)
                    checkpoint = current['progress_detail'].get('current', 0)
                    recovery = current['auto_recovery']
                    advanced = checkpoint > recovery.get('last_checkpoint', 0)
                    attempt = (0 if advanced else recovery.get('attempt', 0)) + 1
                    format_attempt = (0 if advanced else recovery.get('format_attempt', 0)) + (kind == 'format')
                    if kind != 'format' or format_attempt <= 3:
                        delay = self.retry_delays[min(attempt - 1, len(self.retry_delays) - 1)]
                        recovery = {**recovery, 'automatic': True, 'waiting': True, 'attempt': attempt,
                                    'format_attempt': format_attempt, 'last_checkpoint': checkpoint,
                                    'retry_at': time.time() + delay, 'reason': kind}
                        self.library.update_job(job_id, status='queued', error=None, auto_recovery=recovery,
                            progress=f'接口暂时不可用，约 {delay:g} 秒后自动恢复（第 {attempt} 次）；已完成的检查点保留')
                        self.logs.error('关系索引等待自动恢复', exc, attempt=attempt, delay_seconds=delay)
                        return
                if self.stopping.is_set():
                    status, message = "interrupted", "任务已中断，可继续执行"
                    self.logs.event("任务已中断", level=logging.WARNING, elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
                else:
                    status = "failed"
                    # 模型服务的异常可能包含密钥或敏感响应，不能直接回传。
                    message = str(exc) if isinstance(exc, (ValueError, KeyError, APIServiceError)) else f"执行失败（{type(exc).__name__}），请检查接口配置和服务状态"
                    self.logs.error("任务失败", exc, elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
                detail = self.library.job(job_id)["progress_detail"]
                if detail:
                    detail.update(elapsed_seconds=round(time.perf_counter() - started, 1), remaining_seconds=None, eta_at=None)
                self.library.update_job(job_id, status=status, progress=message, error=message, progress_detail=detail)

    async def _execute(self, job: dict) -> dict:
        corpus = Corpus(self.library.directory)
        payload = job["payload"]
        backend = verifier = retriever = None
        progress = lambda message: self.progress(job["id"], message)
        try:
            if job["kind"] == "ingest":
                progress("正在流式导入并建立原文索引")
                result = corpus.ingest(payload["path"], name=payload["name"], encoding=payload.get("encoding", "utf-8-sig"))
                self.library.sync()
                self.library.update_book(result["id"], title=payload["title"], archived=False)
                self.library.update_job(job["id"], book_id=result["id"])
                return result
            book = self.library.book(job["book_id"])
            sources = [book["name"]]
            ids = [book["id"]]
            if job["kind"] == "index":
                retriever = make_retriever(corpus, for_index=True)
                return await retriever.semantic.build(ids, batch_size=payload.get("batch_size", 16), progress=progress)
            if job["kind"] == "ask" and payload.get("extractive"):
                return await ReadingAgent(corpus).ask(payload["question"], sources=sources, extractive=True)
            backend = CompatibleBackend.from_env()
            verifier = CompatibleBackend.from_env(verifier=True)
            if job["kind"] == "ask":
                reused = payload.get('reuse_evidence')
                if reused:
                    if reused['source_sha256'] != book['sha256']:
                        raise ValueError('原文版本已变化，请重新检索')
                else:
                    retriever = make_retriever(corpus, mode=payload.get("retrieval"))
                progress("正在复用本次原文，重新生成并核对引文" if reused else "正在检索原文、生成回答并逐条核对引用")
                extra = {'evidence_ids': reused['chunk_ids'], 'required_entities': reused.get('entities', [])} if reused else {}
                return await ReadingAgent(corpus, backend, verifier=verifier, retriever=retriever,
                                          max_context_chars=payload.get("max_context_chars", 16000)).ask(payload["question"], sources=sources, **extra)
            if job["kind"] == "scan":
                return await FullScanner(corpus, backend, verifier=verifier, batch_chars=payload.get("batch_chars", 12000),
                                         progress=progress).scan(payload["question"], sources=sources)
            if job["kind"] == "graph":
                return await EvidenceGraph(corpus).build(backend, verifier=verifier, sources=sources,
                                                         batch_chars=payload.get("batch_chars", 6000), progress=progress)
            raise ValueError("不支持的任务类型")
        finally:
            try:
                if retriever is not None:
                    await retriever.close()
                if verifier is not None:
                    await verifier.close()
                if backend is not None:
                    await backend.close()
            finally:
                corpus.close()

    def close(self):
        with self.submit_lock:
            self.stopping.set()
            with self.library.connection() as db:
                db.execute("UPDATE jobs SET status='interrupted',progress='服务停止，可继续执行' WHERE status='queued'")
        self.recovery_thread.join(timeout=2)
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.logs.event("任务服务停止")
        if self.owns_logs:
            self.logs.close()
