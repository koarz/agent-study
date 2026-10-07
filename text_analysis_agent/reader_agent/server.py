"""FastAPI 服务与简约前端，由单个服务进程管理个人书库。"""

from __future__ import annotations

import asyncio
import hashlib
import re
import json
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .config import PROJECT, get_config
from .corpus import Corpus
from .graph import EvidenceGraph
from .library import JobService, Library
from .scan import scan_plan
from .logging_utils import Observability
from .local_models import LocalModelManager, MODEL_KEY


SETTING_KEYS = {
    "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_ID", "LLM_TIMEOUT", "LLM_TEMPERATURE",
    "VERIFY_API_KEY", "VERIFY_BASE_URL", "VERIFY_MODEL_ID", "RETRIEVAL_MODE",
    "EMBEDDING_PROVIDER", "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL_ID",
    "EMBEDDING_DIMENSIONS", "EMBEDDING_QUERY_PREFIX", "RERANK_PROVIDER", "RERANK_API_KEY",
    "RERANK_URL", "RERANK_MODEL_ID", "RERANK_MAX_DOCUMENTS", "RERANK_MAX_CHARS", "QDRANT_URL", "QDRANT_API_KEY",
    "LOCAL_RERANK_MODEL_ID", "LOCAL_DEVICE", "LOCAL_RERANK_PASSES",
    "LOCAL_RERANK_BATCH_SIZE", "LOCAL_RERANK_WINDOW_CHARS",
}
SECRET_KEYS = {key for key in SETTING_KEYS if key.endswith("API_KEY")}


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["ask", "index", "graph", "scan"]
    question: str = Field(default="", max_length=4000)
    retrieval: Literal["hybrid", "lexical"] | None = None
    extractive: bool = False
    max_context_chars: int = Field(default=16000, ge=1000, le=64000)
    batch_chars: int = Field(default=12000, ge=1000, le=64000)
    batch_size: int = Field(default=16, ge=1, le=128)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=32)


class ConversationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="新会话", min_length=1, max_length=100)


class BookUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, min_length=1, max_length=200)
    archived: bool | None = None


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: dict[str, str]


def create_app(directory: Path | str = PROJECT / "data" / "default", *, config_path: Path | None = None, model_directory: Path | None = None):
    library = Library(directory)
    settings_path = config_path or PROJECT / ".env"
    logs = Observability(library.directory / "logs")
    local_models = LocalModelManager(model_directory or PROJECT / "models")

    @asynccontextmanager
    async def lifespan(app):
        logs.event("服务启动")
        app.state.jobs = JobService(library, logs=logs)
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.jobs.close)
            await asyncio.to_thread(local_models.close)
            logs.event("服务停止")
            logs.close()

    app = FastAPI(title="原文阅读工作台", lifespan=lifespan)
    app.state.library = library
    app.state.logs = logs
    app.state.local_models = local_models
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        with logs.bind(request_id=request_id):
            try:
                origin = request.headers.get("origin")
                if request.method not in {"GET", "HEAD", "OPTIONS"} and origin and urlparse(origin).netloc != request.headers.get("host"):
                    response = JSONResponse({"detail": "仅接受当前页面发起的操作"}, status_code=403)
                else:
                    response = await call_next(request)
            except Exception as exc:
                logs.error("接口异常", exc, method=request.method)
                response = JSONResponse({"detail": "服务处理失败，请通过请求 ID 查看日志", "request_id": request_id}, status_code=500)
            # 只记录路由模板；不记录查询参数、原文、密钥和请求正文。
            route = request.scope.get("route")
            logs.event("HTTP 请求", channel="access", method=request.method,
                       route=getattr(route, "path", "未匹配路由"), status=response.status_code,
                       elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
            response.headers["X-Request-ID"] = request_id
            if request.url.path.startswith('/assets/'):
                response.headers['Cache-Control'] = 'no-cache'
            elif request.url.path.startswith('/api/'):
                response.headers['Cache-Control'] = 'no-store'
            return response

    @app.exception_handler(KeyError)
    async def missing_handler(request, exc):
        return JSONResponse({"detail": str(exc).strip("'")}, status_code=404)

    @app.exception_handler(ValueError)
    async def value_handler(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(sqlite3.OperationalError)
    async def database_handler(request, exc):
        logs.error("数据库异常", exc, request_id=request.state.request_id)
        return JSONResponse({"detail": "知识库正在写入或暂不可用，请稍后重试"}, status_code=503)

    @app.get("/api/usage")
    def usage(days: int = Query(default=0, ge=0, le=3650)):
        return logs.usage.summary(days)

    @app.get("/api/models")
    def models():
        return [local_models.snapshot()]

    @app.post("/api/models/{model_id}/download", status_code=202)
    def download_model(model_id: str):
        if model_id != MODEL_KEY:
            raise HTTPException(404, "该模型尚未提供本地下载")
        return local_models.start()

    @app.get("/api/health")
    def health():
        config = get_config()
        configured = lambda key: bool(config.get(key)) and not config[key].startswith("YOUR_")
        return {"status": "ok", "model_configured": configured("LLM_API_KEY") and configured("LLM_MODEL_ID"),
                "embedding_configured": configured("EMBEDDING_MODEL_ID"),
                "retrieval_mode": config.get("RETRIEVAL_MODE", "hybrid"), "max_upload_mb": 128}

    @app.get("/api/books")
    def books(archived: bool = False):
        return library.books(archived=archived)

    @app.post("/api/books", status_code=202)
    async def upload_book(request: Request, file: UploadFile = File(...), title: str = Form(""),
                          encoding: str = Form("utf-8-sig")):
        filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        extension = Path(filename).suffix.lower()
        if extension not in {".txt", ".md", ".markdown"}:
            raise HTTPException(400, "请上传 TXT 或 Markdown 原文")
        title = title.strip() or Path(filename).stem
        if not 1 <= len(title) <= 200 or len(encoding) > 50:
            raise HTTPException(400, "书名或编码无效")
        uploads = library.directory / "uploads"
        uploads.mkdir(exist_ok=True)
        destination = uploads / (uuid.uuid4().hex + extension)
        size = 0
        try:
            with destination.open("wb") as writer:
                while piece := await file.read(65536):
                    size += len(piece)
                    if size > 128 * 1024 * 1024:
                        raise HTTPException(413, "文件超过 128 MB 上传上限")
                    writer.write(piece)
            if not size:
                raise HTTPException(400, "原文文件为空")
            return request.app.state.jobs.submit("ingest", {"path": str(destination), "name": title + extension,
                                                            "title": title, "encoding": encoding})
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.get("/api/books/{book_id}")
    def book(book_id: str):
        return library.book(book_id)

    @app.patch("/api/books/{book_id}")
    def edit_book(book_id: str, body: BookUpdate):
        return library.update_book(book_id, title=body.title, archived=body.archived)

    @app.get("/api/books/{book_id}/conversations")
    def conversations(book_id: str):
        return library.conversations(book_id)

    @app.post("/api/books/{book_id}/conversations", status_code=201)
    def create_conversation(book_id: str, body: ConversationUpdate):
        if library.book(book_id)['archived']:
            raise HTTPException(400, "请先恢复这本书再新建会话")
        return library.create_conversation(book_id, title=body.title)

    @app.patch("/api/conversations/{conversation_id}")
    def rename_conversation(conversation_id: str, body: ConversationUpdate):
        return library.rename_conversation(conversation_id, body.title)

    @app.get("/api/books/{book_id}/chapters")
    def chapters(book_id: str):
        library.book(book_id)
        corpus = Corpus(library.directory)
        try:
            return [dict(row) for row in corpus.db.execute("""
                WITH marked AS (
                    SELECT *,CASE WHEN lag(chapter) OVER (ORDER BY ordinal)=chapter THEN 0 ELSE 1 END AS boundary
                    FROM chunks WHERE source_id=?
                ), grouped AS (
                    SELECT *,sum(boundary) OVER (ORDER BY ordinal) AS section FROM marked
                )
                SELECT chapter,min(ordinal) AS first_ordinal,count(*) AS chunks,
                       min(line_start) AS line_start,max(line_end) AS line_end
                FROM grouped GROUP BY section ORDER BY first_ordinal
            """, (book_id,))]
        finally:
            corpus.close()

    @app.get("/api/books/{book_id}/chunks")
    def chunks(book_id: str, offset: int = Query(default=0, ge=0), limit: int = Query(default=8, ge=1, le=30)):
        library.book(book_id)
        corpus = Corpus(library.directory)
        try:
            rows = corpus.db.execute("SELECT id FROM chunks WHERE source_id=? ORDER BY ordinal LIMIT ? OFFSET ?",
                                     (book_id, limit, offset))
            content = [corpus.get_chunk(row["id"]).to_dict() for row in rows]
            return {"items": content, "offset": offset, "total": corpus.stats([book_id])["chunks"]}
        finally:
            corpus.close()

    @app.get("/api/books/{book_id}/chunks/{chunk_id}")
    def chunk(book_id: str, chunk_id: str):
        library.book(book_id)
        corpus = Corpus(library.directory)
        try:
            result = corpus.get_chunk(chunk_id)
            if result.source_id != book_id:
                raise HTTPException(404, "段落不属于当前书籍")
            return {"chunk": result.to_dict(), "neighbors": [c.to_dict() for c in corpus.neighbors(result)]}
        finally:
            corpus.close()

    @app.post("/api/books/{book_id}/jobs", status_code=202)
    def submit_job(book_id: str, body: JobRequest, request: Request):
        book = library.book(book_id)
        if book["archived"]:
            raise HTTPException(400, "请先恢复这本书再进行分析")
        if body.kind in {"ask", "scan"} and not body.question.strip():
            raise HTTPException(400, "请输入问题或扫描主题")
        return request.app.state.jobs.submit(body.kind, body.model_dump(exclude={"kind"}), book_id)

    @app.get("/api/books/{book_id}/scan-plan")
    def plan_scan(book_id: str, batch_chars: int = Query(default=12000, ge=1000, le=64000)):
        library.book(book_id)
        corpus = Corpus(library.directory)
        try:
            return scan_plan(corpus, [book_id], batch_chars)
        finally:
            corpus.close()

    @app.get("/api/books/{book_id}/graph")
    def graph(book_id: str, entity: str = Query(min_length=1, max_length=120)):
        library.book(book_id)
        corpus = Corpus(library.directory)
        try:
            return EvidenceGraph(corpus).facts(entity, [book_id])
        finally:
            corpus.close()

    @app.get("/api/jobs")
    def jobs(book_id: str | None = None, kind: str | None = None, conversation_id: str | None = None, limit: int = Query(default=100, ge=1, le=200)):
        return library.jobs(book_id, kind=kind, conversation_id=conversation_id, limit=limit)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return library.job(job_id)

    @app.post("/api/jobs/{job_id}/retry", status_code=202)
    def retry(job_id: str, request: Request):
        job = library.job(job_id)
        if job['kind'] == 'graph' and job['status'] in {'paused', 'failed', 'interrupted', 'queued', 'running'}:
            return request.app.state.jobs.resume(job_id)
        if job["status"] == "retried" and job["progress_detail"].get("replacement_id"):
            return {**library.job(job["progress_detail"]["replacement_id"]), "reused_job": True}
        payload = job['payload']
        if job['kind'] == 'ask' and job['status'] == 'completed' and (job.get('result') or {}).get('validation_error'):
            book = library.book(job['book_id'])
            coverage = job['result']['coverage']
            scope = next((s for s in coverage.get('sources', []) if s['id'] == book['id']), {})
            traces = coverage.get('context_trace', [])
            if scope.get('sha256') != book['sha256'] or not traces:
                raise HTTPException(400, '原文版本已变化或原文记录不足，请重新提问')
            payload = {**payload, 'reuse_evidence': {
                'chunk_ids': traces[-1]['selected_chunk_ids'], 'source_sha256': book['sha256'],
                'entities': coverage.get('required_entities', []),
            }}
        elif job["status"] not in {"failed", "interrupted"}:
            raise HTTPException(400, "该任务无需继续或正在执行")
        replacement = request.app.state.jobs.submit(job["kind"], payload, job["book_id"])
        library.update_job(job_id, status="retried", progress="已重新执行，新任务 " + replacement["id"][:8],
                           progress_detail={**job["progress_detail"], "replacement_id": replacement["id"]})
        return replacement

    @app.post('/api/jobs/{job_id}/pause')
    def pause_job(job_id: str, request: Request):
        return request.app.state.jobs.pause(job_id)

    @app.get("/api/jobs/{job_id}/download/{artifact}")
    def download(job_id: str, artifact: Literal["report", "evidence"]):
        job = library.job(job_id)
        result = job["result"] or {}
        key = "report" if artifact == "report" else "evidence_file"
        if not result.get(key):
            raise HTTPException(404, "报告尚未生成")
        path = Path(result[key]).resolve()
        if not path.is_relative_to(library.directory) or not path.is_file():
            raise HTTPException(404, "报告不可用")
        return FileResponse(path, filename="原文证据报告.md" if artifact == "report" else "原文证据.jsonl")

    @app.get("/api/settings")
    def settings():
        from dotenv import dotenv_values
        config = get_config() if config_path is None else {k: v for k, v in dotenv_values(settings_path).items() if v is not None}
        return {"values": {k: config.get(k, "") for k in sorted(SETTING_KEYS - SECRET_KEYS)},
                "secrets": {k: bool(config.get(k)) and not config[k].startswith("YOUR_") for k in sorted(SECRET_KEYS)}}

    @app.put("/api/settings")
    def update_settings(body: SettingsUpdate):
        from dotenv import dotenv_values
        if set(body.values) - SETTING_KEYS or any(len(value) > 4000 for value in body.values.values()):
            raise HTTPException(400, "配置包含未知字段或值过长")
        values = {k: v for k, v in dotenv_values(settings_path).items() if v is not None}
        provider = body.values.get("RERANK_PROVIDER", values.get("RERANK_PROVIDER", "api"))
        if provider not in {"api", "local", "none"}:
            raise HTTPException(400, "重排方式必须为 api、local 或 none")
        if "LOCAL_DEVICE" in body.values and body.values["LOCAL_DEVICE"] not in {"auto", "cpu", "cuda"}:
            raise HTTPException(400, "本地设备必须为 auto、cpu 或 cuda")
        for key, minimum, maximum in [('RERANK_MAX_DOCUMENTS', 0, 2048), ('RERANK_MAX_CHARS', 0, 4000000),
                                     ('LOCAL_RERANK_PASSES', 1, 3), ('LOCAL_RERANK_BATCH_SIZE', 1, 32),
                                     ('LOCAL_RERANK_WINDOW_CHARS', 300, 1600)]:
            value = body.values.get(key, '').strip()
            if value:
                try:
                    valid = minimum <= int(value) <= maximum
                except ValueError:
                    valid = False
                if not valid:
                    raise HTTPException(400, f'{key} 必须是 {minimum} 到 {maximum} 的整数')
        for key, value in body.values.items():
            # 密钥输入框留空表示保留原密钥，避免页面保存时意外清除。
            if key not in SECRET_KEYS or value.strip():
                values[key] = value
        temporary = settings_path.with_name(settings_path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as writer:
                for key, value in sorted(values.items()):
                    writer.write(key + "=" + json.dumps(value, ensure_ascii=False) + "\n")
            temporary.replace(settings_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {"saved": True, "environment_overrides": sorted(set(body.values) & set(os.environ))}

    web = PROJECT / "web"
    app.mount("/assets", StaticFiles(directory=web), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        # 即使绕过 start.sh 启动，页面也按当前资源内容生成版本号。
        html = (web / 'index.html').read_text(encoding='utf-8')
        for name in ('app.js', 'styles.css'):
            version = hashlib.sha256((web / name).read_bytes()).hexdigest()[:12]
            path = '/assets/' + name
            html = re.sub(re.escape(path) + r'(?:\?v=[^"\s<>]*)?', path + '?v=' + version, html)
        return HTMLResponse(html, headers={'Cache-Control': 'no-store'})

    return app
