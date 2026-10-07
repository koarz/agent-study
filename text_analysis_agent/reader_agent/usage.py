"""保存接口实际用量，只记录模型名称、计数和任务编号，不保存请求内容或密钥。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path


CATEGORIES = ("llm", "embedding", "rerank")


def category_for(operation):
    if "嵌入" in operation:
        return "embedding"
    if "重排" in operation:
        return "rerank"
    return "llm"


def normalize_usage(value, category):
    """兼容常见 usage 字段；没有返回的数值保留为空，不按字数估算。"""
    def field(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    def number(*keys):
        for key in keys:
            result = field(value, key)
            if type(result) is int and result >= 0:
                return result
        return None

    inputs = number("input_tokens", "prompt_tokens")
    outputs = number("output_tokens", "completion_tokens")
    total = number("total_tokens")
    # 嵌入没有生成文本，部分兼容接口只返回总输入 token。
    if category == "embedding" and (inputs is not None or total is not None):
        if inputs is None:
            inputs = total
        outputs = 0
    if total is None and inputs is not None and outputs is not None:
        total = inputs + outputs
    details = field(value, "prompt_tokens_details") or field(value, "input_tokens_details")
    cached = field(details, "cached_tokens")
    if type(cached) is not int or cached < 0:
        cached = number("prompt_cache_hit_tokens", "cached_input_tokens")
    return {"input_tokens": inputs, "output_tokens": outputs, "total_tokens": total,
            "cached_input_tokens": cached}


class UsageStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS usage (
                    id TEXT PRIMARY KEY, call_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    category TEXT NOT NULL, operation TEXT NOT NULL, model TEXT NOT NULL,
                    status TEXT NOT NULL, source TEXT NOT NULL,
                    input_tokens INTEGER, output_tokens INTEGER, total_tokens INTEGER,
                    cached_input_tokens INTEGER, documents INTEGER, document_chars INTEGER,
                    book_id TEXT, job_id TEXT
                );
                CREATE INDEX IF NOT EXISTS usage_date ON usage(created_at);
                CREATE INDEX IF NOT EXISTS usage_call ON usage(call_id);
                CREATE TABLE IF NOT EXISTS log_imports (path TEXT PRIMARY KEY, size INTEGER, modified INTEGER);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def record(self, values):
        # 显式列白名单，调用方即使携带其他字段也不会写入数据库。
        keys = ("id", "call_id", "created_at", "category", "operation", "model", "status", "source",
                "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "documents",
                "document_chars", "book_id", "job_id")
        row = dict(values)
        row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        for key in keys[8:14]:
            if type(row.get(key)) is not int or row[key] < 0:
                row[key] = None
        with self.connection() as db:
            db.execute(f"INSERT OR IGNORE INTO usage ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})",
                       [row.get(key) for key in keys])

    def import_logs(self, directory):
        """补入仍保留的历史日志；按调用编号去重，新版请求记录优先。"""
        for path in sorted(Path(directory).glob("runtime.log*")):
            if not path.is_file():
                continue
            stat = path.stat()
            with self.connection() as db:
                previous = db.execute("SELECT size, modified FROM log_imports WHERE path=?", (path.name,)).fetchone()
                if previous and tuple(previous) == (stat.st_size, stat.st_mtime_ns):
                    continue
                with path.open(encoding="utf-8", errors="replace") as reader:
                    for line in reader:
                        try:
                            row = json.loads(line)
                            if not isinstance(row, dict) or row.get("event") not in {"模型调用完成", "模型调用失败"}:
                                continue
                            if row.get("usage_tracking") == "api-v1":
                                continue
                            call_id = row.get("call_id")
                            if not isinstance(call_id, str) or not call_id or db.execute(
                                    "SELECT 1 FROM usage WHERE call_id=? LIMIT 1", (call_id,)).fetchone():
                                continue
                            operation, model = row.get("operation"), row.get("model")
                            if not isinstance(operation, str) or not isinstance(model, str):
                                continue
                            timestamp = datetime.fromisoformat(row["time"]).astimezone(timezone.utc).isoformat()
                            category = category_for(operation)
                            tokens = normalize_usage(row, category)
                            numeric = [row.get(key) if type(row.get(key)) is int and row[key] >= 0 else None
                                       for key in ("documents", "document_chars")]
                            db.execute("INSERT OR IGNORE INTO usage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                ["legacy:" + call_id, call_id, timestamp, category, operation, model,
                                 "success" if row["event"] == "模型调用完成" else "failed", "log",
                                 *tokens.values(), *numeric, row.get("book_id"), row.get("job_id")])
                        except (ValueError, KeyError, TypeError):
                            continue
                db.execute("INSERT OR REPLACE INTO log_imports VALUES (?,?,?)",
                           (path.name, stat.st_size, stat.st_mtime_ns))

    def summary(self, days=0):
        start = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat() if days else None
        where = "WHERE created_at>=?" if start else ""
        params = (start,) if start else ()
        columns = """
            COUNT(*) AS requests, SUM(status='success') AS successful_requests,
            SUM(status='failed') AS failed_requests, SUM(source='log') AS historical_calls,
            SUM(source='local') AS local_calls,
            SUM(total_tokens IS NULL) AS missing_usage,
            SUM(input_tokens IS NOT NULL) AS input_records,
            SUM(output_tokens IS NOT NULL) AS output_records,
            SUM(total_tokens IS NOT NULL) AS total_records,
            SUM(cached_input_tokens IS NOT NULL) AS cached_records,
            COALESCE(SUM(input_tokens),0) AS input_tokens,
            COALESCE(SUM(output_tokens),0) AS output_tokens,
            COALESCE(SUM(total_tokens),0) AS total_tokens,
            COALESCE(SUM(cached_input_tokens),0) AS cached_input_tokens,
            COALESCE(SUM(documents),0) AS documents,
            COALESCE(SUM(document_chars),0) AS document_chars
        """
        with self.connection() as db:
            total = dict(db.execute(f"SELECT {columns} FROM usage {where}", params).fetchone())
            groups = [dict(row) for row in db.execute(
                f"SELECT category, {columns} FROM usage {where} GROUP BY category", params)]
            models = [dict(row) for row in db.execute(
                f"SELECT category,model,operation, {columns} FROM usage {where} GROUP BY category,model,operation ORDER BY category,model,operation", params)]
            first = db.execute("SELECT MIN(created_at) FROM usage").fetchone()[0]
        # 空集合上的 SUM 返回 NULL，计数在响应中统一为零。
        for row in [total, *groups, *models]:
            for key in total:
                if row.get(key) is None:
                    row[key] = 0
        empty = {key: 0 for key in total}
        by_category = {row["category"]: row for row in groups}
        return {"days": days, "start": start, "first_record_at": first,
                "updated_at": datetime.now(timezone.utc).isoformat(), "total": total,
                "categories": [{**empty, "category": category, **by_category.get(category, {})}
                               for category in CATEGORIES], "models": models}
