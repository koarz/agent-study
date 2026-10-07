"""保存不可变原文快照，计算精确偏移，并提供本地中英文检索。"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import tempfile
from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def terms(text: str) -> list[str]:
    # 使用中文二元和三元字符词组建立 FTS5 索引，无需下载分词模型。
    # 英文按完整单词索引，索引只参与召回，不作为事实依据。
    result: list[str] = []
    for run in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9_]+", text.casefold()):
        if re.fullmatch(r"[\u3400-\u9fff]+", run):
            for size in (2, 3):
                result.extend(run[i : i + size] for i in range(len(run) - size + 1))
            if len(run) == 1:
                result.append(run)
        else:
            result.append(run)
    return result


@dataclass(frozen=True)
class Chunk:
    id: str
    source_id: str
    source: str
    chapter: str
    start: int
    end: int
    line_start: int
    line_end: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


CHAPTER = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}[ \t]+[^\r\n]+|第[零〇一二三四五六七八九十百千万两0-9]+[章回节卷][^\r\n]*|Chapter[ \t]+\d+[^\r\n]*)",
    re.IGNORECASE,
)


def split_text(text: str, size: int = 1000, overlap: int = 160):
    if size < 100 or not 0 <= overlap < size:
        raise ValueError("分块大小至少 100，重叠长度必须小于分块大小")
    headings = list(CHAPTER.finditer(text))
    sections = [(0, "正文")]
    for match in headings:
        if match.start() == 0:
            sections[0] = (0, match.group().strip())
        else:
            sections.append((match.start(), match.group().strip()))
    for i, (section_start, chapter) in enumerate(sections):
        section_end = sections[i + 1][0] if i + 1 < len(sections) else len(text)
        start = section_start
        while start < section_end:
            end = min(start + size, section_end)
            if end < section_end:
                # 优先在末尾四分之一范围内按段落或句子边界切分。
                floor = start + size * 3 // 4
                for boundary in ("\n", "。", "！", "？", ". "):
                    pos = text.rfind(boundary, floor, end)
                    if pos >= floor:
                        end = pos + len(boundary)
                        break
            if text[start:end].strip():
                yield start, end, chapter
            if end == section_end:
                break
            start = max(start + 1, end - overlap)


@dataclass(frozen=True)
class StreamBlock:
    text: str
    chapter: str
    start: int
    end: int
    byte_start: int
    byte_end: int
    line_start: int
    line_end: int


def split_stream(reader, size: int = 2000, overlap: int = 200):
    """只保留少量原文分块；即使小说没有换行，也不会整本载入内存。

    通过预读和真实行边界识别章节。字节偏移以保存后的 UTF-8 快照为准。
    """
    if size < 100 or not 0 <= overlap < size:
        raise ValueError("分块大小至少 100，重叠长度必须小于分块大小")
    buffer, eof = "", False
    char_offset = byte_offset = 0
    line = 1
    at_line_start = True
    chapter = "正文"
    active_heading = None
    while True:
        while len(buffer) < size * 2 + 1 and not eof:
            piece = reader.read(size * 2 + 1 - len(buffer))
            if not piece:
                eof = True
            buffer += piece
        if not buffer:
            break
        # 缓冲区截断产生的开头不一定是原文的真实行首。
        headings = [match for match in CHAPTER.finditer(buffer)
                    if (match.start() > 0 or at_line_start)
                    and (eof or match.end() < len(buffer))]
        following = None
        for heading in headings:
            absolute = char_offset + heading.start()
            if absolute == active_heading:
                continue
            if heading.start() == 0:
                chapter = heading.group().strip()
                active_heading = absolute
            else:
                following = heading.start()
                break
        end = min(size, len(buffer))
        section_boundary = following is not None and following <= end
        if section_boundary:
            end = following
        elif end < len(buffer):
            floor = size * 3 // 4
            for boundary in ("\n", "。", "！", "？", ". "):
                pos = buffer.rfind(boundary, floor, end)
                if pos >= floor:
                    end = pos + len(boundary)
                    break
        content = buffer[:end]
        if content.strip():
            yield StreamBlock(content, chapter, char_offset, char_offset + end,
                              byte_offset, byte_offset + len(content.encode("utf-8")),
                              line, line + content[:-1].count("\n"))
        if eof and end == len(buffer):
            break
        consumed = end if section_boundary else max(1, end - overlap)
        prefix = buffer[:consumed]
        char_offset += consumed
        byte_offset += len(prefix.encode("utf-8"))
        line += prefix.count("\n")
        at_line_start = prefix.endswith("\n")
        buffer = buffer[consumed:]


def file_digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as reader:
        for piece in iter(lambda: reader.read(65536), b""):
            checksum.update(piece)
    return checksum.hexdigest()


class Corpus:
    def __init__(self, directory: Path | str, *, create: bool = False):
        directory = Path(directory).resolve()
        self.directory = directory
        path = directory / "corpus.sqlite3"
        if create:
            directory.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path, timeout=30)
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL,
                    original_path TEXT NOT NULL, sha256 TEXT NOT NULL, text TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL, chapter TEXT NOT NULL,
                    start INTEGER NOT NULL, end INTEGER NOT NULL,
                    line_start INTEGER NOT NULL, line_end INTEGER NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS source_chunks ON chunks(source_id, ordinal);
                CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
                    chunk_id UNINDEXED, tokens, tokenize='unicode61'
                );
            """)
        else:
            if not path.is_file():
                raise ValueError("知识库不存在，请先运行 ingest 导入原文")
            self.db = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=30)
        self.db.row_factory = sqlite3.Row
        # 允许前端读取和后台短事务写入并行；不能在模型请求期间持有读游标。
        if self.db.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
            self.db.execute("PRAGMA journal_mode=WAL").fetchone()
        self.db.execute("PRAGMA cache_size=-8192")
        self.db.execute("PRAGMA temp_store=FILE")
        # 只增加字段，保留旧 SQLite 快照的读取能力。
        # 新导入使用文件快照和流式处理，原有原文及段落不会被删除。
        additions = {
            "sources": {"chars": "INTEGER", "context_chars": "INTEGER", "chunk_count": "INTEGER",
                        "snapshot_path": "TEXT", "snapshot_bytes": "INTEGER"},
            "chunks": {"byte_start": "INTEGER", "byte_end": "INTEGER", "sha256": "TEXT"},
        }
        with self.db:
            for table, columns in additions.items():
                existing = {row["name"] for row in self.db.execute(f"PRAGMA table_info({table})")}
                for name, kind in columns.items():
                    if name not in existing:
                        self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
            if self.db.execute("SELECT 1 FROM sources WHERE chars IS NULL LIMIT 1").fetchone():
                self.db.execute("""UPDATE sources SET chars=length(text),
                    chunk_count=(SELECT count(*) FROM chunks WHERE source_id=sources.id),
                    context_chars=(SELECT coalesce(sum(length(text)),0) FROM chunks WHERE source_id=sources.id)
                    WHERE chars IS NULL""")
        self._verified: dict[str, str] = {}
        self._newlines: dict[str, list[int]] = {}
        self._hashes: dict[str, str] = {}
        self._snapshots: dict[str, tuple] = {}
        self._handles: dict[str, object] = {}

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self.db.close()

    def ingest(self, path: Path | str, *, name: str | None = None, encoding: str = "utf-8-sig",
               chunk_chars: int = 2000, overlap: int = 200) -> dict:
        path = Path(path).resolve()
        if path.suffix.lower() not in {".txt", ".md", ".markdown"}:
            raise ValueError("当前支持 .txt、.md、.markdown；请先将其他格式导出为文本")
        if chunk_chars < 100 or not 0 <= overlap < chunk_chars:
            raise ValueError("分块大小至少 100，重叠长度必须小于分块大小")
        name = name or path.name
        if not name.strip():
            raise ValueError("原文名称不能为空")
        snapshots = self.directory / "snapshots"
        snapshots.mkdir(exist_ok=True)
        staged = None
        try:
            checksum = hashlib.sha256()
            chars = 0
            has_content = False
            with tempfile.NamedTemporaryFile(dir=snapshots, suffix=".tmp", delete=False) as writer:
                staged = Path(writer.name)
                with path.open("r", encoding=encoding, newline="") as reader:
                    while piece := reader.read(65536):
                        if "\x00" in piece:
                            raise ValueError("文件含有非文本字符")
                        has_content |= bool(piece.strip())
                        chars += len(piece)
                        encoded = piece.encode("utf-8")
                        checksum.update(encoded)
                        writer.write(encoded)
            if not has_content:
                raise ValueError("文件为空")
            sha = checksum.hexdigest()
            existing = self.db.execute("SELECT id,sha256 FROM sources WHERE name=?", (name,)).fetchone()
            if existing:
                if existing["sha256"] == sha:
                    return {"id": existing["id"], "name": name, "unchanged": True}
                raise ValueError(f"已有同名原文 {name!r}，内容不同；请用 --name 指定新版本名称")
            source_id = "s_" + hashlib.sha256((name + "\0" + sha).encode()).hexdigest()[:20]
            snapshot = snapshots / (source_id + ".txt")
            if snapshot.exists():
                if file_digest(snapshot) != sha:
                    raise ValueError("同名快照内容不一致，停止导入")
            else:
                staged.replace(snapshot)
            chunk_count = context_chars = 0
            with self.db:
                self.db.execute("""INSERT INTO sources
                    (id,name,original_path,sha256,text,chars,snapshot_path,snapshot_bytes)
                    VALUES (?,?,?,?,'',?,?,?)""",
                    (source_id, name, str(path), sha, chars, str(snapshot.relative_to(self.directory)), snapshot.stat().st_size))
                with snapshot.open("r", encoding="utf-8", newline="") as reader:
                    for index, block in enumerate(split_stream(reader, chunk_chars, overlap)):
                        chunk_id = f"{source_id}_{index:06d}"
                        self.db.execute("""INSERT INTO chunks
                            (id,source_id,ordinal,chapter,start,end,line_start,line_end,text,byte_start,byte_end,sha256)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (chunk_id, source_id, index, block.chapter, block.start, block.end,
                             block.line_start, block.line_end, block.text, block.byte_start,
                             block.byte_end, digest(block.text)))
                        self.db.execute("INSERT INTO search_index VALUES (?, ?)",
                                        (chunk_id, " ".join(terms(block.text))))
                        chunk_count += 1
                        context_chars += len(block.text)
                self.db.execute("UPDATE sources SET context_chars=?,chunk_count=? WHERE id=?",
                                (context_chars, chunk_count, source_id))
            return {"id": source_id, "name": name, "chunks": chunk_count, "chars": chars,
                    "sha256": sha, "unchanged": False, "storage": "streamed_snapshot"}
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)

    def sources(self) -> list[dict]:
        rows = self.db.execute("""
            SELECT id, name, sha256, chars, chunk_count AS chunks FROM sources ORDER BY name
        """)
        return [dict(row) for row in rows]

    def select_sources(self, names: list[str] | None = None) -> list[str]:
        sources = self.sources()
        if not sources:
            raise ValueError("知识库为空，请先导入原文")
        if not names:
            # 多本书必须明确选择范围，避免无意混用不同作品的情节。
            if len(sources) != 1:
                raise ValueError("知识库有多份原文，请用 --source 名称指定阅读范围，可重复指定")
            return [sources[0]["id"]]
        by_name = {s["name"]: s["id"] for s in sources}
        unknown = set(names) - set(by_name)
        if unknown:
            raise ValueError(f"原文名称不存在：{'、'.join(sorted(unknown))}")
        return list(dict.fromkeys(by_name[name] for name in names))

    def _source_text(self, source_id: str) -> str:
        """兼容旧版 SQLite 原文快照；新导入不会整本读取文件。"""
        if source_id not in self._verified:
            row = self.db.execute("SELECT text, sha256 FROM sources WHERE id=?", (source_id,)).fetchone()
            if not row or digest(row["text"]) != row["sha256"]:
                raise ValueError("原文快照校验失败，停止回答")
            self._verified[source_id] = row["text"]
            self._newlines[source_id] = [i for i, char in enumerate(row["text"]) if char == "\n"]
            self._hashes[source_id] = row["sha256"]
        return self._verified[source_id]

    def begin_read(self):
        self._verified.clear()
        self._newlines.clear()
        self._hashes.clear()
        self._snapshots.clear()
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _snapshot(self, source_id: str):
        if source_id not in self._snapshots:
            row = self.db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
            if not row:
                raise ValueError("原文不存在")
            if row["snapshot_path"] is None:
                self._snapshots[source_id] = (row, None, None)
                return row, None
            path = (self.directory / row["snapshot_path"]).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError("原文快照路径不合法")
            if row["text"] or not path.is_file() or file_digest(path) != row["sha256"]:
                raise ValueError("原文快照校验失败，停止回答")
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
            if stat.st_size != row["snapshot_bytes"]:
                raise ValueError("原文快照长度不一致")
            self._snapshots[source_id] = (row, path, signature)
            self._handles[source_id] = path.open("rb")
        row, path, signature = self._snapshots[source_id]
        if path is not None:
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != signature:
                raise ValueError("原文快照在读取期间发生改变，停止回答")
        return row, path

    def _chunk(self, row) -> Chunk:
        source, snapshot = self._snapshot(row["source_id"])
        if snapshot is not None:
            start, end = row["start"], row["end"]
            if not 0 <= start < end <= source["chars"] or end - start != len(row["text"]):
                raise ValueError("检索段落与原文快照不一致，停止回答")
            byte_start, byte_end = row["byte_start"], row["byte_end"]
            if byte_start is None or byte_end is None or not 0 <= byte_start < byte_end <= source["snapshot_bytes"]:
                raise ValueError("段落字节位置不合法")
            handle = self._handles[row["source_id"]]
            handle.seek(byte_start)
            content = handle.read(byte_end - byte_start).decode("utf-8")
            if content != row["text"] or digest(content) != row["sha256"]:
                raise ValueError("检索段落与原文快照不一致，停止回答")
            return Chunk(row["id"], row["source_id"], row["source"], row["chapter"], start, end,
                         row["line_start"], row["line_end"], content)
        text = self._source_text(row["source_id"])
        start, end = row["start"], row["end"]
        if not 0 <= start < end <= len(text) or row["text"] != text[start:end]:
            raise ValueError("检索段落与原文快照不一致，停止回答")
        return Chunk(row["id"], row["source_id"], row["source"], row["chapter"], start, end,
                     bisect_left(self._newlines[row["source_id"]], start) + 1,
                     bisect_left(self._newlines[row["source_id"]], end - 1) + 1, row["text"])

    def _where(self, source_ids: list[str]):
        if not source_ids:
            raise ValueError("必须指定阅读范围")
        return ",".join("?" for _ in source_ids)

    def stats(self, source_ids: list[str]) -> dict:
        row = self.db.execute(
            f"SELECT coalesce(sum(chunk_count),0) AS chunks, coalesce(sum(context_chars),0) AS context_chars FROM sources WHERE id IN ({self._where(source_ids)})",
            source_ids,
        ).fetchone()
        return dict(row)

    def all_chunks(self, source_ids: list[str]) -> list[Chunk]:
        return list(self.iter_chunks(source_ids))

    def iter_chunks(self, source_ids: list[str]):
        # 按原文顺序小页读取，先耗尽查询再向调用者交出段落。
        # 持续挂起的查询会跨越 API 请求，导致其他任务写入被阻塞或快照升级失败。
        sources = self.db.execute(
            f"SELECT id,name FROM sources WHERE id IN ({self._where(source_ids)}) ORDER BY name,id", source_ids).fetchall()
        for source in sources:
            ordinal = -1
            while True:
                rows = self.db.execute(
                    "SELECT c.*, ? AS source FROM chunks c WHERE source_id=? AND ordinal>? ORDER BY ordinal LIMIT 64",
                    (source["name"], source["id"], ordinal)).fetchall()
                if not rows:
                    break
                ordinal = rows[-1]["ordinal"]
                for row in rows:
                    yield self._chunk(row)

    def get_chunk(self, chunk_id: str) -> Chunk:
        row = self.db.execute("SELECT c.*, s.name AS source FROM chunks c JOIN sources s ON s.id=c.source_id WHERE c.id=?", (chunk_id,)).fetchone()
        if row is None:
            raise ValueError("段落不存在")
        return self._chunk(row)

    def search(self, question: str, source_ids: list[str], *, limit: int = 8,
               must_contain: tuple[str, ...] = ()) -> list[Chunk]:
        if limit < 1:
            raise ValueError("检索数量必须为正整数")
        tokens = list(dict.fromkeys(terms(question)))[:100]
        if not tokens:
            return []
        match = " OR ".join('"' + token + '"' for token in tokens)
        # 组合召回要求主体与事件同时出现，防止高频人名挤掉关键事件。
        required = tuple(dict.fromkeys(word for word in must_contain if word))
        constraints = ''.join(' AND instr(c.text, ?) > 0' for _ in required)
        rows = self.db.execute(
            f"""SELECT c.*, s.name AS source FROM search_index
                JOIN chunks c ON c.id=search_index.chunk_id JOIN sources s ON s.id=c.source_id
                WHERE search_index MATCH ? AND source_id IN ({self._where(source_ids)}){constraints}
                ORDER BY bm25(search_index), c.id LIMIT ?""",
            [match, *source_ids, *required, limit],
        )
        return [self._chunk(row) for row in rows]

    def neighbors(self, chunk: Chunk) -> list[Chunk]:
        row = self.db.execute("SELECT ordinal FROM chunks WHERE id=?", (chunk.id,)).fetchone()
        rows = self.db.execute("""
            SELECT c.*, s.name AS source FROM chunks c JOIN sources s ON s.id=c.source_id
            WHERE c.source_id=? AND c.ordinal BETWEEN ? AND ? ORDER BY c.ordinal
        """, (chunk.source_id, row["ordinal"] - 1, row["ordinal"] + 1))
        return [self._chunk(row) for row in rows]

    def citation(self, chunk: Chunk, quote: str, *, relative_start=None) -> dict:
        if not isinstance(quote, str) or not quote.strip() or quote not in chunk.text:
            raise ValueError("引用必须是检索段落中逐字一致的连续原文")
        position = chunk.text.index(quote) if relative_start is None else relative_start
        if type(position) is not int or position < 0 or chunk.text[position:position + len(quote)] != quote:
            raise ValueError("引用位置与原文不一致")
        start = chunk.start + position
        end = start + len(quote)
        source, snapshot = self._snapshot(chunk.source_id)
        if snapshot is not None:
            # 重新读取快照中的真实字节核对引用，不信任模型生成的位置。
            # 校验时只需要当前段落和引文，内存使用不随全书增长。
            canonical = self.get_chunk(chunk.id)
            if canonical != chunk:
                raise ValueError("引用与原文快照不一致")
            prefix = chunk.text[:position]
            return {"chunk_id": chunk.id, "source_id": chunk.source_id, "source": chunk.source,
                    "chapter": chunk.chapter, "quote": quote, "start": start, "end": end,
                    "line_start": chunk.line_start + prefix.count("\n"),
                    "line_end": chunk.line_start + (prefix + quote[:-1]).count("\n"),
                    "source_sha256": source["sha256"]}
        text = self._source_text(chunk.source_id)
        if text[start:end] != quote:
            raise ValueError("引用与原文快照不一致")
        return {"chunk_id": chunk.id, "source_id": chunk.source_id, "source": chunk.source,
                "chapter": chunk.chapter, "quote": quote, "start": start, "end": end,
                "line_start": bisect_left(self._newlines[chunk.source_id], start) + 1,
                "line_end": bisect_left(self._newlines[chunk.source_id], end - 1) + 1,
                "source_sha256": self._hashes[chunk.source_id]}
