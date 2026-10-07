"""缓存原文相关性分数；仅存散列与分数，不存问题、原文或密钥。"""

import math
import sqlite3
import time
from pathlib import Path


class RerankCache:
    def __init__(self, path: Path, *, ttl=7 * 86400, maximum=50000):
        self.path, self.ttl, self.maximum = path, ttl, maximum
        with sqlite3.connect(path, timeout=30) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS scores (
                identity TEXT NOT NULL, question_hash TEXT NOT NULL, text_hash TEXT NOT NULL,
                score REAL NOT NULL, created_at REAL NOT NULL,
                PRIMARY KEY(identity, question_hash, text_hash))''')
            db.execute('CREATE INDEX IF NOT EXISTS scores_age ON scores(created_at)')

    def get(self, identity, question_hash, text_hashes):
        found = {}
        with sqlite3.connect(self.path, timeout=30) as db:
            for offset in range(0, len(text_hashes), 400):
                batch = text_hashes[offset:offset + 400]
                placeholders = ','.join('?' for _ in batch)
                rows = db.execute(f'''SELECT text_hash, score FROM scores
                    WHERE identity=? AND question_hash=? AND created_at>=?
                    AND text_hash IN ({placeholders})''', [identity, question_hash, time.time() - self.ttl, *batch])
                for key, score in rows:
                    if type(score) in {int, float} and math.isfinite(score):
                        found[key] = score
        return found

    def put(self, identity, question_hash, scores):
        with sqlite3.connect(self.path, timeout=30) as db:
            db.executemany('INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?)',
                           [(identity, question_hash, key, score, time.time()) for key, score in scores.items()])
            db.execute('DELETE FROM scores WHERE created_at<?', (time.time() - self.ttl,))
            count = db.execute('SELECT count(*) FROM scores').fetchone()[0]
            if count > self.maximum:
                db.execute('DELETE FROM scores WHERE rowid IN (SELECT rowid FROM scores ORDER BY created_at LIMIT ?)',
                           (count - self.maximum,))
