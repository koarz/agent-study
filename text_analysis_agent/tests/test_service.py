from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient

from reader_agent.library import JobService, Library
from reader_agent.server import create_app
from reader_agent.corpus import Corpus
from reader_agent.graph import EvidenceGraph


class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.app = create_app(self.root / "library", config_path=self.root / ".env")
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temporary.cleanup()

    def wait_job(self, job):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = self.client.get("/api/jobs/" + job["id"]).json()
            if result["status"] in {"completed", "failed", "interrupted"}:
                return result
            time.sleep(0.03)
        self.fail("后台任务没有及时结束")

    def upload(self, title, text):
        response = self.client.post("/api/books", files={"file": (title + ".txt", text.encode(), "text/plain")})
        self.assertEqual(response.status_code, 202, response.text)
        job = self.wait_job(response.json())
        self.assertEqual(job["status"], "completed", job)
        return job["book_id"]

    def test_browser_assets_and_multi_book_extractive_citations(self):
        self.assertIn("阅读工作台", self.client.get("/").text)
        self.assertEqual(self.client.get("/assets/app.js").status_code, 200)
        first = self.upload("甲书", "第一章 夜色\n林舟用铜钥匙打开了铁门。\n第二章 清晨\n天亮了。")
        second = self.upload("乙书", "第一章 旧街\n林舟用银钥匙打开了木门。")
        self.assertEqual(len(self.client.get("/api/books").json()), 2)
        response = self.client.post(f"/api/books/{first}/jobs", json={"kind": "ask", "question": "钥匙 门", "extractive": True})
        answer = self.wait_job(response.json())["result"]
        self.assertEqual(answer["status"], "extractive")
        self.assertTrue(answer["evidence"])
        self.assertTrue(all(c["source_id"] == first for c in answer["evidence"]))
        self.assertNotIn("银钥匙", json.dumps(answer, ensure_ascii=False))
        chunk_id = answer["evidence"][0]["id"]
        quote = self.client.get(f"/api/books/{first}/chunks/{chunk_id}").json()
        self.assertIn("铜钥匙", quote["chunk"]["text"])
        self.assertEqual(self.client.get(f"/api/books/{second}/chunks/{chunk_id}").status_code, 404)
        self.assertEqual(len(self.client.get(f"/api/books/{first}/chapters").json()), 2)
        self.assertGreater(self.client.get(f"/api/books/{first}/scan-plan").json()["total_batches"], 0)
        self.client.patch(f"/api/books/{first}", json={"title": "显示书名", "archived": True})
        self.assertEqual(len(self.client.get("/api/books").json()), 1)
        archived = self.client.get("/api/books?archived=true").json()[0]
        self.assertEqual(archived["title"], "显示书名")
        self.assertEqual(archived["name"], "甲书.txt")
        self.assertEqual(self.client.post(f"/api/books/{first}/jobs", json={"kind": "ask", "question": "门", "extractive": True}).status_code, 400)
        self.client.patch(f"/api/books/{first}", json={"archived": False})
        self.assertEqual(len(self.client.get("/api/books").json()), 2)

    def test_settings_secrets_and_input_validation(self):
        secret = "仅用于本地测试的假密钥"
        saved = self.client.put("/api/settings", json={"values": {"LLM_API_KEY": secret, "LLM_MODEL_ID": "模拟模型"}})
        self.assertEqual(saved.status_code, 200)
        response = self.client.get("/api/settings")
        self.assertNotIn(secret, response.text)
        self.assertTrue(response.json()["secrets"]["LLM_API_KEY"])
        self.client.put("/api/settings", json={"values": {"LLM_API_KEY": ""}})
        self.assertIn(secret, (self.root / ".env").read_text())
        self.assertEqual((self.root / ".env").stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.client.put("/api/settings", json={"values": {"UNKNOWN_KEY": "x"}}).status_code, 400)
        self.assertEqual(self.client.put('/api/settings', json={'values': {'RERANK_MAX_DOCUMENTS': '128', 'RERANK_MAX_CHARS': '200000'}}).status_code, 200)
        for key, value in [('RERANK_MAX_DOCUMENTS', '-1'), ('RERANK_MAX_DOCUMENTS', '12.5'), ('RERANK_MAX_CHARS', '4000001')]:
            self.assertEqual(self.client.put('/api/settings', json={'values': {key: value}}).status_code, 400)
        self.assertEqual(self.client.put("/api/settings", json={"values": {}}, headers={"Origin": "http://another-site.test"}).status_code, 403)
        self.assertEqual(self.client.post("/api/books", files={"file": ("假书.pdf", b"text")}).status_code, 400)
        self.assertEqual(self.client.post("/api/books", files={"file": ("空书.txt", b"")}).status_code, 400)
        self.assertEqual(self.client.get("/api/books/no-such-book").status_code, 404)

    def test_book_index_status_distinguishes_partial_complete_and_changed_model(self):
        book_id = self.upload("索引状态测试", "第一章\n明确的原文。\n第二章\n另一段原文。")
        library = self.app.state.library
        def details():
            detail = self.client.get('/api/books/' + book_id).json()
            listed = self.client.get('/api/books').json()[0]
            for key in ('index_status', 'vector_chunks', 'graph_facts'):
                self.assertEqual(detail[key], listed[key])
            return detail
        with patch('reader_agent.library.configured_embedding_identity', return_value='模型甲'):
            self.assertEqual(details()['index_status'], {'index': 'missing', 'graph': 'missing'})
            corpus = Corpus(library.directory)
            try:
                chunks = corpus.all_chunks([book_id])
                self.assertGreater(len(chunks), 1)
                with corpus.db:
                    corpus.db.execute('CREATE TABLE vector_progress (model TEXT, chunk_id TEXT, text_sha256 TEXT, PRIMARY KEY(model,chunk_id))')
                    corpus.db.execute('INSERT INTO vector_progress VALUES (?,?,?)', ('模型甲', chunks[0].id, '测试哈希'))
                self.assertEqual(details()['index_status']['index'], 'partial')
                with corpus.db:
                    corpus.db.executemany('INSERT OR IGNORE INTO vector_progress VALUES (?,?,?)', [('模型甲', c.id, '测试哈希') for c in chunks])
                self.assertEqual(details()['index_status']['index'], 'ready')
                EvidenceGraph(corpus)
                with corpus.db:
                    corpus.db.execute('INSERT INTO graph_facts VALUES (?,?,?,?,?,?,?)', ('关系测试', book_id, '人物甲', '遇见', '人物乙', 0, '{}'))
                self.assertEqual(details()['index_status']['graph'], 'partial')
                task = library.create_job('graph', {}, book_id)
                self.assertEqual(details()['index_status']['graph'], 'building')
                source = corpus.sources()[0]
                result = {'status': 'scan_partial', 'coverage': {'mode': 'partial_scan', 'total_chunks': len(chunks), 'selected_chunks': 1, 'sources': [source]}}
                library.update_job(task['id'], status='completed', result=result)
                self.assertEqual(details()['index_status']['graph'], 'partial')
                # 全文处理完成但没有提取到关系时，也应显示已建立。
                with corpus.db:
                    corpus.db.execute('DELETE FROM graph_facts')
                result.update(status='scan_complete', coverage={'mode': 'full_scan', 'total_chunks': len(chunks), 'selected_chunks': len(chunks), 'sources': [source]})
                library.update_job(task['id'], result=result)
                self.assertEqual(details()['index_status']['graph'], 'ready')
                result['coverage']['sources'] = [{**source, 'sha256': '另一版本原文'}]
                library.update_job(task['id'], result=result)
                self.assertEqual(details()['index_status']['graph'], 'missing')
                with patch('reader_agent.library.configured_embedding_identity', return_value='模型乙'):
                    self.assertEqual(details()['index_status']['index'], 'missing')
            finally:
                corpus.close()

    def test_jobs_retry_and_download_stay_inside_library(self):
        book = self.upload("测试书", "原文内容。")
        self.assertEqual(self.client.post(f"/api/books/{book}/jobs", json={"kind": "unknown"}).status_code, 422)
        self.assertEqual(self.client.post(f"/api/books/{book}/jobs", json={"kind": "ask"}).status_code, 400)
        failed = self.app.state.library.create_job("ask", {"question": "原文", "extractive": True}, book)
        self.app.state.library.update_job(failed["id"], status="interrupted")
        retried = self.client.post("/api/jobs/" + failed["id"] + "/retry")
        self.assertEqual(retried.status_code, 202)
        self.assertEqual(self.app.state.library.job(failed["id"])["status"], "retried")
        duplicate_retry = self.client.post("/api/jobs/" + failed["id"] + "/retry").json()
        self.assertEqual(duplicate_retry["id"], retried.json()["id"])
        job = self.wait_job(retried.json())
        self.assertEqual(job["status"], "completed")
        self.assertEqual(self.client.post("/api/jobs/" + job["id"] + "/retry").status_code, 400)
        outside = self.root / "private.md"
        outside.write_text("不能由报告接口下载")
        self.app.state.library.update_job(job["id"], result={"report": str(outside)})
        self.assertEqual(self.client.get("/api/jobs/" + job["id"] + "/download/report").status_code, 404)
        inside = self.root / "library" / "report.md"
        inside.write_text("带引用的测试报告", encoding="utf-8")
        self.app.state.library.update_job(job["id"], result={"report": str(inside)})
        self.assertEqual(self.client.get("/api/jobs/" + job["id"] + "/download/report").text, "带引用的测试报告")

    def test_conversations_isolate_records_and_validate_book_ownership(self):
        first = self.upload('会话甲书', '第一章\n林舟用铜钥匙打开了铁门。')
        second = self.upload('会话乙书', '第一章\n另一份原文。')
        sessions = [self.client.post(f'/api/books/{first}/conversations', json={}).json() for _ in range(2)]
        for session in sessions:
            job = self.client.post(f'/api/books/{first}/jobs', json={'kind': 'ask', 'question': '铜钥匙', 'extractive': True, 'conversation_id': session['id']}).json()
            self.assertEqual(self.wait_job(job)['status'], 'completed')
        for session in sessions:
            jobs = self.client.get(f"/api/jobs?book_id={first}&conversation_id={session['id']}").json()
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]['conversation_id'], session['id'])
        self.assertEqual(self.client.get(f'/api/books/{second}/conversations').json(), [])
        renamed = self.client.patch('/api/conversations/' + sessions[0]['id'], json={'title': '人物讨论'}).json()
        self.assertEqual(renamed['title'], '人物讨论')
        self.assertNotEqual(self.app.state.library.conversation(sessions[1]['id'])['title'], '人物讨论')
        self.assertEqual(self.client.get(f"/api/jobs?book_id={second}&conversation_id={sessions[0]['id']}").status_code, 400)
        self.assertEqual(self.client.post(f'/api/books/{second}/jobs', json={'kind': 'ask', 'question': '问题', 'conversation_id': sessions[0]['id']}).status_code, 400)
        self.assertEqual(self.client.post(f'/api/books/{first}/jobs', json={'kind': 'index', 'conversation_id': sessions[0]['id']}).status_code, 400)
        self.assertEqual(self.client.get('/api/jobs?conversation_id=不存在').status_code, 404)

    def test_service_does_not_load_or_pass_any_history(self):
        book = self.upload('独立问题测试', '明确的原文。')
        library = self.app.state.library
        session = library.create_conversation(book)
        old = library.create_job('ask', {'question': '旧问题中的主体', 'conversation_id': session['id']}, book)
        library.update_job(old['id'], status='completed', result={'claims': [{'text': '旧回答里的事实'}]})
        backend = AsyncMock()
        ask = AsyncMock(return_value={'status': 'unclear', 'claims': []})
        with patch('reader_agent.library.CompatibleBackend.from_env', return_value=backend), \
             patch('reader_agent.library.make_retriever', return_value=None), \
             patch('reader_agent.library.ReadingAgent') as agent:
            agent.return_value.ask = ask
            job = self.client.post(f'/api/books/{book}/jobs', json={'kind': 'ask',
                'question': '他在哪里', 'conversation_id': session['id']}).json()
            self.assertEqual(self.wait_job(job)['status'], 'completed')
        self.assertEqual(ask.await_args.args, ('他在哪里',))
        self.assertEqual(set(ask.await_args.kwargs), {'sources'})
        self.assertEqual(len(self.client.get(f"/api/jobs?book_id={book}&conversation_id={session['id']}").json()), 2)

    def test_validation_retry_reuses_current_original_without_reranking(self):
        book = self.upload('复核测试', '第一章\n林舟打开了门。')
        library = self.app.state.library
        corpus = Corpus(library.directory)
        try:
            chunks = corpus.all_chunks([book])
            sources = corpus.sources()
        finally:
            corpus.close()
        old = library.create_job('ask', {'question': '林舟做了什么？', 'retrieval': 'hybrid'}, book)
        library.update_job(old['id'], status='completed', result={'validation_error': '测试失败',
            'coverage': {'sources': sources, 'required_entities': ['林舟'],
                'context_trace': [{'selected_chunk_ids': [chunks[0].id]}]}})
        ask = AsyncMock(return_value={'status': 'answered', 'claims': []})
        with patch('reader_agent.library.CompatibleBackend.from_env', return_value=AsyncMock()), \
             patch('reader_agent.library.make_retriever') as retriever, \
             patch('reader_agent.library.ReadingAgent') as agent:
            agent.return_value.ask = ask
            response = self.client.post('/api/jobs/' + old['id'] + '/retry')
            self.assertEqual(response.status_code, 202, response.text)
            self.assertEqual(self.wait_job(response.json())['status'], 'completed')
            retriever.assert_not_called()
        self.assertEqual(ask.await_args.args, ('林舟做了什么？',))
        self.assertEqual(ask.await_args.kwargs['evidence_ids'], [chunks[0].id])
        self.assertEqual(ask.await_args.kwargs['required_entities'], ['林舟'])
        self.assertNotIn('history', ask.await_args.kwargs)
        self.assertEqual(self.client.post('/api/jobs/' + old['id'] + '/retry').json()['id'], response.json()['id'])

    def test_html_versions_match_assets_and_cache_policy(self):
        import hashlib
        response = self.client.get('/')
        self.assertEqual(response.headers['cache-control'], 'no-store')
        for name in ('app.js', 'styles.css'):
            asset = self.client.get('/assets/' + name)
            digest = hashlib.sha256(asset.content).hexdigest()[:12]
            self.assertIn(name + '?v=' + digest, response.text)
            self.assertEqual(asset.headers['cache-control'], 'no-cache')


class RecoveryTest(unittest.TestCase):
    def test_old_questions_migrate_once_without_losing_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / '旧书.txt'
            original.write_text('原文。', encoding='utf-8')
            corpus = Corpus(root / 'library', create=True)
            book_id = corpus.ingest(original)['id']
            corpus.close()
            library = Library(root / 'library')
            job = library.create_job('ask', {'question': '旧问题'}, book_id)
            library.update_job(job['id'], status='completed', result={'claims': []})
            with library.connection() as db:
                db.execute('UPDATE jobs SET conversation_id=NULL')
                db.execute('DELETE FROM conversations')
            migrated = Library(root / 'library')
            session = migrated.conversations(book_id)[0]
            self.assertEqual(session['title'], '历史会话')
            self.assertEqual(migrated.job(job['id'])['conversation_id'], session['id'])
            self.assertEqual(migrated.job(job['id'])['result'], {'claims': []})
            again = Library(root / 'library')
            self.assertEqual(len(again.conversations(book_id)), 1)
            self.assertEqual(again.job(job['id'])['conversation_id'], session['id'])
    def test_duplicate_active_index_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            library = Library(temporary)
            service = JobService(library)
            try:
                active = library.create_job("index", {"batch_size": 16}, "测试书籍ID")
                repeated = service.submit("index", {"batch_size": 32}, "测试书籍ID")
                self.assertEqual(repeated["id"], active["id"])
                self.assertTrue(repeated["reused_job"])
                self.assertEqual(len(library.jobs()), 1)
            finally:
                service.close()

    def test_restart_marks_unfinished_jobs_as_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            library = Library(temporary)
            queued = library.create_job("index", {})
            running = library.create_job("scan", {})
            library.update_job(running["id"], status="running")
            service = JobService(library)
            try:
                self.assertEqual(library.job(queued["id"])["status"], "interrupted")
                self.assertEqual(library.job(running["id"])["status"], "interrupted")
            finally:
                service.close()
