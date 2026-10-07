"""使用模拟任务验证关系索引恢复、暂停和故障分类，不调用任何模型接口。"""

import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx

from reader_agent.library import JobService, Library
from reader_agent.llm import ModelOutputError
from reader_agent.api_calls import APIServiceError


class GraphRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.library = Library(self.temp.name)
        self.services = []

    def tearDown(self):
        for service in reversed(self.services):
            if not service.stopping.is_set():
                service.close()
        self.temp.cleanup()

    def service(self, **kwargs):
        service = JobService(self.library, retry_delays=(.04, .06), recovery_poll_interval=.01, **kwargs)
        self.services.append(service)
        return service

    def wait(self, job_id, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = self.library.job(job_id)
            if predicate(job):
                return job
            time.sleep(.01)
        self.fail('任务未及时达到预期状态：' + str(self.library.job(job_id)))

    def test_restart_only_resumes_latest_graph_with_same_id_and_checkpoint(self):
        old = self.library.create_job('graph', {}, '甲')
        latest = self.library.create_job('graph', {'batch_chars': 6000}, '甲')
        self.library.update_job(latest['id'], status='running', progress_detail={'current': 7})
        paused = self.library.create_job('graph', {}, '乙')
        self.library.update_job(paused['id'], status='paused')
        ask = self.library.create_job('index', {}, '丙')
        calls = []
        async def execute(service, job):
            calls.append((job['id'], job['payload'], job['progress_detail'].get('current')))
            return {'status': 'scan_complete'}
        with patch.object(JobService, '_execute', execute):
            service = self.service()
            result = self.wait(latest['id'], lambda j: j['status'] == 'completed')
        self.assertEqual(calls, [(latest['id'], {'batch_chars': 6000}, 7)])
        self.assertEqual(result['auto_recovery']['restarts'], 1)
        self.assertEqual(self.library.job(old['id'])['status'], 'interrupted')
        self.assertEqual(self.library.job(paused['id'])['status'], 'paused')
        self.assertEqual(self.library.job(ask['id'])['status'], 'interrupted')
        self.assertEqual(len(self.library.jobs()), 4)

    def test_temporary_failure_automatically_retries_without_new_job(self):
        calls = []
        async def execute(service, job):
            calls.append(job['id'])
            if len(calls) < 3:
                raise ConnectionError('模拟断网')
            return {'status': 'scan_complete'}
        with patch.object(JobService, '_execute', execute):
            service = self.service()
            job = service.submit('graph', {}, '甲')
            result = self.wait(job['id'], lambda j: j['status'] == 'completed')
        self.assertEqual(calls, [job['id']] * 3)
        self.assertEqual(len(self.library.jobs()), 1)
        self.assertFalse(result['auto_recovery']['waiting'])

    def test_pause_waiting_job_survives_restart_and_resumes_same_job(self):
        async def fail(service, job):
            raise ConnectionError('模拟断网')
        with patch.object(JobService, '_execute', fail):
            service = self.service()
            # 保留较长等待，测试暂停会取消待恢复任务。
            service.retry_delays = (30,)
            job = service.submit('graph', {}, '甲')
            self.wait(job['id'], lambda j: j['auto_recovery'].get('waiting'))
            service.pause(job['id'])
            service.close()
        calls = []
        async def succeed(service, job):
            calls.append(job['id']);return {'status': 'scan_complete'}
        with patch.object(JobService, '_execute', succeed):
            restarted = self.service()
            self.assertEqual(self.library.job(job['id'])['status'], 'paused')
            self.assertEqual(calls, [])
            resumed = restarted.resume(job['id'])
            self.wait(job['id'], lambda j: j['status'] == 'completed')
        self.assertEqual(resumed['id'], job['id'])
        self.assertEqual(calls, [job['id']])
        self.assertEqual(len(self.library.jobs()), 1)

    def test_immediate_resume_during_inflight_pause_does_not_duplicate_worker(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        async def execute(service, job):
            calls.append(job['id'])
            if len(calls) == 1:
                entered.set();release.wait(3)
                service.progress(job['id'], '当前批次结束')
            return {'status': 'scan_complete'}
        with patch.object(JobService, '_execute', execute):
            service = self.service()
            job = service.submit('graph', {}, '甲')
            try:
                self.assertTrue(entered.wait(1))
                service.pause(job['id'])
                service.resume(job['id']);service.resume(job['id'])
            finally:
                release.set()
            self.wait(job['id'], lambda j: j['status'] == 'completed')
        self.assertEqual(calls, [job['id']] * 2)
        self.assertEqual(len(self.library.jobs()), 1)

    def test_invalid_json_stops_after_three_automatic_recoveries(self):
        calls = []
        async def execute(service, job):
            calls.append(job['id']);raise ModelOutputError('模型未返回有效 JSON')
        with patch.object(JobService, '_execute', execute):
            service = self.service()
            job = service.submit('graph', {}, '甲')
            self.wait(job['id'], lambda j: j['status'] == 'failed')
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(self.library.jobs()), 1)

    def test_quota_and_authentication_errors_are_not_retried(self):
        for status, payload in [(401, {}), (403, {}), (429, {'error': {'code': 'insufficient_quota'}})]:
            raw = httpx.HTTPStatusError('模拟接口失败', request=httpx.Request('POST', 'https://example.invalid'),
                response=httpx.Response(status, json=payload))
            wrapped = APIServiceError('接口拒绝请求');wrapped.__cause__ = raw
            self.assertIsNone(JobService.recovery_kind(wrapped))
        self.assertIsNone(JobService.recovery_kind(ValueError('本地配置无效')))
        self.assertEqual(JobService.recovery_kind(ConnectionError()), 'temporary')

    def test_restart_preserves_pending_recovery_deadline(self):
        job = self.library.create_job('graph', {}, '甲')
        deadline = time.time() + 100
        self.library.update_job(job['id'], status='interrupted',
            auto_recovery={'waiting': True, 'retry_at': deadline, 'attempt': 2})
        calls = []
        async def execute(service, job):
            calls.append(job['id']);return {'status': 'scan_complete'}
        with patch.object(JobService, '_execute', execute):
            service = self.service()
            restored = self.library.job(job['id'])
            self.assertEqual(restored['status'], 'queued')
            self.assertEqual(restored['auto_recovery']['retry_at'], deadline)
            self.assertEqual(calls, [])
            service.pause(job['id'])
