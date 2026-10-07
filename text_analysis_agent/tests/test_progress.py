import tempfile
import unittest
from unittest.mock import patch

from reader_agent.library import Library, JobService
from reader_agent.progress import ProgressTracker


class ProgressTest(unittest.TestCase):
    def test_resume_does_not_treat_cached_work_as_new_processing_speed(self):
        with patch("reader_agent.progress.time.perf_counter", side_effect=[0, 20, 20, 30]):
            tracker = ProgressTracker(100)
            restoring = tracker.update("复用", current=80, reused=80)
            self.assertIsNone(restoring.detail["remaining_seconds"])
            working = tracker.update("建立", current=90, reused=80)
            self.assertEqual(working.detail["remaining_seconds"], 20)
            self.assertEqual(working.detail["percent"], 90)
            finished = tracker.update("完成", current=100, reused=80)
            self.assertEqual(finished.detail["remaining_seconds"], 0)

    def test_progress_survives_database_reload_and_is_compatible_with_text_callbacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            library = Library(temporary)
            service = JobService(library)
            try:
                job = library.create_job("index", {})
                update = ProgressTracker(200).update("语义索引：已完成 100 段", current=100)
                service.progress(job["id"], update)
                self.assertTrue(update.startswith("语义索引"))
                saved = Library(temporary).job(job["id"])
                self.assertEqual(saved["progress_detail"]["current"], 100)
                self.assertEqual(saved["progress_detail"]["total"], 200)
                self.assertEqual(saved["progress_detail"]["percent"], 50)
            finally:
                service.close()
