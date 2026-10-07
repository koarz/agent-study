"""验证缓存只保存散列、按原文版本区分并限制保留时间和条数。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reader_agent.corpus import digest
from reader_agent.rerank_cache import RerankCache


class CacheTest(unittest.TestCase):
    def test_content_hash_version_and_no_plaintext(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.sqlite3'
            cache = RerankCache(path)
            question, text = '林舟在哪里得到钥匙', '林舟在码头得到钥匙。'
            cache.put('模型一', digest(question), {digest(text): 0.8})
            self.assertEqual(cache.get('模型一', digest(question), [digest(text)]), {digest(text): 0.8})
            self.assertEqual(cache.get('模型二', digest(question), [digest(text)]), {})
            self.assertEqual(cache.get('模型一', digest(question), [digest(text + '新版本')]), {})
            self.assertNotIn(question.encode(), path.read_bytes())
            self.assertNotIn(text.encode(), path.read_bytes())

    def test_expiration_and_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = RerankCache(Path(directory) / 'cache.sqlite3', ttl=10, maximum=2)
            with patch('reader_agent.rerank_cache.time.time', return_value=100):
                cache.put('模型', '问题散列', {'旧段': 0.8})
            with patch('reader_agent.rerank_cache.time.time', return_value=111):
                self.assertEqual(cache.get('模型', '问题散列', ['旧段']), {})
                cache.put('模型', '问题散列', {'甲': 0.1, '乙': 0.2, '丙': 0.3})
                self.assertEqual(len(cache.get('模型', '问题散列', ['旧段', '甲', '乙', '丙'])), 2)
