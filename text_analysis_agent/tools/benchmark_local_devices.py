"""在同一原文样本上比较 CPU/GPU 重排速度，只保存数字统计，不调用模型 API。"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

from reader_agent.api_calls import api_context
from reader_agent.corpus import Corpus
from reader_agent.local_reranker import QwenLocalReranker


async def benchmark(args):
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，不能执行 GPU 基准")
    corpus = Corpus(args.corpus)
    try:
        rows = corpus.db.execute("SELECT id FROM chunks WHERE source_id=? ORDER BY id LIMIT 4", (args.source,)).fetchall()
        documents = [corpus.get_chunk(row['id']).text for row in rows]
    finally:
        corpus.close()
    if not documents:
        raise SystemExit("没有可测试的原文样本")
    question = "当前段落描述了什么事件？"
    result = {'gpu': torch.cuda.get_device_name(0), 'documents': len(documents), 'api_calls': 0, 'devices': {}}
    with api_context(lambda message: None, lambda: False):
        for device in args.devices:
            ranker = QwenLocalReranker(args.models, device=device)
            # 首次加载和一次预热单独计时，吞吐量不混入模型加载耗时。
            started = time.perf_counter()
            await ranker.score(question, [documents[0]])
            warmup = time.perf_counter() - started
            windows = sum(ranker.window_count(text) for text in documents)
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            values = await ranker.score(question, documents)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert len(values) == len(documents) and all(math.isfinite(value) and 0 <= value <= 1 for value in values)
            result['devices'][device] = {'warmup_seconds': round(warmup, 3), 'scoring_seconds': round(elapsed, 3),
                'windows': windows, 'seconds_per_window': round(elapsed / windows, 4),
                'peak_gpu_memory_mb': round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == 'cuda' else None}
    if 'cpu' in result['devices'] and 'cuda' in result['devices']:
        result['speedup'] = round(result['devices']['cpu']['scoring_seconds'] / result['devices']['cuda']['scoring_seconds'], 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=Path('data/default'))
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--source', required=True)
    parser.add_argument('--devices', nargs='+', default=['cuda', 'cpu'], choices=['cuda', 'cpu'])
    parser.add_argument('--output', type=Path, default=Path('artifacts/local-device-benchmark.json'))
    asyncio.run(benchmark(parser.parse_args()))
