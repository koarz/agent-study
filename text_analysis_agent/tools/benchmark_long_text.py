"""用合成长文本验证流式导入、末尾线索检索和内存占用，不调用模型。"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reader_agent.corpus import Corpus
from reader_agent.scan import scan_plan


def benchmark(size: int, output: Path | None):
    if size < 10000:
        raise ValueError("测试文本至少一万字")
    with tempfile.TemporaryDirectory(prefix="reader-benchmark-") as temporary:
        directory = Path(temporary)
        novel = directory / "合成长篇.txt"
        clue = "结尾的唯一测试线索：顾青用紫星钥匙打开了地下档案室。"
        remaining = size - len(clue)
        paragraph = "晨雨打在窗沿。顾青穿过旧街，看见邮局的灯还亮着。他记下当天的见闻，然后继续赶路。\n"
        with novel.open("w", encoding="utf-8", newline="") as writer:
            chapter = 1
            while remaining:
                piece = (f"第{chapter}章 旧城记录\n" + paragraph * 25)[:remaining]
                writer.write(piece)
                remaining -= len(piece)
                chapter += 1
            writer.write(clue)
        corpus = Corpus(directory / "corpus", create=True)
        try:
            started = time.perf_counter()
            imported = corpus.ingest(novel)
            import_seconds = time.perf_counter() - started
            ids = corpus.select_sources()
            started = time.perf_counter()
            hits = corpus.search("紫星钥匙 地下档案室", ids, limit=8)
            hit = next((chunk for chunk in hits if clue in chunk.text), None)
            if hit is None:
                raise AssertionError("未找回末尾的测试线索")
            citation = corpus.citation(hit, clue)
            query_seconds = time.perf_counter() - started
            if citation["end"] != size or imported["chars"] != size:
                raise AssertionError("原文字数或末尾引用偏移不准确")
            if corpus._verified:
                raise AssertionError("新快照不应整本载入内存")
            plan = scan_plan(corpus, ids, 12000)
            result = {"dataset": "合成中文文本，非真实小说或语义模型评测", "chars": size,
                      "chunks": imported["chunks"], "import_seconds": round(import_seconds, 3),
                      "last_clue_query_seconds": round(query_seconds, 3),
                      "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
                      "storage_mb": round(sum(p.stat().st_size for p in corpus.directory.rglob('*') if p.is_file()) / 1048576, 2),
                      "last_clue_exactly_found": True, "citation_end": citation["end"],
                      "whole_source_loaded": False, "scan_plan": plan,
                      "paid_model_calls": 0}
        finally:
            corpus.close()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chars", type=int, default=1000000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    benchmark(args.chars, args.output)
