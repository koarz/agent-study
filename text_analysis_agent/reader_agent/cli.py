from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

from .agent import ReadingAgent
from .corpus import Corpus
from .llm import CompatibleBackend
from .scan import FullScanner, scan_plan
from .retrieval import make_retriever
from .graph import EvidenceGraph
from .logging_utils import Observability


def render(result: dict) -> str:
    statuses = {"answered": "有原文依据", "unclear": "信息不明确 / 依据不足",
                "not_found": "未找到回答依据", "extractive": "原文检索",
                "scan_complete": "全文扫描完成", "scan_partial": "全文扫描未完成"}
    coverage = result["coverage"]
    scope = "完整原文已纳入上下文" if coverage["mode"] == "full_context" else "仅检索片段，不能据此断言全文没有相关内容"
    if coverage["mode"] in {"full_scan", "partial_scan"}:
        scope = "逐批核对原文，扫描覆盖不等于模型提取无遗漏"
    lines = [f"状态：{statuses[result['status']]}", result["notice"],
             f"范围：{scope}（{coverage['selected_chunks']}/{coverage['total_chunks']} 个段落）",
             "原文：" + "、".join(s["name"] for s in coverage["sources"])]
    if "report" in result:
        lines.extend([f"已保存证据：{result['finding_count']} 条", f"报告：{result['report']}",
                      f"结构化证据：{result['evidence_file']}"])
    for index, claim in enumerate(result["claims"], 1):
        lines.extend(["", f"{index}. {claim['text']}"])
        for citation in claim["citations"]:
            lines.append(f"   [{citation['source']} · {citation['chapter']} · 第 {citation['line_start']}—{citation['line_end']} 行]")
            lines.extend("   > " + line for line in citation["quote"].splitlines())
    for chunk in result.get("evidence", []):
        lines.extend(["", f"[{chunk['source']} · {chunk['chapter']} · 第 {chunk['line_start']}—{chunk['line_end']} 行 · {chunk['id']}]"])
        lines.extend("> " + line for line in chunk["text"].splitlines())
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="依据原文回答文本 / 小说问题的独立 Agent")
    parser.add_argument("--corpus", type=Path, default=Path("data/default"), help="独立知识库目录")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="保存并索引 TXT / Markdown 原文快照，不调用模型")
    ingest.add_argument("file", type=Path)
    ingest.add_argument("--name", help="原文名称；同名但内容不同的版本不会覆盖")
    ingest.add_argument("--encoding", default="utf-8-sig", help="解码编码，如 utf-8-sig 或 gb18030")
    ingest.add_argument("--chunk-chars", type=int, default=2000, help="流式原文分块大小")
    ingest.add_argument("--overlap", type=int, default=200, help="相邻段落重叠字符数")
    sub.add_parser("list", help="列出知识库原文")
    serve = sub.add_parser("serve", help="启动服务后端和浏览器阅读工作台")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    scan = sub.add_parser("scan", help="针对问题逐批扫描全文、保存证据并支持断点恢复")
    scan.add_argument("question")
    scan.add_argument("--source", action="append")
    scan.add_argument("--batch-chars", type=int, default=12000)
    scan.add_argument("--dry-run", action="store_true", help="只查看批次数和调用量，不调用模型")
    scan.add_argument("--json", action="store_true")
    index = sub.add_parser("index", help="分批构建可恢复的语义向量索引")
    index.add_argument("--source", action="append")
    index.add_argument("--batch-size", type=int, default=16)
    index.add_argument("--dry-run", action="store_true")
    index.add_argument("--repair", action="store_true", help="清除当前模型的索引进度并重新写入所有原文向量")
    graph_build = sub.add_parser("build-graph", help="扫描原文建立带引用的实体关系和事件图谱")
    graph_build.add_argument("--source", action="append")
    graph_build.add_argument("--batch-chars", type=int, default=6000)
    graph_build.add_argument("--dry-run", action="store_true")
    graph_build.add_argument("--json", action="store_true")
    graph_view = sub.add_parser("graph", help="查看实体的有依据关系，保留原文引用")
    graph_view.add_argument("entity")
    graph_view.add_argument("--source", action="append")
    graph_view.add_argument("--limit", type=int, default=100)
    for command, help_text in [("ask", "提出一个问题"), ("chat", "连续提问，每个问题独立核对原文")]:
        item = sub.add_parser(command, help=help_text)
        if command == "ask":
            item.add_argument("question")
        item.add_argument("--source", action="append", help="原文名称，可重复；多份原文时必须指定")
        item.add_argument("--extractive", action="store_true", help="只检索并显示原文，不调用模型")
        item.add_argument("--json", action="store_true", help="输出结构化 JSON，含引用偏移量和校验信息")
        item.add_argument("--max-context-chars", type=int, default=16000)
        item.add_argument("--top-k", type=int, default=6)
        item.add_argument("--search-rounds", type=int, choices=(1, 2, 3), default=2)
        item.add_argument("--retrieval", choices=("hybrid", "lexical"), help="指定检索方式；默认使用 .env 配置的混合检索")
    return parser


async def run_questions(args, corpus: Corpus):
    corpus.select_sources(args.source)
    backend = None if args.extractive else CompatibleBackend.from_env()
    retriever = verifier = None
    try:
        if not args.extractive:
            retriever = make_retriever(corpus, mode=args.retrieval)
            verifier = CompatibleBackend.from_env(verifier=True)
        agent = ReadingAgent(corpus, backend, max_context_chars=args.max_context_chars,
                             top_k=args.top_k, search_rounds=args.search_rounds,
                             retriever=retriever, verifier=verifier)

        async def answer(question):
            result = await agent.ask(question, sources=args.source, extractive=args.extractive)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else render(result))

        if args.command == "ask":
            await answer(args.question)
        else:
            print("输入问题（请写明人物或事件）；/exit 退出。每轮都重新核对原文。", file=sys.stderr)
            while True:
                try:
                    question = input("问题> ").strip()
                except EOFError:
                    break
                if question in {"/exit", "/quit"}:
                    break
                if question:
                    await answer(question)
    finally:
        if retriever is not None:
            await retriever.close()
        if verifier is not None:
            await verifier.close()
        if backend is not None:
            await backend.close()


async def run_scan(args, corpus: Corpus):
    source_ids = corpus.select_sources(args.source)
    if args.dry_run:
        plan = scan_plan(corpus, source_ids, args.batch_chars)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    backend = CompatibleBackend.from_env()
    verifier = None
    try:
        verifier = CompatibleBackend.from_env(verifier=True)
        scanner = FullScanner(corpus, backend, batch_chars=args.batch_chars,
                              verifier=verifier,
                              progress=lambda message: print(message, file=sys.stderr, flush=True))
        result = await scanner.scan(args.question, sources=args.source)
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else render(result))
    finally:
        if verifier is not None:
            await verifier.close()
        await backend.close()


async def run_index(args, corpus: Corpus):
    source_ids = corpus.select_sources(args.source)
    if not 1 <= args.batch_size <= 128:
        raise ValueError("嵌入批次大小须为 1 到 128")
    if args.dry_run:
        stats = corpus.stats(source_ids)
        print(json.dumps({**stats, "embedding_calls_if_all_new": (stats["chunks"] + args.batch_size - 1) // args.batch_size,
                          "notice": "调用量按全部新建估计；实际会复用已建立的索引。"}, ensure_ascii=False, indent=2))
        return
    retriever = make_retriever(corpus, for_index=True)
    try:
        if args.repair:
            retriever.semantic.repair()
        result = await retriever.semantic.build(source_ids, batch_size=args.batch_size,
                        progress=lambda message: print(message, file=sys.stderr, flush=True))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await retriever.close()


async def run_graph_build(args, corpus: Corpus):
    source_ids = corpus.select_sources(args.source)
    if args.dry_run:
        print(json.dumps(scan_plan(corpus, source_ids, args.batch_chars), ensure_ascii=False, indent=2))
        return
    backend = CompatibleBackend.from_env()
    verifier = None
    try:
        verifier = CompatibleBackend.from_env(verifier=True)
        result = await EvidenceGraph(corpus).build(backend, sources=args.source, verifier=verifier,
                          batch_chars=args.batch_chars,
                          progress=lambda message: print(message, file=sys.stderr, flush=True))
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else render(result))
    finally:
        if verifier is not None:
            await verifier.close()
        await backend.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = None
    logs = None
    binding = None
    try:
        if args.command == "serve":
            import uvicorn
            from .server import create_app
            app = create_app(args.corpus)
            # 控制台保留服务日志；访问文件使用应用日志，避免记录查询参数。
            from uvicorn.config import LOGGING_CONFIG
            from logging.config import dictConfig
            dictConfig(LOGGING_CONFIG)
            app.state.logs.capture_server()
            uvicorn.run(app, host=args.host, port=args.port, log_level="info", log_config=None, access_log=False)
            return 0
        logs = Observability(args.corpus / "logs")
        binding = logs.bind(command=args.command)
        binding.__enter__()
        logs.event("命令开始")
        corpus = Corpus(args.corpus, create=args.command == "ingest")
        if args.command == "ingest":
            result = corpus.ingest(args.file, name=args.name, encoding=args.encoding,
                                   chunk_chars=args.chunk_chars, overlap=args.overlap)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "list":
            print(json.dumps(corpus.sources(), ensure_ascii=False, indent=2))
        elif args.command == "scan":
            asyncio.run(run_scan(args, corpus))
        elif args.command == "index":
            asyncio.run(run_index(args, corpus))
        elif args.command == "build-graph":
            asyncio.run(run_graph_build(args, corpus))
        elif args.command == "graph":
            corpus.begin_read()
            print(json.dumps(EvidenceGraph(corpus).facts(args.entity, corpus.select_sources(args.source), limit=args.limit),
                             ensure_ascii=False, indent=2))
        else:
            asyncio.run(run_questions(args, corpus))
        return 0
    except KeyboardInterrupt:
        print("已结束。", file=sys.stderr)
        return 130
    except (ValueError, OSError, LookupError, sqlite3.Error, ImportError) as exc:
        if logs is not None:
            logs.error("命令执行失败", exc)
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if logs is not None:
            logs.error("命令执行失败", exc)
        # 模型服务错误可能含请求地址或敏感响应，终端只显示错误类型。
        print(f"模型调用失败（{type(exc).__name__}），请检查模型配置及服务状态；本次没有输出未验证结论。", file=sys.stderr)
        return 1
    finally:
        if corpus is not None:
            corpus.close()
        if logs is not None:
            logs.event("命令结束")
            binding.__exit__(None, None, None)
            logs.close()
