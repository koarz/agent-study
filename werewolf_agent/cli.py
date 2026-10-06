from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from .game import GameConfig, WerewolfGame
from .llm import FatalLLMError, OpenAICompatibleBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行一局全 LLM 玩家驱动的 9 人狼人杀")
    parser.add_argument("--seed", type=int, default=None, help="身份和规则裁决的随机种子")
    parser.add_argument("--max-days", type=int, default=12, help="最大游戏天数")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="对局输出根目录")
    parser.add_argument("--model", default=None, help="覆盖 LLM_MODEL_ID")
    parser.add_argument("--base-url", default=None, help="覆盖 LLM_BASE_URL")
    parser.add_argument("--api-key", default=None, help="覆盖 LLM_API_KEY；建议使用环境变量")
    parser.add_argument("--timeout", type=int, default=None, help="单次模型请求超时秒数")
    parser.add_argument("--quiet", action="store_true", help="关闭实时控制台日志")
    return parser


async def async_main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    model = args.model or os.getenv("LLM_MODEL_ID")
    base_url = args.base_url or os.getenv("LLM_BASE_URL")
    api_key = args.api_key or os.getenv("LLM_API_KEY")
    timeout = args.timeout or int(os.getenv("LLM_TIMEOUT", "120"))
    missing = [name for name, value in (("LLM_MODEL_ID", model), ("LLM_BASE_URL", base_url), ("LLM_API_KEY", api_key)) if not value]
    if missing:
        raise SystemExit(f"缺少配置：{', '.join(missing)}。请写入 .env 或通过参数提供。")

    backend = OpenAICompatibleBackend(api_key=api_key, base_url=base_url, model=model, timeout=timeout)
    game = WerewolfGame(
        backend,
        GameConfig(
            output_dir=args.output_dir,
            seed=args.seed,
            max_days=args.max_days,
            temperature=args.temperature,
            verbose=not args.quiet,
        ),
    )
    if args.seed is not None:
        print(f"提示：当前使用固定随机种子 {args.seed}，每次身份分配将保持一致。")
    print(f"对局开始，实时记录目录：{game.run_dir}")
    try:
        result = await game.run()
    except FatalLLMError as exc:
        failure = {"error": str(exc), "type": "fatal_llm_error", "day": game.day}
        game.recorder.write_json("failure.json", failure)
        game.recorder.emit(f"[对局终止] {exc}")
        print(f"失败记录：{game.run_dir / 'failure.json'}")
        return 2
    winner = {"wolves": "狼人阵营", "village": "好人阵营", "draw": "平局"}[result["winner"]]
    print(f"对局结束，胜方：{winner}")
    print(f"完整记录：{game.run_dir}")
    return 0


def main() -> int:
    return asyncio.run(async_main())
