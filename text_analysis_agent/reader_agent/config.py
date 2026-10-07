"""每个任务独立读取最新配置，不修改进程级环境变量。"""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def get_config() -> dict[str, str]:
    from dotenv import dotenv_values

    values = {key: value for key, value in dotenv_values(PROJECT / ".env").items() if value is not None}
    values.update(os.environ)
    return values
