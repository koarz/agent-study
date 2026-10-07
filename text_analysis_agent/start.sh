#!/usr/bin/env bash
# 从项目目录启动前后端，优先使用本项目的独立虚拟环境。
set -e
cd -- "$(dirname -- "$0")"
if [ -x .venv/bin/python ]; then
    reader_python=.venv/bin/python
else
    reader_python=python3
fi
# 启动时按当前内容更新脚本与样式版本，浏览器会重新获取改变的文件。
"$reader_python" tools/version_assets.py
exec "$reader_python" -m reader_agent serve "$@"
