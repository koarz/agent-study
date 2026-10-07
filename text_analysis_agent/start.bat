@echo off
rem 从项目目录启动前后端，优先使用本项目的独立虚拟环境。
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -m reader_agent serve %*
) else (
    py -3 -m reader_agent serve %*
)
