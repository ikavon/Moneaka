@echo off
rem 双击打开命令行版聊天测试
cd /d %~dp0
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=src
"C:\Users\FYK\anaconda3\envs\moneaka\python.exe" tools\chat_cli.py %*
