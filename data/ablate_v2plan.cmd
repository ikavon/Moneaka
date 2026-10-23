@echo off
cd /d D:\Python_projects\Moneaka
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=src
C:\Users\FYK\anaconda3\envs\moneaka\python.exe -u tools\pick_checkpoint.py --adapter-dir data\adapters\run8 --base data\models\Qwen3-1.7B --pairs data\pairs\split_holdout210_v2.jsonl --final-only --temperature 0 --tag g210v2 --workers 2
