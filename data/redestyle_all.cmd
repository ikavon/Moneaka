@echo off
cd /d D:\Python_projects\Moneaka
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=src
C:\Users\FYK\anaconda3\envs\moneaka\python.exe -u tools\redestyle.py --src data\pairs\tagged_official.jsonl --out data\pairs\tagged_official_v2.jsonl --workers 4
C:\Users\FYK\anaconda3\envs\moneaka\python.exe -u tools\redestyle.py --src data\pairs\tagged_ds.jsonl --out data\pairs\tagged_ds_v2.jsonl --workers 4
