@echo off
cd /d D:\Python_projects\Moneaka
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=src
C:\Users\FYK\anaconda3\envs\moneaka\python.exe -u tools\run_dpo_phase.py --only dpo-b
C:\Users\FYK\anaconda3\envs\moneaka\python.exe -u tools\run_dpo_phase.py --only dpo-d
