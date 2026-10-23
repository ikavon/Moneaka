@echo off
rem 双击即可打开台词测试台（Streamlit 界面）
cd /d %~dp0
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
"C:\Users\FYK\anaconda3\envs\moneaka\python.exe" -m streamlit run streamlit_app.py
