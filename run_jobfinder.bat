@echo off
chcp 65001 >nul
cd /d C:\project\FindingJob

echo.
echo === Starting LM Studio server and loading models ===
call lms server start
call lms unload --all
call lms load text-embedding-nomic-embed-text-v1.5 --gpu max
call lms load qwen/qwen3.5-9b --gpu max --context-length 16384

echo.
echo === Currently loaded models ===
call lms ps

echo.
echo === Starting background score backfill in a new window ===
start "JobFinder Backfill" cmd /k "cd /d C:\project\FindingJob && echo Waiting 60s before backfill... && timeout /t 60 /nobreak && py backfill_scores.py --watch"
echo.
echo === Launching dashboard, browser will open shortly ===
echo Close this window to stop the dashboard.
echo (Backfill runs in its own window; close that one to stop backfilling.)
echo.

py -m streamlit run app.py

pause