@echo off
cd /d C:\project\FindingJob
echo === Loading LM Studio models ===
call lms server start
call lms load text-embedding-nomic-embed-text-v1.5 --gpu max
call lms load qwen/qwen3.5-9b --gpu max --context-length 16384
echo === Launching JobFinder desktop window ===
py launcher.py
