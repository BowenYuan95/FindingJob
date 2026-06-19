Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "C:\project\FindingJob"
sh.Run "cmd /c ""lms server start && lms load text-embedding-nomic-embed-text-v1.5 --gpu max && lms load qwen/qwen3.5-9b --gpu max --context-length 16384""", 0, True
sh.Run "cmd /c ""cd /d C:\project\FindingJob && py launcher.py""", 0, False