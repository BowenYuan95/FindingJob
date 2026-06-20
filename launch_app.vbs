Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "C:\project\FindingJob"
sh.Run "cmd /c ""lms server start""", 0, True
sh.Run "cmd /c ""cd /d C:\project\FindingJob && py launcher.py""", 0, False