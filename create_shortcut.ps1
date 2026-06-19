# create_shortcut.ps1 — Create a desktop shortcut for run_jobfinder.bat
# Run once:  powershell -ExecutionPolicy Bypass -File create_shortcut.ps1

$projectDir = "C:\project\FindingJob"
$batPath    = Join-Path $projectDir "run_jobfinder.bat"
$desktop    = [Environment]::GetFolderPath("Desktop")
$lnkPath    = Join-Path $desktop "JobFinder.lnk"

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnkPath)
$sc.TargetPath       = $batPath
$sc.WorkingDirectory = $projectDir
$sc.Description      = "Launch JobFinder dashboard"
$sc.WindowStyle      = 1
# Optional custom icon: put an .ico in the project folder and uncomment:
# $sc.IconLocation   = Join-Path $projectDir "jobfinder.ico"
$sc.Save()

Write-Host "Desktop shortcut created: $lnkPath" -ForegroundColor Green
