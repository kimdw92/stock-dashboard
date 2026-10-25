# PowerShell script to create Desktop Shortcuts for RS Dashboard
$WshShell = New-Object -ComObject WScript.Shell
$Desktop = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)

$WorkspaceDir = $PSScriptRoot
$RunBat = Join-Path $WorkspaceDir "run.bat"
$PythonExe = Join-Path $WorkspaceDir ".venv\Scripts\python.exe"

# 1. 메인 런처 바로가기 생성
$ShortcutPath = Join-Path $Desktop "RS Dashboard.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $RunBat
$Shortcut.WorkingDirectory = $WorkspaceDir
$Shortcut.Description = "RS Dashboard 실행 런처"
if (Test-Path $PythonExe) {
    $Shortcut.IconLocation = "$PythonExe,0"
}
$Shortcut.Save()

Write-Host " [성공] 바탕화면에 바로가기가 생성되었습니다: $ShortcutPath" -ForegroundColor Green
Write-Host " 이제 바탕화면의 'RS Dashboard' 아이콘을 더블클릭하면 바로 실행됩니다." -ForegroundColor Cyan
