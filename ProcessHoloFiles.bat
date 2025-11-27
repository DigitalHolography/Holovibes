@echo off
setlocal

rem Locate the PowerShell script next to this batch file.
set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%ProcessHoloFiles.ps1"

if not exist "%PS_SCRIPT%" (
    echo [ProcessHoloFiles] Unable to locate "%PS_SCRIPT%".
    exit /b 1
)

rem Prefer the system PowerShell if available, otherwise fall back to PATH.
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" (
    set "POWERSHELL_EXE=powershell.exe"
)

"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %*
set "ERR=%ERRORLEVEL%"
endlocal & exit /b %ERR%
