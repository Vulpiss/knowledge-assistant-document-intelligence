@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0SCAL_I_ROZPAKUJ.ps1"
if errorlevel 1 (
    echo.
    echo Operacja nie powiodla sie. Przeczytaj komunikat powyzej.
    pause
    exit /b 1
)
echo.
echo Paczka jest gotowa.
pause
