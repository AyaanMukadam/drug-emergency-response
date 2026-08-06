@echo off
title Drug Emergency Response — Live Server
color 0B
echo.
echo  =====================================================
echo   Drug Emergency Response Prediction - LIVE SERVER
echo  =====================================================
echo.

:: Start Flask in background
start "Flask Server" cmd /k "cd /d "%~dp0" && set PYTHONIOENCODING=utf-8 && python app.py"

:: Wait for Flask to boot
timeout /t 4 /nobreak >nul

:: Start Cloudflare Tunnel
echo  [*] Starting public tunnel...
echo  [*] Your public URL will appear below:
echo  [*] Share it with anyone in the world!
echo.
"%~dp0cloudflared.exe" tunnel --url http://localhost:5000

pause
