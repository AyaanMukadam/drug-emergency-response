@echo off
echo =====================================================
echo   Drug Emergency Response Prediction - CIAP Project
echo =====================================================
echo.

cd /d "%~dp0"

echo [1/2] Checking if models exist...
if not exist "model\rating_model.pkl" (
    echo [!] Models not found. Training now...
    echo     This will take 1-2 minutes...
    set PYTHONIOENCODING=utf-8
    python train_model.py
    if errorlevel 1 (
        echo [ERROR] Training failed! Check the output above.
        pause
        exit /b 1
    )
    echo [OK] Models trained successfully!
    echo.
) else (
    echo [OK] Models found.
    echo.
)

echo [2/2] Starting Flask server...
echo     Open your browser at: http://127.0.0.1:5000
echo.
set FLASK_APP=app.py
set FLASK_ENV=development
set PYTHONIOENCODING=utf-8
python app.py
pause
