@echo off
echo [%date% %time%] Starting MCP Chatbot...

echo Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Installing/updating dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

:start
    echo [%date% %time%] Starting application on http://localhost:7861
    python -u app.py
    
    if %ERRORLEVEL% neq 0 (
        echo [%date% %time%] Application crashed with error code %ERRORLEVEL%
        echo [%date% %time%] Restarting in 5 seconds...
        timeout /t 5 >nul
        goto start
    )

echo [%date% %time%] Application closed successfully
pause
