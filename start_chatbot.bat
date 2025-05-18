@echo off
echo Starting MCP Chatbot...
set LOG_LEVEL=INFO
:retry
python -u app.py
if %ERRORLEVEL% neq 0 (
    echo Application crashed with error code %ERRORLEVEL%. Restarting in 5 seconds...
    timeout /t 5
    goto retry
)
