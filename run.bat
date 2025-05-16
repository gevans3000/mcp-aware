@echo off

:: Create and activate virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Start MCP server in background
echo Starting MCP server in background...
start "MCP Server" cmd /c "call venv\Scripts\activate.bat && python server.py"
echo MCP server running on http://localhost:6789

:: Set environment variables for OpenAI (if needed)
:: set OPENAI_API_KEY=your_api_key_here
:: set LLM_BACKEND=openai

:: Start the Gradio web interface
echo Starting Gradio web interface...
echo Open http://localhost:7860 in your browser
python app.py

:: Cleanup
echo Shutting down...
taskkill /f /im python.exe >nul 2>&1
