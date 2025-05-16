#!/usr/bin/env bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# start MCP tool server in background
python server.py &
SERVER_PID=$!
echo "✅ MCP server running (PID $SERVER_PID)"

# optional: export OPENAI_API_KEY before launching for remote backend
# export OPENAI_API_KEY="sk-..."
# export LLM_BACKEND=openai      # comment out to use local model

python app.py

# stop server when UI closes
kill $SERVER_PID
