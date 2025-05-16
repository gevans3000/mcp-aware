# MCP Chatbot

A chatbot that can perform calculations using an MCP server and optionally use OpenAI's API for more advanced responses.

## Features

- **Multiple Backend Support**: Switch between OpenAI's GPT models, Google Gemini, and local models
- **MCP Integration**: Connect to MCP servers for specialized tools and resources
- **Environment Configuration**: Easy configuration through environment variables
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **Logging**: Comprehensive logging for debugging and monitoring

## Prerequisites

- Python 3.8+
- OpenAI API key (for GPT models)
- Google API key (for Gemini models)
- MCP server (for MCP tools)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mcp_chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file and update with your settings:
   ```bash
   copy .env.example .env
   ```
   Edit the `.env` file and add your API keys (OpenAI and/or Google Gemini).

## Configuration

Edit the `.env` file to configure the application:

### API Keys
- `OPENAI_API_KEY`: Your OpenAI API key (required for GPT models)
- `GOOGLE_API_KEY`: Your Google API key (required for Gemini models)

### Backend Selection
- `LLM_BACKEND`: Set to 'openai', 'gemini', or 'local' (default: 'openai')

### Gemini Settings (when using Gemini backend)
- `GEMINI_MODEL`: Model to use (default: gemini-1.5-flash)
- `GEMINI_TEMPERATURE`: Controls randomness (0.0 to 1.0, default: 0.7)
- `GEMINI_MAX_TOKENS`: Maximum tokens in response (default: 2048)

### MCP Configuration
- `MCP_SERVER_URL`: URL of your MCP server (default: http://localhost:6789)
- `RATE_LIMIT_REQUESTS`: Maximum number of requests per time window
- `RATE_LIMIT_SECONDS`: Time window for rate limiting in seconds
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `LOG_FILE`: Path to the log file

## Running the Application

1. Start the MCP server in a separate terminal:
   ```bash
   python server.py
   ```

2. In another terminal, start the chatbot:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to: `http://localhost:7860`

## Usage

1. Type your message in the input box and press Enter or click Send.
2. The chatbot will respond based on the selected backend.
3. Use the dropdown to switch between local and OpenAI backends (if configured).

### Examples

- "What is 5 plus 3?"
- "Add 10 and 20"
- "Hello! How are you?"

## Project Structure

- `app.py`: Main application with Gradio interface
- `server.py`: MCP server implementation
- `config.py`: Configuration settings
- `requirements.txt`: Python dependencies
- `.env.example`: Example environment variables
- `README.md`: This file

## License

[MIT License](LICENSE)
