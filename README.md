# MCP Chatbot

A chatbot that can perform calculations using an MCP server and optionally use OpenAI's API for more advanced responses.

## Features

- **Dual Backend Support**: Switch between local (flan-t5-small) and OpenAI backends
- **Math Operations**: Performs addition via MCP server
- **Rate Limiting**: Prevents abuse of the chatbot
- **Input Sanitization**: Protects against injection attacks
- **Configurable**: Easy configuration via environment variables
- **Logging**: Comprehensive logging for debugging and monitoring

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- (Optional) OpenAI API key if using the OpenAI backend

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
   Edit the `.env` file and add your OpenAI API key if using the OpenAI backend.

## Configuration

Edit the `.env` file to configure the application:

- `DEFAULT_BACKEND`: Set to 'openai' or 'local'
- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI backend)
- `MCP_SERVER_URL`: URL of the MCP server (default: http://localhost:6789)
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
