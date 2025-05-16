"""Configuration settings for the MCP Chatbot application."""
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Application settings
APP_NAME = "MCP Chatbot"
VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Server configuration
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7861"))
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:6789")
MCP_ADD_ENDPOINT = f"{MCP_SERVER_URL}/tool/add"
MCP_GREETING_ENDPOINT = f"{MCP_SERVER_URL}/greeting/Test"

# Rate limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", "60"))

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "150"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# Chat configuration
DEFAULT_BACKEND = os.getenv("DEFAULT_BACKEND", "local").lower()
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "1000"))

# Math operation keywords
MATH_KEYWORDS = ["add", "plus", "+", "sum", "total", "what's", "what is"]

# Default responses
DEFAULT_RESPONSES = {
    "greeting": "Hello! I'm your MCP Chatbot. I can help with simple calculations. Try asking me to add two numbers!",
    "how_are_you": "I'm just a simple bot, but I'm functioning well! How can I assist you today?",
    "thanks": "You're welcome! Is there anything else I can help you with?",
    "fallback": "I'm a simple chatbot that can help with basic math. Try asking me to add two numbers!",
    "no_numbers": "I need two numbers to add. For example: 'Add 5 and 7' or 'What's 3 plus 4'",
    "server_error": "I couldn't connect to the calculator service. Please try again later.",
    "calculation_error": "I had trouble with that calculation. Please try again.",
    "invalid_input": "Please provide a valid message.",
    "no_openai_key": "Error: OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable or switch to local mode.",
    "rate_limit": "Rate limit exceeded. Please try again later."
}

def get_config() -> Dict[str, Any]:
    """Return the current configuration as a dictionary."""
    return {
        "app_name": APP_NAME,
        "version": VERSION,
        "debug": DEBUG,
        "server": {
            "host": SERVER_HOST,
            "port": SERVER_PORT,
            "mcp_url": MCP_SERVER_URL
        },
        "rate_limiting": {
            "requests": RATE_LIMIT_REQUESTS,
            "seconds": RATE_LIMIT_SECONDS
        },
        "openai": {
            "model": OPENAI_MODEL,
            "temperature": OPENAI_TEMPERATURE,
            "max_tokens": OPENAI_MAX_TOKENS
        },
        "logging": {
            "level": LOG_LEVEL,
            "format": LOG_FORMAT,
            "file": LOG_FILE
        },
        "chat": {
            "default_backend": DEFAULT_BACKEND,
            "max_input_length": MAX_INPUT_LENGTH
        }
    }
