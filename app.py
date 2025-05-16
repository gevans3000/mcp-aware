import json
import time
import logging
import requests
import gradio as gr
from collections import deque
from typing import Deque, Optional, List, Tuple, Any, Dict, Union

# Import configuration
from config import (
    LOG_LEVEL, LOG_FORMAT, LOG_FILE,
    DEFAULT_BACKEND, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS,
    MCP_ADD_ENDPOINT, MCP_GREETING_ENDPOINT, SERVER_HOST, SERVER_PORT,
    RATE_LIMIT_REQUESTS, RATE_LIMIT_SECONDS, MATH_KEYWORDS, DEFAULT_RESPONSES,
    MAX_INPUT_LENGTH, get_config
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting
request_timestamps: Deque[float] = deque()

# Global runtime backend (default from config)
chat_backend = DEFAULT_BACKEND

def set_backend(backend: str) -> None:
    """Set the chat backend to either 'openai' or 'local'.
    
    Args:
        backend: The backend to use ('openai' or 'local')
    """
    global chat_backend
    if backend in ["openai", "local"]:
        chat_backend = backend
        logger.info(f"Switched to {backend} backend")
    else:
        logger.warning(f"Invalid backend: {backend}. Keeping current backend: {chat_backend}")

from openai import OpenAI

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def extract_numbers(text: str) -> list[int]:
    """
    Extract numbers from text, handling digits, number words, and basic arithmetic.
    
    Args:
        text: Input text to extract numbers from
        
    Returns:
        List of integers found in the text
    """
    import re
    
    # Mapping of number words to their numeric values
    number_words = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
        'million': 1000000, 'billion': 1000000000
    }
    
    # First, try to extract numbers in digit form
    numbers = [int(match) for match in re.findall(r'\b\d+\b', text)]
    
    # If no digit numbers found, try to find number words
    if not numbers:
        text_lower = text.lower()
        # Find all number words in the text
        found_words = [word for word in number_words if f' {word} ' in f' {text_lower} ']
        
        # Convert found words to numbers
        if found_words:
            numbers = [number_words[word] for word in found_words]
    
    # If still no numbers, try to find spelled out numbers as separate words
    if not numbers:
        words = re.findall(r'\b\w+\b', text.lower())
        numbers = [number_words[word] for word in words if word in number_words]
    
    return numbers

# Simple model for responses
def openai_chat_response(user_input, history=None):
    try:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if history:
            for user, bot in history:
                messages.append({"role": "user", "content": user})
                if bot:
                    messages.append({"role": "assistant", "content": bot})
        messages.append({"role": "user", "content": user_input})
        
        if not openai_client:
            return DEFAULT_RESPONSES["no_openai_key"]
            
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API Error: {str(e)}", exc_info=True)
        return "I'm sorry, I encountered an error processing your request."

def check_rate_limit() -> Optional[str]:
    """Check if the rate limit has been exceeded.
    
    Returns:
        str: Error message if rate limit exceeded, None otherwise
    """
    current_time = time.time()
    
    # Remove timestamps older than the time window
    while request_timestamps and request_timestamps[0] < current_time - RATE_LIMIT_SECONDS:
        request_timestamps.popleft()
    
    # Check if we've exceeded the rate limit
    if len(request_timestamps) >= RATE_LIMIT_REQUESTS:
        return "Rate limit exceeded. Please try again later."
    
    # Add current timestamp and return None (no error)
    request_timestamps.append(current_time)
    return None

def get_response(user_input: str, history=None) -> str:
    """Minimal backend switch between OpenAI and local LLM, using runtime dropdown.
    
    Args:
        user_input: The user's input message
        history: Optional chat history
        
    Returns:
        str: The assistant's response
    """
    # Check rate limit first
    if rate_limit_error := check_rate_limit():
        return rate_limit_error
    if chat_backend == "openai":
        if not OPENAI_API_KEY:
            return DEFAULT_RESPONSES["no_openai_key"]
        return openai_chat_response(user_input, history)
        
    # --- Local LLM logic ---
    user_input_lower = user_input.lower()
    # Check for math operations
    if any(word in user_input_lower for word in MATH_KEYWORDS):
        try:
            # Extract numbers from input
            numbers = extract_numbers(user_input)
            logger.debug(f"Extracted numbers: {numbers}")
            
            if len(numbers) >= 2:
                a, b = numbers[0], numbers[1]
                logger.info(f"Sending calculation request: {a} + {b}")
                try:
                    # Using params instead of json for query parameters
                    response = requests.post(
                        f"{MCP_ADD_ENDPOINT}?a={a}&b={b}",
                        timeout=5
                    )
                    logger.debug(f"Response status: {response.status_code}, content: {response.text}")
                    
                    if response.status_code == 200:
                        result = response.json().get("result")
                        if result is not None:
                            logger.info(f"Calculation successful: {a} + {b} = {result}")
                            return f"The result of {a} + {b} is {result}"
                        else:
                            logger.warning(f"Invalid response format: {response.text}")
                            return f"The server didn't return a valid result. Response: {response.text}"
                    else:
                        logger.error(f"Server error: {response.status_code} - {response.text}")
                        return f"I had trouble with the calculation. Server returned status {response.status_code}"
                        
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request failed: {str(e)}", exc_info=True)
                    return "I couldn't connect to the calculator service. Please try again later."
                    
            else:
                logger.warning(f"Insufficient numbers found in input: {user_input}")
                return "I need two numbers to add. For example: 'Add 5 and 7' or 'What's 3 plus 4'"
                
        except Exception as e:
            logger.error(f"Unexpected error in calculation: {str(e)}", exc_info=True)
            return "I had trouble with that calculation. Please try again."
        
    # Simple responses
    if "hello" in user_input_lower or "hi" in user_input_lower:
        return DEFAULT_RESPONSES["greeting"]
    
    if "how are you" in user_input_lower:
        return DEFAULT_RESPONSES["how_are_you"]
    
    if "thank" in user_input_lower:
        return DEFAULT_RESPONSES["thanks"]
    
    # Default response
    return DEFAULT_RESPONSES["fallback"]

# Simple chat function
def sanitize_input(text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
    """Sanitize user input to prevent injection attacks and limit length.
    
    Args:
        text: The input text to sanitize
        max_length: Maximum allowed length of the input
        
    Returns:
        Sanitized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Truncate to max length
    text = text[:max_length]
    
    # Remove any null bytes and other potentially dangerous characters
    text = text.replace('\x00', '').replace('\r', '').replace('\n', ' ')
    
    # Remove any HTML/script tags
    import re
    text = re.sub(r'<[^>]*>', '', text)
    
    return text.strip()

def chat_fn(message: str, history: List[Tuple[str, str]]) -> str:
    """
    Process a chat message and return a response.
    
    This function sanitizes the input, processes it through the selected backend,
    and returns an appropriate response. It also handles logging of the conversation.
    
    Args:
        message: The user's input message
        history: List of tuples containing (user_message, bot_response) pairs
        
    Returns:
        The assistant's response as a string
    """
    # Sanitize input
    sanitized_message = sanitize_input(message)
    if not sanitized_message:
        logger.warning("Received empty or invalid message after sanitization")
        return "Please provide a valid message."
    
    logger.info(f"User message: {sanitized_message}")
    response = get_response(sanitized_message, history)
    logger.info(f"Bot response: {response}")
    return response

# Create and launch the interface
def main() -> None:
    """
    Main function to initialize and run the MCP Chatbot.
    
    This function sets up the Gradio interface, tests the MCP server connection,
    and launches the chat application.
    """
    logger.info("Starting MCP Chatbot...")
    logger.info(f"MCP Server URL: {MCP_GREETING_ENDPOINT.replace('/greeting/Test', '')}")
    logger.info(f"Visit http://{SERVER_HOST}:{SERVER_PORT} to chat")
    
    # Test the server connection
    try:
        logger.info("Testing connection to MCP server...")
        response = requests.get(MCP_GREETING_ENDPOINT, timeout=5)
        logger.info(f"Server test successful: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not connect to MCP server: {str(e)}")
        logger.warning("Please make sure server.py is running in another terminal")
    
    # Create and launch the interface
    with gr.Blocks() as demo:
        backend_selector = gr.Dropdown(
            choices=["openai", "local"],
            value=chat_backend,
            label="Choose Chat Backend"
        )
        backend_selector.change(fn=set_backend, inputs=backend_selector, outputs=None)
        gr.ChatInterface(chat_fn, title="MCP Chatbot")
    
    # Launch the interface
    demo.launch(server_name=SERVER_HOST, server_port=SERVER_PORT)

if __name__ == "__main__":
    main()
