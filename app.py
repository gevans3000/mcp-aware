import json
import os
import time
import json
import logging
import requests
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Deque, Tuple
from collections import deque
from dotenv import load_dotenv, find_dotenv
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify required environment variables
REQUIRED_KEYS = ['GOOGLE_API_KEY']
for key in REQUIRED_KEYS:
    if not os.getenv(key):
        logger.error(f"Missing required environment variable: {key}")
        raise ValueError(f"Missing required environment variable: {key}")

# Import configuration
from config import (
    LOG_LEVEL, LOG_FORMAT, LOG_FILE,
    DEFAULT_BACKEND, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS,
    GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS,
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

# Initialize LLM clients and global variables
openai_client = None
chat_backend = os.getenv("DEFAULT_CHAT_BACKEND", "gemini")  # Default to gemini

def set_backend(backend: str) -> None:
    """Set the chat backend to either 'openai' or 'gemini'.
    
    Args:
        backend: The backend to use ('openai' or 'gemini')
    """
    global chat_backend
    if backend in ["openai", "gemini"]:
        chat_backend = backend
        logger.info(f"Switched to {backend} backend")
    else:
        logger.warning(f"Invalid backend: {backend}. Must be 'openai' or 'gemini'. Keeping current backend: {chat_backend}")

import openai

# Initialize OpenAI client if API key is available
if OPENAI_API_KEY:
    try:
        # Initialize with the older API
        openai.api_key = OPENAI_API_KEY
        openai_client = openai  # For compatibility with existing code
        logger.debug("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        logger.warning("OpenAI functionality will be disabled")
else:
    logger.warning("OPENAI_API_KEY not found in environment variables. OpenAI features will be disabled.")

try:
    import google.generativeai as genai
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_client = genai.GenerativeModel(GEMINI_MODEL)
    else:
        gemini_client = None
except ImportError:
    logger.warning("Google Generative AI package not installed. Install with: pip install google-generativeai")
    gemini_client = None
except Exception as e:
    logger.warning(f"Failed to initialize Gemini client: {e}")
    gemini_client = None

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
def openai_chat_response(user_input, history=None, max_retries=3, initial_delay=1):
    """
    Generate a response using OpenAI's chat completion API with retry logic.
    
    Args:
        user_input: The user's message
        history: List of (user_message, bot_response) tuples
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds (exponential backoff)
        
    Returns:
        str: The generated response or an error message
    """
    if history is None:
        history = []
    
    def make_attempt(attempt=0, delay=None):
        if delay is None:
            delay = initial_delay
            
        try:
            if not openai_client:
                return "OpenAI client is not properly configured. Check your API key and installation."
            
            # Format messages for the API (older API format)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
            
            # Add conversation history
            for user_msg, bot_msg in history:
                if user_msg:  # Only add non-empty user messages
                    messages.append({"role": "user", "content": str(user_msg)})
                if bot_msg:  # Only add non-empty bot responses
                    messages.append({"role": "assistant", "content": str(bot_msg)})
            
            # Add the current user input
            messages.append({"role": "user", "content": user_input})
            
            logger.debug(f"Sending to OpenAI (attempt {attempt + 1}): {user_input[:100]}...")
            
            # Make the API call with older API format
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=float(OPENAI_TEMPERATURE),
                max_tokens=int(OPENAI_MAX_TOKENS)
            )
            
            # Extract the response text (older API format)
            response_text = response.choices[0].message['content'].strip()
            logger.debug(f"Received response from OpenAI: {response_text[:100]}...")
            return response_text
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"OpenAI API attempt {attempt + 1} failed: {error_msg}")
            
            # Don't retry for these errors
            if any(msg in error_msg for msg in ["api key", "authentication", "not found", "invalid"]):
                return f"Error: {str(e)}. Please check your OpenAI API key and model settings."
                
            # Fall back to math processing if possible
            if attempt == 0:  # Only try math fallback on first attempt
                math_fallback = process_math_operation(user_input)
                if math_fallback:
                    return math_fallback
            
            # Retry for rate limits or temporary issues
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                return make_attempt(attempt + 1, delay)
                
            return f"I'm sorry, I'm having trouble connecting to the AI service: {e}"
    
    return make_attempt()

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
        
def process_math_operation(user_input: str) -> Optional[str]:
    """
    Process math operations in the user input by calling the MCP server.
    Currently supports adding the first two numbers found.
    
    Args:
        user_input: The user's input string
        
    Returns:
        str: The result of the math operation or None if no operation was performed
    """
    try:
        # Extract numbers from input
        numbers = extract_numbers(user_input)
        if len(numbers) < 2:
            return None
            
        # Call MCP server's add endpoint with query parameters
        params = {"a": numbers[0], "b": numbers[1]}
        response = requests.get(
            MCP_ADD_ENDPOINT,
            params=params,
            timeout=5
        )
        response.raise_for_status()
        result = response.json().get("result")
        return f"The result is: {result}"
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to call MCP add endpoint: {e}")
        # Fallback to local calculation if MCP server is not available
        if len(numbers) >= 2:
            return f"The result is: {sum(numbers[:2])} (local fallback)"
        return None
    except Exception as e:
        logger.debug(f"Math processing error: {str(e)}")
        return None

def gemini_chat_response(user_input, history=None, max_retries=3, initial_delay=1):
    """
    Generate a response using Google's Gemini API with retry logic.
    
    Args:
        user_input: The user's message
        history: List of (user_message, bot_response) tuples
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds (exponential backoff)
        
    Returns:
        str: The generated response or an error message
    """
    if history is None:
        history = []
    
    def make_attempt(attempt=0, delay=None):
        if delay is None:
            delay = initial_delay
            
        try:
            if not gemini_client:
                return "Gemini client is not properly configured. Check your API key and installation."
            
            # First, check if this is a math operation we can handle
            math_response = process_math_operation(user_input)
            if math_response:
                return math_response
            
            # Format the conversation history for Gemini
            messages = []
            
            # Add system prompt first
            system_prompt = {
                "role": "user",
                "parts": ["You are a helpful assistant that can perform calculations. "
                          "When asked to perform math operations, provide the exact numerical result. "
                          "Be concise and direct in your responses."]
            }
            messages.append(system_prompt)
            
            # Add conversation history
            for user_msg, bot_msg in history:
                if user_msg:  # Only add non-empty user messages
                    messages.append({"role": "user", "parts": [str(user_msg)]})
                if bot_msg:  # Only add non-empty bot responses
                    messages.append({"role": "model", "parts": [str(bot_msg)]})
            
            # Add the current user input
            messages.append({"role": "user", "parts": [user_input]})
            
            logger.debug(f"Sending to Gemini (attempt {attempt + 1}): {user_input[:100]}...")
            
            # Generate response with timeout
            response = gemini_client.generate_content(
                messages,
                generation_config={
                    "temperature": float(os.getenv("GEMINI_TEMPERATURE", 0.7)),
                    "max_output_tokens": int(os.getenv("GEMINI_MAX_TOKENS", 1000)),
                    "top_p": 0.8,
                    "top_k": 40
                },
                request_options={"timeout": 30}  # 30 seconds timeout
            )
            
            # Extract the response text
            if response is None:
                logger.error("Gemini returned None response")
                return "I'm sorry, I couldn't generate a response. Please try again."
                
            if hasattr(response, 'text'):
                response_text = response.text.strip()
                if not response_text:  # Empty response
                    return "I'm sorry, I couldn't generate a response. Please try again."
                return response_text
                
            elif hasattr(response, 'candidates') and response.candidates:
                try:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                        parts = candidate.content.parts
                        if parts and len(parts) > 0 and hasattr(parts[0], 'text'):
                            response_text = parts[0].text.strip()
                            if response_text:
                                return response_text
                except Exception as e:
                    logger.error(f"Error extracting text from Gemini response: {e}")
                    
            # If we get here, we couldn't extract a valid response
            logger.error(f"Unexpected Gemini response format: {response}")
            return "I'm sorry, I couldn't understand the AI's response. Please try again."
            
            logger.debug(f"Received response from Gemini: {response_text[:100]}...")
            return response_text
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"Gemini API attempt {attempt + 1} failed: {error_msg}")
            
            # Don't retry for these errors
            if any(msg in error_msg for msg in ["api key", "authentication", "not found", "invalid"]):
                return f"Error: {str(e)}. Please check your Gemini API key and model settings."
                
            # Fall back to math processing if possible
            if attempt == 0:  # Only try math fallback on first attempt
                math_fallback = process_math_operation(user_input)
                if math_fallback:
                    return math_fallback
            
            # Retry for rate limits or temporary issues
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                return make_attempt(attempt + 1, delay)
                

def get_response(user_input: str, history=None, max_retries=3, retry_delay=2, backend=None):
    """
    Get a response from the selected AI backend.
    
    Args:
        user_input: The user's input message
        history: List of (user_message, bot_response) tuples
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        backend: The backend to use ('openai' or 'gemini'). If None, uses the default backend.
        
    Returns:
        str: The AI's response
    """
    global chat_backend
    
    try:
        logger.info(f"get_response called with backend: {backend}, current chat_backend: {chat_backend}")
        
        # If no backend is provided, use the default from environment
        if backend is None:
            backend = os.getenv("DEFAULT_CHAT_BACKEND", "gemini")  # Default to gemini if not set
            logger.info(f"Using default backend: {backend}")
            
        rate_limit_error = check_rate_limit()
        if rate_limit_error:
            logger.warning(f"Rate limit exceeded: {rate_limit_error}")
            return rate_limit_error

        # Try to process as a math operation first
        math_result = process_math_operation(user_input)
        if math_result is not None:
            logger.info("Processed as math operation")
            return math_result
        
        # Ensure history is a list
        if history is None:
            history = []
            
        # If not a math operation, use the selected backend
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} with backend: {chat_backend}")
                
                if chat_backend == "openai":
                    if not openai_client:
                        error_msg = "OpenAI client is not properly initialized. Please check your API key."
                        logger.error(error_msg)
                        return error_msg
                    return openai_chat_response(user_input, history, max_retries=1, initial_delay=retry_delay)
                
                elif chat_backend == "gemini":
                    if not gemini_client:
                        error_msg = "Gemini client is not properly initialized. Please check your API key."
                        logger.error(error_msg)
                        return error_msg
                    return gemini_chat_response(user_input, history, max_retries=1, initial_delay=retry_delay)
                    
                else:  # Default to gemini if somehow an invalid backend is selected
                    logger.warning(f"Invalid backend '{chat_backend}' selected, defaulting to gemini")
                    if gemini_client:
                        return gemini_chat_response(user_input, history, max_retries=1, initial_delay=retry_delay)
                    return "Error: No valid AI backend is properly configured."
                    
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # Last attempt
                    error_msg = f"Network error after {max_retries} attempts: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    return f"I'm having trouble connecting to the AI service. Please check your internet connection and try again. Error: {str(e)}"
                
                # Exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
            except Exception as e:
                error_msg = f"Unexpected error in get_response: {str(e)}"
                logger.error(error_msg, exc_info=True)
                if attempt == max_retries - 1:  # Last attempt
                    return f"I encountered an error: {str(e)}"
                
                # Exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"Error in attempt {attempt + 1}/{max_retries}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    except Exception as e:
        error_msg = f"Critical error in get_response: {str(e)}"
        logger.error(f"Error in get_response: {str(e)}", exc_info=True)
        return f"I encountered a critical error: {str(e)}"
            
    return "I'm sorry, I couldn't generate a response. Please try again later."

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

def chat_fn(message: str, history=None):
    """
    Process a chat message and return a response.
    
    This function sanitizes the input, processes it through the selected backend,
    and returns an appropriate response. It also handles logging of the conversation.
    
    Args:
        message: The user's input message
        history: List of previous messages in the conversation as (user_msg, bot_msg) tuples
        
    Returns:
        tuple: (response_message, updated_history) - The assistant's response and updated chat history
    """
    if history is None:
        history = []
    
    try:
        # Validate input
        if not message or not isinstance(message, str):
            logger.warning("Received empty or invalid message")
            return "I didn't receive a valid message. Please try again.", history
        
        # Sanitize input
        sanitized_message = sanitize_input(message)
        if not sanitized_message:
            logger.warning("Message became empty after sanitization")
            return "I couldn't process that message. Please try rephrasing.", history
        
        logger.debug(f"Processing message: {sanitized_message}")
        
        # Convert history to the format expected by get_response
        conv_history = []
        for user_msg, bot_msg in history:
            if user_msg and bot_msg:  # Only add complete message pairs
                conv_history.append((str(user_msg), str(bot_msg)))
        
        # Get response from the appropriate backend
        response = get_response(sanitized_message, conv_history)
        
        # Validate response
        if not response or not isinstance(response, str):
            logger.warning(f"Received invalid response: {response}")
            return "I'm sorry, I couldn't generate a response. Please try again.", history
        
        # Clean up the response
        response = response.strip()
        if not response:
            return "I'm sorry, I couldn't generate a response. Please try a different query.", history
        
        logger.debug(f"Generated response: {response[:200]}...")
        
        # Add the new message and response to history
        updated_history = history + [[sanitized_message, response]]
        
        # Return empty string to clear the input, and updated history
        return "", updated_history
        
    except Exception as e:
        error_msg = "I'm sorry, I encountered an error processing your request. Please try again later."
        logger.error(f"Error in chat_fn: {str(e)}", exc_info=True)
        return error_msg, history

# Create and launch the interface
def find_available_port(start_port: int = 7861, max_attempts: int = 10) -> int:
    """Find an available port starting from the specified port."""
    import socket
    from contextlib import closing
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError(f"No available port found in range {start_port}-{start_port + max_attempts - 1}")

def toggle_sidebar(state):
    new_state = not state
    return gr.update(visible=new_state), new_state

def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def main() -> None:
    """
    Main function to initialize and run the MCP Chatbot.
    
    This function sets up the Gradio interface, tests the MCP server connection,
    and launches the chat application on an available port.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('mcp_chatbot.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Enable CORS for all routes
    from fastapi.middleware.cors import CORSMiddleware
    import gradio as gr
    
    # Create the Gradio app with CORS support
    with gr.Blocks(title="MCP Chatbot") as demo:
        # Store chat history
        chat_history = gr.State([])
        
        # Chat interface
        chatbot = gr.Chatbot(
            [],
            label="MCP Chatbot",
            height=600
        )
        
        # Input area
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                container=False,
                scale=8,
                min_width=0,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Action buttons
        with gr.Row():
            clear_btn = gr.Button("Clear Chat")
            regenerate_btn = gr.Button("Regenerate Response")
            
        # Backend selection
        with gr.Row():
            backend_dropdown = gr.Dropdown(
                ["openai", "gemini"],
                value=chat_backend,
                label="AI Model",
                interactive=True
            )
        
        # Backend state
        backend_state = gr.State(chat_backend)
        
        # Event handlers
        def on_backend_change(backend, state):
            state = backend
            return f"Switched to {backend} backend", state
        
        backend_dropdown.change(
            fn=on_backend_change,
            inputs=[backend_dropdown, backend_state],
            outputs=[gr.Textbox(visible=False), backend_state]
        )
        
        # Handle message submission
        def on_message_submit(message, history, backend):
            if not message.strip():
                return "", history, backend
            
            try:
                # Get response from the selected backend
                response = get_response(message, history, backend)
                
                # Update history
                if history is None:
                    history = []
                history.append((message, response))
                
                return "", history, backend
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logger.error(f"Error in message submission: {error_msg}", exc_info=True)
                return "", history + [[message, error_msg]], backend
        
        msg.submit(
            fn=on_message_submit,
            inputs=[msg, chat_history, backend_state],
            outputs=[msg, chatbot, backend_state],
            queue=True
        )
        
        submit_btn.click(
            fn=on_message_submit,
            inputs=[msg, chat_history, backend_state],
            outputs=[msg, chatbot, backend_state],
            queue=True
        )
        
        clear_btn.click(
            lambda: ([], []),
            None,
            [chatbot, chat_history],
            queue=False
        )
        
        regenerate_btn.click(
            lambda history: (history[:-1], history[:-1]),
            chat_history,
            [chatbot, chat_history],
            queue=True
        )
    
    # Add CORS middleware
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS middleware configured")

    try:
        # Check for required environment variables
        load_dotenv()

        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            global openai_client
            try:
                # Initialize with the older API
                openai.api_key = openai_api_key
                openai_client = openai  # For compatibility with existing code
                logger.debug("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
                logger.warning("OpenAI functionality will be disabled")
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables. OpenAI features will be disabled.")

        # Initialize Gemini client if API key is available
        global gemini_client
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if gemini_api_key:
            try:
                # Configure and initialize Gemini client
                genai.configure(api_key=gemini_api_key)
                gemini_client = genai.GenerativeModel('gemini-pro')
                
                # Test the client with a simple query
                test_response = gemini_client.generate_content("Hello")
                if test_response and hasattr(test_response, 'text'):
                    logger.info(f"Gemini client initialized and tested successfully: {test_response.text[:20]}...")
                else:
                    logger.warning("Gemini client initialized but test response format is unexpected")
            except Exception as e:
                error_msg = f"Failed to initialize Gemini client: {str(e)}"
                logger.error(error_msg)
                gemini_client = None

        # Check MCP server connection
        mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:6790")
        if mcp_server_url:
            try:
                response = requests.get(f"{mcp_server_url}/tool/add?a=2&b=3", timeout=5)
                if response.status_code == 200:
                    logger.info(f"Connected to MCP server at {mcp_server_url}")
                else:
                    logger.warning(f"MCP server returned status code {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Could not connect to MCP server at {mcp_server_url}: {e}")
        
        # Create the Gradio interface
        with gr.Blocks() as demo:
            gr.Markdown("""
            # MCP Chatbot
            Chat with an AI assistant powered by MCP.
            """)
            
            # Chat interface
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500
                # No type parameter - use default
            )
            
            with gr.Row():
                # Message input
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    show_label=False,
                    container=False,
                    scale=8,
                    min_width=200
                )
                
                # Submit button
                submit_btn = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    min_width=100
                )
            
            # Backend selection
            with gr.Row():
                backend_radio = gr.Radio(
                    ["gemini", "openai"],
                    value=chat_backend,
                    label="AI Backend",
                    interactive=True
                )
            
            # Handle backend changes
            def on_backend_change(backend):
                set_backend(backend)
                return f"Switched to {backend} backend"
                
            backend_radio.change(
                fn=on_backend_change,
                inputs=backend_radio,
                outputs=None
            )
            
            # Handle form submission
            def respond(message, chat_history):
                try:
                    if not message or not message.strip():
                        return "", chat_history or []
                        
                    logger.info(f"Processing message with backend: {chat_backend}")
                    
                    # Initialize chat_history if None
                    if chat_history is None:
                        chat_history = []
                    
                    # Ensure chat_history is a list of tuples
                    if not isinstance(chat_history, list):
                        logger.warning(f"chat_history is not a list: {type(chat_history)}")
                        chat_history = []
                    
                    # Get response from the selected backend
                    try:
                        bot_message = get_response(message, chat_history)
                        if not isinstance(bot_message, str):
                            bot_message = str(bot_message)
                        
                        # Update chat history
                        chat_history.append((message, bot_message))
                        return "", chat_history
                        
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        chat_history.append((message, f"Error: {error_msg}"))
                        return "", chat_history
                        
                except Exception as e:
                    error_msg = f"Unexpected error: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    if not isinstance(chat_history, list):
                        chat_history = []
                    chat_history.append((message, f"System error: {error_msg}"))
                    return "", chat_history
                
            # Connect the submit button
            submit_btn.click(
                fn=respond,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=False  # Disable queueing to avoid 500 errors
            )
            
            # Also submit on Enter key
            msg.submit(
                fn=respond,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=False  # Disable queueing to avoid 500 errors
            )
        
        # Launch the interface with CORS support
        demo.launch(
            server_name="0.0.0.0",  # Listen on all interfaces
            server_port=7860,  # Use a standard port
            show_error=True,
            share=False
        )

    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
