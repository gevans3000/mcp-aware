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

# Global runtime backend (default from config)
chat_backend = DEFAULT_BACKEND

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

from openai import OpenAI

# Initialize LLM clients
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.debug("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        logger.warning("OpenAI functionality will be disabled")

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
def openai_chat_response(user_input, history=None):
    """Generate a response using OpenAI's chat completion API.
    
    Args:
        user_input: The user's message
        history: List of (user_message, bot_response) tuples
        
    Returns:
        str: The generated response or an error message
    """
    if not openai_client:
        error_msg = "OpenAI client is not initialized. Please check your API key."
        logger.error(error_msg)
        return error_msg
    
    try:
        # Prepare the conversation history
        messages = [{"role": "system", "content": "You are a helpful assistant that can perform calculations and answer questions concisely."}]
        
        # Add conversation history if available
        if history:
            for user_msg, bot_msg in history:
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if bot_msg:
                    messages.append({"role": "assistant", "content": bot_msg})
        
        # Add the current user input
        messages.append({"role": "user", "content": user_input})
        
        logger.debug(f"Sending to OpenAI: {messages}")
        
        # Make the API call
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
        )
        
        # Extract and return the response
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            logger.debug(f"Received from OpenAI: {content}")
            return content.strip()
        else:
            error_msg = "Received empty or invalid response from OpenAI"
            logger.error(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"OpenAI API Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"I'm sorry, I encountered an error with OpenAI: {str(e)}"

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
    """Process math operations in the user input."""
    try:
        # Check for math-related keywords
        math_keywords = ["+", "-", "*", "/", "plus", "minus", "times", "divided by", "add", "subtract", "multiply", "divide"]
        if not any(keyword in user_input.lower() for keyword in math_keywords):
            return None
            
        # Extract numbers using the existing function
        numbers = extract_numbers(user_input)
        if len(numbers) < 2:
            return None
            
        # For simplicity, just add the numbers for now
        # In a real app, you'd want to parse the actual operation
        result = sum(numbers)
        return f"The result is: {result}"
        
    except Exception as e:
        logger.debug(f"Math processing error: {str(e)}")
        return None

def gemini_chat_response(user_input, history=None):
    """Get response from Google Gemini model with math operation handling."""
    if not gemini_client:
        return "Error: Gemini client is not properly configured. Check your API key and installation."
    
    # First, check if this is a math operation we can handle
    math_response = process_math_operation(user_input)
    if math_response:
        return math_response
    
    try:
        # Format the conversation history for Gemini
        messages = []
        if history:
            for user_msg, bot_msg in history:
                messages.append({"role": "user", "parts": [user_msg]})
                if bot_msg:
                    messages.append({"role": "model", "parts": [bot_msg]})
        
        # Add the current user input
        messages.append({"role": "user", "parts": [user_input]})
        
        # Generate response with more specific instructions
        system_prompt = {"role": "user", "parts": ["You are a helpful assistant that can perform calculations. "
                                                "When asked to perform math operations, provide the exact numerical result. "
                                                "Be concise and direct in your responses."]}
        
        # Insert system prompt at the beginning if there's no history
        if not history:
            messages.insert(0, system_prompt)
        
        # Generate response
        response = gemini_client.generate_content(
            messages,
            generation_config={
                "temperature": GEMINI_TEMPERATURE,
                "max_output_tokens": GEMINI_MAX_TOKENS,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        
        # Extract the response text
        if hasattr(response, 'text'):
            return response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            logger.error(f"Unexpected Gemini response format: {response}")
            return "I'm sorry, I couldn't process the response from Gemini."
            
    except Exception as e:
        logger.error(f"Gemini API Error: {str(e)}", exc_info=True)
        # Fall back to local math processing if Gemini fails
        math_fallback = process_math_operation(user_input)
        if math_fallback:
            return math_fallback
        return "I'm sorry, I encountered an error processing your request with Gemini."


def get_response(user_input, history=None):
    """Get response from the appropriate backend based on current selection."""
    try:
        # Check rate limiting first
        rate_limit_error = check_rate_limit()
        if rate_limit_error:
            return rate_limit_error
            
        logger.debug(f"Using backend: {chat_backend} for input: {user_input}")
        
        if chat_backend == "gemini":
            if not GOOGLE_API_KEY or not gemini_client:
                error_msg = "Google API key is not set or Gemini client is not initialized"
                logger.error(error_msg)
                return f"Error: {error_msg}. Please check your GOOGLE_API_KEY in the .env file."
            return gemini_chat_response(user_input, history)
            
        elif chat_backend == "openai":
            if not OPENAI_API_KEY or not openai_client:
                error_msg = "OpenAI API key is not set or client is not initialized"
                logger.error(error_msg)
                return f"Error: {error_msg}. Please check your OPENAI_API_KEY in the .env file."
            return openai_chat_response(user_input, history)
            
        else:  # Default to gemini if somehow an invalid backend is selected
            logger.warning(f"Invalid backend '{chat_backend}' selected, defaulting to gemini")
            return gemini_chat_response(user_input, history)
            
    except Exception as e:
        logger.error(f"Error in get_response: {str(e)}", exc_info=True)
        return f"An error occurred while generating a response: {str(e)}"
        
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
    try:
        # Validate input
        if not message or not isinstance(message, str):
            logger.warning("Received empty or invalid message")
            return "I didn't receive a valid message. Please try again."
        
        # Sanitize input
        sanitized_message = sanitize_input(message)
        if not sanitized_message:
            logger.warning("Message became empty after sanitization")
            return "I couldn't process that message. Please try rephrasing."
        
        logger.debug(f"Processing message: {sanitized_message}")
        
        # Get response from the appropriate backend
        response = get_response(sanitized_message, history)
        
        # Validate response
        if not response or not isinstance(response, str):
            logger.warning(f"Received invalid response: {response}")
            return "I'm sorry, I couldn't generate a response. Please try again."
            
        # Clean up the response
        response = response.strip()
        if not response:
            return "I'm sorry, I couldn't generate a response. Please try a different query."
            
        logger.debug(f"Generated response: {response[:200]}...")
        return response
        
    except Exception as e:
        error_msg = "I'm sorry, I encountered an error processing your request. Please try again later."
        logger.error(f"Error in chat_fn: {str(e)}", exc_info=True)
        return error_msg

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

def main() -> None:
    """
    Main function to initialize and run the MCP Chatbot.
    
    This function sets up the Gradio interface, tests the MCP server connection,
    and launches the chat application on an available port.
    """
    # Find an available port
    try:
        port = find_available_port()
        logger.info("Starting MCP Chatbot...")
        logger.info(f"MCP Server URL: {MCP_GREETING_ENDPOINT.replace('/greeting/Test', '')}")
        logger.info(f"Visit http://{SERVER_HOST}:{port} to chat")
        
        # Test the server connection
        try:
            logger.info("Testing connection to MCP server...")
            response = requests.get(MCP_GREETING_ENDPOINT, timeout=5)
            logger.info(f"Server test successful: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to MCP server: {str(e)}")
            logger.warning("Please make sure server.py is running in another terminal")
        
        # Create the interface
        with gr.Blocks() as demo:
          # Create backend selection dropdown
            backend_dropdown = gr.Dropdown(
                ["openai", "gemini"],
                value=chat_backend,
                label="Backend",
                info="Select the LLM backend to use"
            )
            backend_dropdown.change(fn=set_backend, inputs=backend_dropdown, outputs=None)
            gr.ChatInterface(chat_fn, title="MCP Chatbot")
        
        # Launch the interface
        demo.launch(server_name=SERVER_HOST, server_port=port, show_error=True)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    main()
