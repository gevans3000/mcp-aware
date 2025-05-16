import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Global runtime backend (default from env)
chat_backend = LLM_BACKEND

def set_backend(backend):
    global chat_backend
    chat_backend = backend

from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
MCP_URL = "http://localhost:6789"
ADD_ENDPOINT = f"{MCP_URL}/tool/add"

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_numbers(text):
    """Extract numbers from text, handling different formats"""
    import re
    # Find all numbers in the text (supports digits, words, and symbols)
    number_words = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90
    }
    
    # Try to extract numbers in digit form
    numbers = [int(n) for n in re.findall(r'\d+', text)]
    
    # If no digit numbers found, try word numbers
    if not numbers:
        for word, num in number_words.items():
            if word in text.lower():
                numbers.append(num)
    
    # If still no numbers, look for spelled out numbers
    if not numbers:
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if word in number_words:
                numbers.append(number_words[word])
    
    return numbers

# Simple model for responses
def openai_chat_response(user_input, history=None):
    import openai
    openai.api_key = OPENAI_API_KEY
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if history:
        for user, bot in history:
            messages.append({"role": "user", "content": user})
            if bot:
                messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=150,
    )
    return response.choices[0].message["content"].strip()

def get_response(user_input, history=None):
    """Minimal backend switch between OpenAI and local LLM, using runtime dropdown."""
    if chat_backend == "openai" and OPENAI_API_KEY:
        return openai_chat_response(user_input, history)
    # --- Existing local LLM logic below ---
    user_input_lower = user_input.lower()
    # Check for math operations
    math_keywords = ["add", "plus", "+", "sum", "total", "what's", "what is"]
    if any(word in user_input_lower for word in math_keywords):
        try:
            # Extract numbers from input
            numbers = extract_numbers(user_input)
            print(f"Extracted numbers: {numbers}")  # Debug print
            
            if len(numbers) >= 2:
                a, b = numbers[0], numbers[1]
                print(f"Sending request to {ADD_ENDPOINT} with a={a}, b={b}")
                try:
                    # Using params instead of json for query parameters
                    response = requests.post(
                        f"{ADD_ENDPOINT}?a={a}&b={b}",
                        timeout=5
                    )
                    print(f"Response status: {response.status_code}, content: {response.text}")
                    if response.status_code == 200:
                        result = response.json().get("result")
                        if result is not None:
                            return f"The result of {a} + {b} is {result}"
                        else:
                            return f"The server didn't return a valid result. Response: {response.text}"
                    else:
                        return f"I had trouble with the calculation. Server returned status {response.status_code}: {response.text}"
                except requests.exceptions.RequestException as e:
                    return f"I couldn't connect to the calculator service. Error: {str(e)}"
            else:
                return "I need two numbers to add. For example: 'Add 5 and 7' or 'What's 3 plus 4'"
        except Exception as e:
            print(f"Error in calculation: {str(e)}")  # Debug print
            return f"I had trouble with that calculation: {str(e)}"
    
    # Simple responses
    if "hello" in user_input_lower or "hi" in user_input_lower:
        return "Hello! I'm your MCP Chatbot. I can help with simple calculations. Try asking me to add two numbers!"
    
    if "how are you" in user_input_lower:
        return "I'm just a simple bot, but I'm functioning well! How can I assist you today?"
    
    if "thank" in user_input_lower:
        return "You're welcome! Is there anything else I can help you with?"
    
    # Default response
    return "I'm a simple chatbot that can help with basic math. Try asking me to add two numbers!"

# Simple chat function
def chat_fn(message, history):
    """Handle chat messages"""
    print(f"User: {message}")
    response = get_response(message, history)
    print(f"Bot: {response}")
    return response

# Create and launch the interface
def main():
    print("Starting MCP Chatbot...")
    print(f"MCP Server URL: {MCP_URL}")
    print("Visit http://localhost:7860 to chat")
    
    # Test the server connection
    try:
        response = requests.get(f"{MCP_URL}/greeting/Test")
        print(f"Server test: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Warning: Could not connect to MCP server: {str(e)}")
        print("Please make sure server.py is running in another terminal")
    
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
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
