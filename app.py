import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MCP_URL = "http://localhost:6789"
ADD_ENDPOINT = f"{MCP_URL}/tool/add"

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
def get_response(user_input):
    """Simple response generator for the demo"""
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
    if "hello" in user_input or "hi" in user_input:
        return "Hello! I'm your MCP Chatbot. I can help with simple calculations. Try asking me to add two numbers!"
    
    if "how are you" in user_input:
        return "I'm just a simple bot, but I'm functioning well! How can I assist you today?"
    
    if "thank" in user_input:
        return "You're welcome! Is there anything else I can help you with?"
    
    # Default response
    return "I'm a simple chatbot that can help with basic math. Try asking me to add two numbers!"

# Simple chat function
def chat_fn(message, history):
    """Handle chat messages"""
    print(f"User: {message}")
    response = get_response(message)
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
    demo = gr.ChatInterface(
        fn=chat_fn,
        title="MCP Chatbot",
        description="A simple chatbot that can perform calculations using MCP tools"
    )
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()
