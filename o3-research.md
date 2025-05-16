Perfect. I’ll prepare a full one-shot prompt you can give to another programming AI agent, based on Hugging Face’s MCP course. It will instruct it to build a context-aware chatbot MVP with full MCP integration, OpenAI and local model support, and clear structure to demonstrate MCP abilities like tool use and contextual memory.

I’ll return shortly with everything organized into a ready-to-copy prompt.


# MCP Chatbot MVP Project (Python)

## Project Overview

This project creates a **beginner-friendly chatbot** with **Model Context Protocol (MCP)** support. The chatbot can use external tools via MCP – for example, a calculator – to enhance its answers. We will build a minimal **MCP chatbot MVP** that you can run locally or on Hugging Face Spaces. Key features include:

* **Dual Model Support:** Automatically use a local open-source model (like `google/flan-t5-small`) if no API key is provided, or use OpenAI’s GPT-3.5/4 via API if an `OPENAI_API_KEY` is set.
* **MCP Tool Usage:** The AI can call a simple calculator tool for arithmetic. We define an MCP **tool server** with this capability (and demonstrate a resource). The AI will decide when to call the tool, the tool executes and returns results, and the AI incorporates the results into its replies.
* **Gradio Chat Interface:** A user-friendly web UI for chatting. The conversation is remembered (context awareness), so the AI can handle follow-up questions.
* **Latest Best Practices:** We follow Hugging Face’s MCP course guidelines – defining tools and resources with clear metadata (docstrings), using the `mcp[cli]` library (FastMCP), providing a system prompt that instructs the AI how and when to use tools, and feeding tool results back into the model’s context.

By the end, you'll have a working chatbot that leverages MCP to answer questions and perform calculations live. The code is organized into clear files with comments and includes instructions to run and troubleshoot the project.

## Project Structure and Files

We will create the following files:

* **`requirements.txt`:** Python dependencies for the project (MCP library, ML models, Gradio, etc.).
* **`.env.sample`:** A sample environment file to configure the OpenAI API key (if using OpenAI models). This can be copied to `.env`.
* **`server.py`:** The MCP tool server definition. It uses FastMCP to define available tools and resources. We include a calculator tool and a sample resource. This server can run independently to provide tool functionality.
* **`app.py`:** The main chat application. It loads the model (OpenAI or local), sets up the Gradio interface, and manages the conversation. It instructs the AI about available tools (using the tool metadata from `server.py`) and handles the logic of calling the MCP tool when the AI requests it.
* **`README.md`:** (Optional) Project README with usage instructions, model selection notes, and troubleshooting tips. We will include key instructions in comments as well for convenience.

Following is the content of each file:

---

**File: `requirements.txt`**
Lists all required Python packages for this project.

```text
mcp[cli]          # Model Context Protocol library (includes FastMCP for server & tools)
openai            # OpenAI API client for GPT models
transformers      # Hugging Face Transformers (for local model)
torch             # PyTorch (backend for local model inference)
python-dotenv     # To load environment variables from .env (OpenAI API key)
gradio            # Web UI for the chatbot
```

---

**File: `.env.sample`**
A template for environment variables. Copy this to a file named `.env` and fill in your OpenAI API key if you want to use OpenAI models. If no key is provided, the app will default to the local model.

```text
# Copy this file to .env and insert your OpenAI API key if available.
OPENAI_API_KEY=<your_openai_api_key_here>
```

---

**File: `server.py`**
Defines the MCP server with tools and resources using the FastMCP framework. We create a simple calculator tool and an example resource. Clear docstrings and metadata are provided so the AI understands when to use them.

```python
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server instance with a name and (optional) dependencies.
# The name identifies our tool server; dependencies would auto-install needed packages for tools.
mcp = FastMCP("ChatbotMCP")

# Define a calculator tool using the @mcp.tool() decorator.
@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result as a string.

    Args:
        expression: A mathematical expression to evaluate, e.g. "2+2*3".
    """
    # Basic implementation: use Python's eval safely for arithmetic.
    # Note: This is a simple approach. In a real app, consider using a math parser for safety.
    try:
        result = eval(expression)
    except Exception as e:
        return f"Error: {e}"
    return str(result)

# (Optional) Define a sample resource using @mcp.resource.
# Resources provide data the application can supply to the model (application-controlled).
# Here we add a trivial resource that always returns a fun fact.
@mcp.resource("info://fun_fact")
def fun_fact() -> str:
    """Provide a fun fact that the AI can use."""
    return "Did you know? The Eiffel Tower can be 15 cm taller during hot days."

if __name__ == "__main__":
    # Run the MCP server (this will listen for tool calls via MCP protocol, e.g., JSON-RPC).
    # In this simple project, we typically run server.py in a separate process.
    mcp.run()
```

*Explanation:*
We use **FastMCP** to register tools and resources. The `@mcp.tool()` decorator turns the `calculate()` function into an MCP tool that the model can call. The tool’s docstring describes its purpose and usage; this metadata helps the AI decide when to use it. We also define a resource `fun_fact` with a unique URI (`info://fun_fact`). Resources are usually provided to the model by the system (instead of called by the model), but we include it to illustrate the concept. The `mcp.run()` at the end starts the server so it can handle requests. (For simplicity, the chat app will call `calculate()` directly when needed, but you could run this server as a separate process to handle JSON-RPC tool calls in a fully decoupled way.)

---

**File: `app.py`**
This is the main application script. It loads the model (OpenAI or local), sets up the Gradio interface for chatting, and implements the conversation loop with tool usage. The AI is given a **system prompt** that lists available tools and instructions on how to use them. When the AI’s response indicates a tool call (in JSON format), the app calls the `server.py` tool function, gets the result, and feeds it back into the model’s context. Finally, it returns the assistant’s answer to the user.

```python
import os
import json
import importlib
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists) to get OPENAI_API_KEY, etc.
load_dotenv()

# Import the MCP server module to access tool functions and metadata.
# We won't run the server here, but we'll use its definitions.
server = importlib.import_module("server")

# Determine which model backend to use based on environment variable.
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = "gpt-3.5-turbo"  # default OpenAI model to use (can adjust to "gpt-4" if available)

# If using OpenAI, set up the OpenAI library with the API key.
if USE_OPENAI:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    # If not using OpenAI, load a local Hugging Face model (e.g., flan-t5-small).
    from transformers import pipeline
    LOCAL_MODEL = "google/flan-t5-small"  # small, instruct-tuned model
    # Initialize a text generation pipeline for the local model.
    # Using text2text-generation for encoder-decoder like T5.
    local_pipeline = pipeline("text2text-generation", model=LOCAL_MODEL)

# Prepare system instructions and tool metadata for the AI
tool_descriptions = []
# We only have one tool 'calculate', but let's prepare this generally.
if hasattr(server, "calculate"):
    calc_doc = server.calculate.__doc__ or "No description."
    calc_doc = calc_doc.strip()
    tool_descriptions.append(f'**Tool** "calculate(expression: str) -> str": {calc_doc}')
# Include resource info (optional, we won't have the model explicitly call this, but it's part of MCP).
if hasattr(server, "fun_fact"):
    resource_doc = server.fun_fact.__doc__ or "No description."
    resource_doc = resource_doc.strip()
    tool_descriptions.append(f'**Resource** "info://fun_fact": {resource_doc}')

# Build the system prompt string:
system_prompt = (
    "You are an intelligent assistant with access to tools.\n"
    "You can answer user questions and also use the following tool when needed:\n"
    + "\n".join(tool_descriptions) + "\n\n"
    "Instructions:\n"
    "- If the user asks a question that requires calculation or tool usage, respond with a JSON **tool request** instead of a direct answer.\n"
    "  Use the format: `{\"tool\": \"tool_name\", \"params\": { ... }}` with no additional text. For example, to use the calculator, output `{\"tool\": \"calculate\", \"params\": {\"expression\": \"2+2\"}}`.\n"
    "- If no tool is needed, just answer normally.\n"
    "- After a tool is used, the result will be given, and then you should incorporate that result into your final answer.\n"
    "- Never reveal the internal tool JSON or your reasoning. Just use it silently when needed."
)

# We will maintain the conversation history. 
# For OpenAI, we'll use the chat message format; for local model, we'll build a text prompt from history.
conversation_history = []  # to store tuples of (user_message, assistant_reply)

# Define a function to generate the assistant's response given a new user message.
def generate_response(user_message):
    """Generate the assistant's response to user_message, using a model (OpenAI or local) and tools via MCP."""
    # Append user message to conversation history
    conversation_history.append((user_message, None))  # Assistant reply is None for now

    if USE_OPENAI:
        # Construct messages for OpenAI ChatCompletion
        messages = [{"role": "system", "content": system_prompt}]
        # Add previous conversation turns
        for user, assistant in conversation_history[:-1]:  # all previous completed turns
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        # Add the new user prompt
        messages.append({"role": "user", "content": user_message})

        # First, request a response from the model
        response = openai.ChatCompletion.create(model=OPENAI_MODEL, messages=messages)
        assistant_reply = response["choices"][0]["message"]["content"].strip()
        # Check if the model is requesting a tool usage (by outputting a JSON object)
        tool_call = None
        if assistant_reply.startswith("{"):
            try:
                tool_call = json.loads(assistant_reply)
            except json.JSONDecodeError:
                tool_call = None

        if tool_call and "tool" in tool_call:
            # The model asked to use a tool
            tool_name = tool_call.get("tool")
            params = tool_call.get("params", {})
            tool_result = None

            # Call the appropriate tool from the server
            if tool_name == "calculate" and hasattr(server, "calculate"):
                # For simplicity, directly call the function
                expr = params.get("expression", "")
                tool_result = server.calculate(expr)
            # (If more tools exist, handle them here accordingly)

            # Prepare a follow-up message informing the model of the tool output
            result_message = f'Tool "{tool_name}" returned: {tool_result}'
            messages.append({"role": "assistant", "content": assistant_reply})       # log the tool call
            messages.append({"role": "system", "content": result_message})           # provide tool result

            # Ask the model again, now that it has the tool result, to get the final answer
            response = openai.ChatCompletion.create(model=OPENAI_MODEL, messages=messages)
            assistant_reply = response["choices"][0]["message"]["content"].strip()
        # Update the history with the assistant's final answer for this turn
        conversation_history[-1] = (user_message, assistant_reply)
        return assistant_reply

    else:
        # Local model: build a single prompt including system instructions, history, and the new question.
        prompt = system_prompt + "\n\n"
        # Include previous conversation if any
        for user, assistant in conversation_history[:-1]:
            prompt += f"User: {user}\nAssistant: {assistant}\n"
        prompt += f"User: {user_message}\nAssistant:"

        # Generate a response with the local model pipeline
        result = local_pipeline(prompt, max_length=256, do_sample=False, num_return_sequences=1)
        assistant_output = result[0]["generated_text"].strip()
        # The local model's output may contain the prompt text again (depending on model). If so, remove it.
        # (Flan-T5 tends to only output the completion, but we include this step just in case.)
        if assistant_output.startswith(prompt):
            assistant_output = assistant_output[len(prompt):].strip()

        # Check for tool request in the output
        tool_call = None
        if assistant_output.startswith("{"):
            try:
                tool_call = json.loads(assistant_output)
            except json.JSONDecodeError:
                tool_call = None

        if tool_call and "tool" in tool_call:
            # The model wants to use a tool
            tool_name = tool_call.get("tool")
            params = tool_call.get("params", {})
            tool_result = ""
            if tool_name == "calculate" and hasattr(server, "calculate"):
                expr = params.get("expression", "")
                tool_result = server.calculate(expr)
            # Formulate the assistant's final answer now that we have the tool result.
            # We append the tool result to the conversation and prompt the model again.
            prompt += f' {assistant_output}\n'                        # include the tool call the assistant made
            prompt += f'Tool "{tool_name}" output: {tool_result}\n'   # provide the tool result to the model
            prompt += "Assistant:"                                   # ask the model to continue from here
            result = local_pipeline(prompt, max_length=256, do_sample=False)
            assistant_output = result[0]["generated_text"].strip()
            # Remove the prompt part again if present
            if assistant_output.startswith(prompt):
                assistant_output = assistant_output[len(prompt):].strip()
        # Update conversation history with the final answer.
        assistant_reply = assistant_output
        conversation_history[-1] = (user_message, assistant_reply)
        return assistant_reply

# Set up Gradio chat interface
import gradio as gr

def respond(user_input, history):
    """Gradio event handler: given user_input and chat history, generate a response and update history."""
    # Generate AI response
    bot_response = generate_response(user_input)
    # history is a list of [user, bot] pairs for display; we append the new pair
    history = history or []
    history.append((user_input, bot_response))
    return history, ""  # return updated history and clear the input box

# Create a Gradio Blocks interface for the chat
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 MCP Chatbot\nAsk the chatbot anything. It can do math using a calculator tool!")
    chatbox = gr.Chatbot()
    message = gr.Textbox(show_label=False, placeholder="Type your message and press enter")
    message.style(container=False)
    submit_btn = gr.Button("Send")
    # When user submits, call respond()
    submit_btn.click(respond, [message, chatbox], [chatbox, message])
    message.submit(respond, [message, chatbox], [chatbox, message])  # allow pressing Enter to submit

# To run locally, uncomment the line below:
# demo.launch()
```

*Key points in `app.py`:*

* We detect at startup whether to use OpenAI or a local model. If an `OPENAI_API_KEY` is present in the environment, we use `openai.ChatCompletion` with `gpt-3.5-turbo`. Otherwise, we load a local Hugging Face model (Flan-T5) via `transformers.pipeline`.
* We construct a **system prompt** containing tool descriptions and instructions for tool use. This prompt follows best practices: it clearly lists the tool’s name, signature, and purpose (from the `calculate` docstring) and tells the AI *when and how* to call the tool. For example, it instructs the model to output a JSON object like `{"tool": "calculate", "params": {...}}` when a calculation is needed, instead of answering directly.
* The conversation is managed in a `conversation_history` list. For OpenAI, we maintain a `messages` list of dicts (roles: system/user/assistant) as required by the API. For the local model, we build a single text prompt string that includes the prior dialogue (in a simple "User: ... / Assistant: ..." format).
* **Tool call handling:** After getting the model’s initial response, we check if it starts with a `{` (indicating a JSON tool request). We attempt to parse it as JSON. If the model requested a tool (e.g., `{"tool": "calculate", "params": {"expression": "2+2"}}`):

  * We identify the tool name and parameters, then call the corresponding function from `server.py` (here we directly call `server.calculate(...)`). This simulates the MCP tool call – in a fully deployed setting, the `mcp` framework would handle this via JSON-RPC behind the scenes.
  * We then take the tool’s result and inject it back into the model’s context. For OpenAI, we append a new system message like `Tool "calculate" returned: 4` and ask for a new completion. For the local model, we append a line with the tool output into the prompt and generate again. This way, the model can use the result to formulate the final answer. **This demonstrates MCP’s loop: model -> tool -> model**, ensuring the AI’s answer includes live tool data.
* We use **Gradio** to create a simple web interface. The `gr.Chatbot` component displays the chat history, and a `Textbox` + `Button` handle user input. The `respond` function ties it together by generating a response with `generate_response()` and updating the history. We also provide a Markdown header in the UI.
* The code includes plenty of **comments** to explain each step to a beginner. This makes it easier to follow along, modify the model or tools, and troubleshoot if something goes wrong.

---

**File: `README.md`** *(Optional but recommended)*
This README provides instructions on how to set up and run the project, and offers troubleshooting tips. (These notes largely complement the in-code comments.)

````markdown
# MCP Chatbot MVP – Usage Guide

## Installation

1. **Install Dependencies:** Run `pip install -r requirements.txt` to install the required packages. Ensure you have Python 3.9+ and internet access (to download the model on first run).
2. **Configure OpenAI (Optional):** If you want to use OpenAI's GPT model, copy `.env.sample` to `.env` and put your OpenAI API key in it. If you skip this, the app will use a local Hugging Face model (no API key required).

## Running the Tool Server

Before starting the chat app, launch the MCP tool server:
```bash
python server.py
````

This will start the MCP server (which hosts the calculator tool and resource). It will listen for tool requests from the chat application. You should see console output from FastMCP indicating the server is running (it may not print much by default).

*Note:* In this project, the chat application directly calls the tool function for simplicity, so running the server separately is optional. However, it's good to run it to simulate a real MCP setup and ensure everything is registered.

## Running the Chatbot App

In a new terminal (while the server is running), start the Gradio chat interface:

```bash
python app.py
```

* If running locally, Gradio will output a local URL (like `http://127.0.0.1:7860`) and possibly a network URL. Open the local URL in your browser to use the chatbot.
* On Hugging Face Spaces, the interface will appear automatically when the space is deployed (you might need to integrate the server in the same process due to space constraints – see notes below).

Now you can chat with the bot. For example:

* **User:** "Hello, what can you do?"
* **Assistant:** (The bot will greet you and mention it can do calculations.)
* **User:** "What's 2 + 2 times 3?"
* **Assistant:** The bot should recognize this needs calculation, call the calculator tool (internally via MCP), get the result (8), and reply with the answer "8". You won't see the JSON or tool call – it's all behind the scenes.

## How It Works (Summary)

* The assistant is instructed about the **Calculator tool** via a system prompt. Whenever a question involves math, the AI will output a JSON tool request instead of guessing.
* The app catches this JSON, calls the tool on the MCP server, and then gives the result back to the AI. The AI then provides the final answer with the calculation done.
* We used the **Model Context Protocol (MCP)** to standardize tool usage. Tools are defined in `server.py` with docstrings (metadata), and the protocol (handled by the `mcp` library) would normally use JSON-RPC to communicate between the AI (client) and tool server. In our implementation, we simplified the call by invoking the function directly, but the flow is the same.

## Model Selection and Configuration

* With no OpenAI key, the app defaults to **`google/flan-t5-small`**, a lightweight open-source model. This model understands instructions but may be limited in conversation ability. You can replace it with a bigger model (like `google/flan-t5-xl` or another chat model) if you have the resources, by changing the model name in `app.py`.
* With `OPENAI_API_KEY` set, the app uses OpenAI’s **GPT-3.5 Turbo** by default. You can change the model to GPT-4 (if you have access) by updating `OPENAI_MODEL` in `app.py`.
* The OpenAI path will generally yield better conversational quality and more reliable tool usage. The local model path is offline and free, but may sometimes not follow instructions perfectly (small models can make mistakes). We include both to align with various needs and budgets.

## Troubleshooting

* **Tool not being used:** If the assistant’s answer is wrong or it doesn’t use the calculator when it should, check that the system prompt is correctly constructed and the tool server functions are imported. The AI might sometimes try to answer without the tool if it’s confident. You can rephrase the question or enforce usage by asking something like "Use the calculator to compute ...".
* **OpenAI errors or no response:** Make sure your API key is correct and has available quota. The `.env` file should be in the same directory as `app.py`. If issues persist, try using the local model to isolate the problem.
* **Local model issues:** The first run will download the model; ensure you have internet for that. If you get a `ModuleNotFoundError` for `torch` or similar, make sure PyTorch installed correctly (`pip install torch`). For performance, note that `flan-t5-small` is quick on CPU, but larger models will be slower without a GPU.
* **MCP server not running:** In our setup we directly call the tool function, so the app should still work. But if you want full MCP separation, ensure `server.py` is running. You can test the server by running `fastmcp inspect server.py` (if you have the MCP CLI) or simply observe that no errors occurred on startup.
* **Deploying on Hugging Face Spaces:** You might need to run both the server and app in one script. One approach is to start the MCP server in a background thread within `app.py` (or call the tool functions directly as we did). Because we directly import and call the tool, our app should work on Spaces as is. Just be mindful of hardware limitations (use a small model or enable GPU if needed).

Feel free to customize the project – add more tools, upgrade the model, or tweak the UI. This MVP is a foundation demonstrating how **MCP allows LLMs to use tools** in a standardized way. Happy coding!

```

---

**Note:** The above prompt (with all files and instructions) is meant to be given to a coding AI assistant. The assistant will then generate the project files accordingly. Once you have those files, follow the README steps to install requirements and run the application. Enjoy your MCP-enabled chatbot!
```
