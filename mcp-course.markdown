# Instructions for Building MCP Course Assistants

## Overview
You are tasked with building two AI assistants for the Hugging Face MCP Course, a free course teaching the Model Context Protocol (MCP) for AI and machine learning. One assistant will use the OpenAI API, and the other will use Google’s Gemini model via Vertex AI’s free tier. Both assistants should answer questions, explain MCP concepts, and guide users through assignments. Below are detailed steps for each.

## Part 1: OpenAI API Assistant

### Objective
Create a Python-based chatbot using the OpenAI API to assist with the Hugging Face MCP Course.

### Steps

1. **Obtain OpenAI API Key**:
   - Access your OpenAI account at [OpenAI Platform](https://platform.openai.com/).
   - Navigate to account settings and retrieve your API key.
   - Store the key securely (e.g., as an environment variable).

2. **Choose a Model**:
   - Use `o4-mini` for cost-effectiveness or `o3-mini` for advanced capabilities, depending on budget.

3. **Install Required Libraries**:
   - Install the `openai` library:
     ```bash
     pip install openai
     ```

4. **Write the Code**:
   - Create a Python script named `mcp_assistant_openai.py` with the following content:
     ```python
     import openai
     import os

     # Set your OpenAI API key (replace with actual key or use environment variable)
     openai.api_key = 'your_openai_api_key_here'

     # System prompt for the assistant
     system_prompt = """
     You are an assistant for the Hugging Face MCP Course. Your role is to help users understand the Model Context Protocol (MCP), answer questions about course content, and provide guidance on assignments. Be knowledgeable about AI, machine learning, and MCP concepts.
     """

     # Function to get responses
     def get_response(user_input):
         try:
             response = openai.ChatCompletion.create(
                 model="gpt-3.5-turbo",
                 messages=[
                     {"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_input}
                 ],
                 max_tokens=150,
                 temperature=0.7
             )
             return response['choices'][0]['message']['content']
         except Exception as e:
             return f"Error: {str(e)}"

     # Main interaction loop
     if __name__ == "__main__":
         print("Welcome to the MCP Course Assistant (OpenAI)!")
         print("Type 'exit' to quit.")
         while True:
             user_input = input("You: ")
             if user_input.lower() == 'exit':
                 break
             assistant_response = get_response(user_input)
             print("Assistant:", assistant_response)
     ```

5. **Run the Assistant**:
   - Execute the script:
     ```bash
     python mcp_assistant_openai.py
     ```
   - Test with questions like “What is MCP?” or “How do I set up tools for Unit 0?”

6. **Optional Deployment**:
   - Deploy on a platform like Heroku or AWS for online access, or share the script for local use.

## Part 2: Google Gemini Assistant (Vertex AI)

### Objective
Create a Python-based chatbot using Google’s Gemini model via Vertex AI’s free tier to assist with the MCP Course.

### Steps

1. **Set Up Google Cloud Account**:
   - Create a project at [Google Cloud Console](https://console.cloud.google.com/).
   - Enable billing to access $300 in free credits for new users.
   - Note your project ID and location (e.g., `us-central1`).

2. **Enable Vertex AI API**:
   - Go to **APIs & Services > Enabled APIs & Services**.
   - Enable the Vertex AI API.

3. **Install Required Libraries**:
   - Install the `google-cloud-aiplatform` library:
     ```bash
     pip install google-cloud-aiplatform
     ```

4. **Set Up Authentication**:
   - Authenticate using:
     ```bash
     gcloud auth application-default login
     ```
   - Alternatively, use a service account key from **IAM & Admin > Service Accounts**.

5. **Write the Code**:
   - Create a Python script named `mcp_assistant_google.py` with the following content:
     ```python
     from google.cloud import aiplatform

     # Initialize Vertex AI
     aiplatform.init(project="your-project-id", location="us-central1")

     # System prompt for the assistant
     system_prompt = """
     You are an assistant for the Hugging Face MCP Course. Your role is to help users understand the Model Context Protocol (MCP), answer questions about course content, and provide guidance on assignments. Be knowledgeable about AI, machine learning, and MCP concepts.
     """

     # Function to get responses
     def get_response(user_input):
         try:
             model = aiplatform.ChatModel.from_model_id("gemini-1.0-pro")
             chat = model.start_chat(context=system_prompt)
             response = chat.send_message(user_input)
             return response.text
         except Exception as e:
             return f"Error: {str(e)}"

     # Main interaction loop
     if __name__ == "__main__":
         print("Welcome to the MCP Course Assistant (Google Gemini)!")
         print("Type 'exit' to quit.")
         while True:
             user_input = input("You: ")
             if user_input.lower() == 'exit':
                 break
             assistant_response = get_response(user_input)
             print("Assistant:", assistant_response)
     ```

6. **Run the Assistant**:
   - Execute the script:
     ```bash
     python mcp_assistant_google.py
     ```
   - Test with similar questions as the OpenAI version.

7. **Monitor Usage**:
   - Check usage in **Billing** to stay within the $300 free credit limit.

## Testing and Validation
- Test both assistants with sample questions:
  - “What is the Model Context Protocol?”
  - “Explain Unit 1 concepts.”
  - “Help with Unit 2 assignment.”
- Ensure responses are accurate and relevant to the MCP Course.

## Notes
- **OpenAI Costs**: Usage incurs costs based on token consumption.
- **Google Free Tier**: Limited to $300 in credits; monitor to avoid charges.
- **Customization**: Adjust prompts or add course-specific details as needed.
- **Deployment**: Consider web deployment for broader access.