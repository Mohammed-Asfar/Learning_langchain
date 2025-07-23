from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.tavily_search import TavilySearchResults
from langchain.tools.shell import ShellTool
from langchain.memory import ConversationBufferMemory
from langchain.agents.initialize import initialize_agent
from langchain.agents import AgentType
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define system message
system_message = SystemMessagePromptTemplate.from_template(
    """
You are Kana, an advanced AI assistant developed by Mohammed Asfar.
You are designed to help the user, whose name is also Mohammed Asfar.
You have access to tools like web search and shell commands, and you can use them whenever needed.
Always be helpful, respectful, and accuclearrate in your responses.
Only use tools when necessary to fulfill the user’s request.
"""
)

# Human message template
human_message = HumanMessagePromptTemplate.from_template("{input}")

# Chat prompt structure
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

# Memory to retain chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Tools
tools = [TavilySearchResults(), ShellTool()]

# Final agent setup with custom prompt and system message
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# CLI loop
print("Kana is ready. Type 'exit' to quit.")
while True:
    user_input = input(">> ")
    if user_input.strip().lower() == "exit":
        break
    try:
        # Format the input with the prompt (to include the system message context)
        formatted_input = chat_prompt.format_messages(input=user_input)
        response = agent.invoke({"input": user_input})
        print(response)
    except Exception as e:
        print(f"[ERROR] {e}")
