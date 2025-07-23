# Lesson 6: ChatPromptTemplate — Multi-turn Conversational Prompts in LangChain v0.3.26 💬🤖
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains.llm import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.runnables import Runnable

load_dotenv()

# Step 1: Define the structured chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a helpful and friendly Python tutor. And your name is kana."
        ),
        HumanMessage(content="{question}"),
    ]
)

# Step 2: Create the chain with Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
chain: Runnable = prompt | llm

# Step 3: Ask a question
print(
    chain.invoke(
        {"question": "What is the difference between a list and a tuple in Python?"}
    )
)
