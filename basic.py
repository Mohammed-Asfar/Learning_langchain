# 🚀 Project 1: Basic Q&A Bot
# Goal: Use LangChain to answer questions using an LLM (no tools).

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate.from_template("What is {topic}?")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
)

chains = prompt | llm

res = chains.invoke({"topic": "quntum physics"})

print(res.content)
