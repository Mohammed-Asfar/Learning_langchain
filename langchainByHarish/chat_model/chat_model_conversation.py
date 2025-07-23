from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

messages = [
    SystemMessage("You are an expert in social media content strategy"),
    HumanMessage("Give a short tip to create engagaing posts on Instagram"),
]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result = llm.invoke(messages)

print(result.content)
