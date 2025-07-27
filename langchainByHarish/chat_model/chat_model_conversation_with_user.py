from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat_history = [
    SystemMessage("Your name is Kana. You are an assistant of Mohammed Asfar."),
]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

while True:
    n = input(">>")
    if n == "exit":
        break
    chat_history.append(HumanMessage(n))
    result = llm.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print(result.content)
