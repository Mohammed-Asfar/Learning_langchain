# 💬 Project 2: Chatbot with Memory
# Goal: Make a chatbot that remembers conversation history.

# ✅ Learn:

# ConversationChain

# ConversationBufferMemory


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

memory = ConversationBufferMemory()

chains = ConversationChain(llm=llm, memory=memory, verbose=True)

while True:
    n = input(">>")
    if n == "exit":
        break
    print(chains.run(n))
