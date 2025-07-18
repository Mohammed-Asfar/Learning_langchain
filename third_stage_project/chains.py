from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.tavily_search import TavilySearchResults
from langchain.tools.shell import ShellTool
from langchain.memory import ConversationBufferMemory
from langchain.agents.initialize import initialize_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

memory = ConversationBufferMemory(memory_key="chat_history")

tools = [TavilySearchResults(), ShellTool()]

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

while True:
    n = input(">>")
    if n == "exit":
        break
    print(agent.run(n))
