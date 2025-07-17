from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

tools = [TavilySearchResults()]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("What is the population of the United States in 2025?")
