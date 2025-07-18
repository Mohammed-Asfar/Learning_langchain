from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.shell import ShellTool
from langchain.agents import initialize_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

tools = [ShellTool()]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("close the chrome using cmd")
