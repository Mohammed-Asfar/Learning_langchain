from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor, initialize_agent
from langchain.tools import tool
from dotenv import load_dotenv
import datetime

load_dotenv()


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""
    current_time = datetime.datetime.now().strftime(format)
    return current_time


query = "What is the current time? Just show the current time not date"

prompt = hub.pull("hwchase17/react")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

tools = [
    get_system_time,
]

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

result = agent_executor.invoke({"input": query})

print(result)
