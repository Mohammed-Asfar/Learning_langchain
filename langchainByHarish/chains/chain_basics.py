from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows fact about {animal}."),
        ("human", "Tell me {fact_count} facts."),
    ]
)

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"animal": "cat", "fact_count": 2})

print(result)
