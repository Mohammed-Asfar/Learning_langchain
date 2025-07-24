from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableLambda, RunnableSequence

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows fact about {animal}."),
        ("human", "Tell me {fact_count} facts."),
    ]
)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_llm = RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)


chain = RunnableSequence(first=format_prompt, middle=[invoke_llm], last=parse_output)


res = chain.invoke({"animal": "cat", "fact_count": 2})

print(res)
