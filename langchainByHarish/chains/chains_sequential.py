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

translation_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a translator and convert the provided text into {language}",
        ),
        ("human", "Translate the following text to {language}: {text}"),
    ]
)


count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(
    lambda output: {"text": output, "language": "french"}
)

chain = (
    prompt_template
    | llm
    | StrOutputParser()
    | prepare_for_translation
    | translation_template
    | llm
    | StrOutputParser()
)


res = chain.invoke({"animal": "cat", "fact_count": 2})

print(res)
