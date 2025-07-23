from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate.from_template(
    "Give 3 {tone} marketing slogans for a product in {industry}"
)

partial_prompt = prompt.partial(tone="Bold")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chain = LLMChain(llm=llm, prompt=partial_prompt)

print(chain.run(industry="fashion"))
