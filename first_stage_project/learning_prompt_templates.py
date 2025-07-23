from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate.from_template(
    "Suggest 3 startup ideas in the field of {industry} for a beginner."
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chains = LLMChain(llm=llm, prompt=prompt)

print(chains.run(industry="Software"))
