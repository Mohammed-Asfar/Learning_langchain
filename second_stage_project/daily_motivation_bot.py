from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time as t

load_dotenv()

prompt = PromptTemplate.from_template(
    "Give me a short, powerful daily motivation quote or message that will inspire [your audience: me / students / entrepreneurs / fitness lovers / etc.] to start their day with [emotion: confidence / hope / energy / gratitude / discipline]. Keep it [tone: positive / tough-love / spiritual / poetic / practical], and under [word limit: 30/50/100] words."
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

chains = prompt | llm

while True:
    res = chains.invoke(
        {
            "input": "Create a 1-line motivational quote for entrepreneurs to stay consistent and focused. Tone: tough-love, Limit: 30 words."
        }
    )
    print(res.content)
    t.sleep(86400)  # 24 hours
