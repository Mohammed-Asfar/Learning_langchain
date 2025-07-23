from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template = "write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"

prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke(
    {
        "tone": "energentic",
        "company": "samsung",
        "position": "AI Engineer",
        "skill": "AI",
    }
)

result = llm.invoke(prompt)

print(result.content)
