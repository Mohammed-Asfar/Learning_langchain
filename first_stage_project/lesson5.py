# Lesson 5: FewShotPromptTemplate

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv

load_dotenv()

# Step 1: Format for each example
example_template = PromptTemplate.from_template(
    "Concept: {concept}\nAnalogy: {analogy}\n"
)

# Step 2: Few-shot examples
examples = [
    {"concept": "Internet", "analogy": "Like a spiderweb connecting all computers."},
    {
        "concept": "CPU",
        "analogy": "Like the brain of a computer, controlling all actions.",
    },
    {"concept": "RAM", "analogy": "Like a desk: temporary space to work on tasks."},
]

# Step 3: FewShotPromptTemplate setup
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    suffix="Concept: {concept}\nAnalogy:",
    input_variables=["concept"],
)

# Step 4: Load model and chain
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
chain = LLMChain(llm=llm, prompt=few_shot_prompt)

# Step 5: Run it
print(chain.run(concept="Firewall"))
