# Lesson 4: Prompt Template Formats & Security in LangChain v0.3.26 🧠💻
from langchain_core.prompts import PromptTemplate


template = """
{% if language=='English' %}
Hello {{ name }}, welcome!

{% else %}
"Bonjour {{ name }}, bienvenue!"

{% endif %}
"""


prompt = PromptTemplate(
    template=template, template_format="jinja2", input_variables=["name", "language"]
)

print(prompt.format(name="Asfar", language="English"))
