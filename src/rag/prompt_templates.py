
from langchain_core.prompts import PromptTemplate


scene_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""
You are a grounded scene explanation assistant.

Use ONLY the evidence captions to explain the scene.

Query:
{query}

Evidence:
{context}

Answer:
"""
)