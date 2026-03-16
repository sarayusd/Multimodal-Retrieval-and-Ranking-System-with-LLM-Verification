
from openai import OpenAI

client = OpenAI()


def generate_scene_explanation(query, context):

    prompt = f"""
Explain the scene using the provided evidence.

Query:
{query}

Evidence:
{context}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content