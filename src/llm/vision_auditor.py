import base64
from openai import OpenAI


client = OpenAI()


def encode_image_base64(path):

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def visual_auditor(query, image_path):

    img = encode_image_base64(image_path)

    prompt = f"""
You are verifying whether the image matches the query.

Query: {query}

Return JSON:

{{
 "audit_score": number between 0 and 1,
 "audit_verdict": "relevant" or "irrelevant"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": [{"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]
            }
        ]
    )

    return response