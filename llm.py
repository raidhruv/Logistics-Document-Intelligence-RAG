import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)


def build_prompt(query: str, approved: list[dict]):

    context = "\n\n".join([r["metadata"]["text"] for r in approved])

    prompt = f"""
You are a logistics compliance assistant.

You must answer using ONLY the provided context below.
Do NOT use external knowledge.
Do NOT make assumptions.

If the answer cannot be derived from the context,
respond exactly with:

"Insufficient information in provided document."

---------------------
Context:
{context}
---------------------

Question:
{query}

Answer:
"""

    return prompt




def call_granite(prompt: str) -> str:
    response = client.chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()





    return response.strip()
