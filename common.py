import os
import openai
from typing import List

# LITELLM_API_KEY      = os.getenv("LITELLM_API_KEY")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
MODEL_NAME           = os.getenv("MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# if not LITELLM_API_KEY:
#     raise ValueError("Set LITELLM_API_KEY in your environment")


llm = openai.OpenAI(
    # base_url=os.getenv("LITELLM_PROXY_URL", "http://localhost:4000"),
    api_key=OPENAI_API_KEY,
)

def chat_completion(messages: list[dict], *, end_user: str):
    """
    Send the request through the proxy, tagging it with Discord user-id
    so spend is broken down per account in the dashboard.
    """
    resp = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        # user=end_user
    )

    return resp.choices[0].message.content.strip()

def get_embedding(text: str) -> List[float]:
    """Generates an embedding for the given text."""
    text_to_embed = text.replace("\n", " ").strip()
    if not text_to_embed:
        # Embedding models may error on empty strings or return trivial embeddings.
        raise ValueError("Cannot generate embedding for empty text.")

    resp = llm.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=[text_to_embed], # API expects a list of texts
        # user="embedding"
    )
    return resp.data[0].embedding
