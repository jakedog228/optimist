import os
import openai
from typing import List, Optional, Dict

from openai.types.chat import ChatCompletionMessage

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
MODEL_NAME           = os.getenv("COMPLETION_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

llm = openai.OpenAI(
    api_key=OPENAI_API_KEY,
)


def chat_completion(messages: list[dict], *, tools: Optional[List[Dict]] = None) -> ChatCompletionMessage:
    """
    Send a request to OpenAI's chat completion endpoint, using a pre-defined model.
    Returns the full response message object, which includes the assistant's reply.
    """

    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools

    resp = llm.chat.completions.create(**kwargs)

    return resp.choices[0].message

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
