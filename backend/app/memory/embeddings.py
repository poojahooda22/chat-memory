"""Turn text into an embedding vector via the one OpenAI-compatible client.

The client's base_url decides the provider (AI Gateway with text-embedding-3-small by
default, or Ollama locally); this function does not care which is behind it.
"""

from typing import Protocol


class _EmbeddingClient(Protocol):
    # minimal shape we depend on — lets tests pass a fake without importing openai
    class embeddings:  # noqa: N801
        @staticmethod
        def create(model: str, input: str): ...


def embed_text(client, model: str, text: str) -> list[float]:
    response = client.embeddings.create(model=model, input=text)
    return list(response.data[0].embedding)