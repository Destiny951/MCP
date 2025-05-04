import json
import os
from typing import List

import aiohttp

from .util import log_title
from .vector_store import VectorStore


class EmbeddingRetriever:
    def __init__(self, embedding_model: str, api_url: str):
        self.embedding_model = embedding_model
        self.api_url = api_url
        self.vector_store = VectorStore()

    async def embed_document(self, document: str) -> List[float]:
        embedding = await self._embed(document)
        print(f"Embedding: {embedding}\n")
        await self.vector_store.add_embedding(embedding, document)
        return embedding

    async def embed_query(self, query: str) -> List[float]:
        log_title("EMBEDDING QUERY")
        return await self._embed(query)

    async def _embed(self, document: str) -> List[float]:

        payload = {
            "model": self.embedding_model,
            "input": document,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, json=payload) as response:
                async for raw_line in response.content:
                    try:
                        data = json.loads(raw_line.decode("utf-8"))
                        embeddings = data.get("embeddings", {})

                    except Exception as e:
                        print(f"解析错误: {e}")
                return embeddings[0]

    async def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = await self.embed_query(query)
        return await self.vector_store.search(query_embedding, top_k)
