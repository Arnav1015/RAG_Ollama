import chromadb
from chromadb.config import Settings
from typing import List, Dict
import os

class ChromaVectorStore:
    def __init__(self, collection_name="hybrid_data", persist_directory="chroma_db"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, ids: List[str], texts: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def get_all_filenames(self):
        results = self.collection.get(include=["metadatas"], limit=10000)
        return set(meta["filename"] for meta in results["metadatas"] if "filename" in meta)

    def query(self, query_embedding: List[float], top_k: int = 5, where: Dict = None):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where or {}
        )

    def delete_by_ids(self, ids: List[str]):
        self.collection.delete(ids=ids)

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
