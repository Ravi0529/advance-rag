from langchain_qdrant import QdrantVectorStore
from embedder import get_embedder


def get_retriever(qdrant_url: str, collection_name: str):
    embedder = get_embedder()
    return QdrantVectorStore.from_existing_collection(
        url=qdrant_url,
        collection_name=collection_name,
        embedding=embedder,
    )
