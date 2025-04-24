from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from embedder import get_embedder


def ingest_pdf_to_qdrant():
    pdf_file_path = Path(__file__).parent / "data" / "nodejs.pdf"
    loader = PyPDFLoader(file_path=pdf_file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(documents=docs)

    embedder = get_embedder()
    vector_store = QdrantVectorStore.from_documents(
        documents=[],
        url="http://localhost:6333",
        collection_name="step_back_prompting",
        embedding=embedder,
    )
    vector_store.add_documents(documents=split_docs)

    print("Ingested successfully")


if __name__ == "__main__":
    ingest_pdf_to_qdrant()
