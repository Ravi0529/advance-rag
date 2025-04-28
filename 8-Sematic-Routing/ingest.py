from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from embedder import get_embedder


def ingest_pdf(pdf_path, collection_name):
    loader = PyPDFLoader(file_path=pdf_path)
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
        collection_name=collection_name,
        embedding=embedder,
    )

    vector_store.add_documents(documents=split_docs)

    print(f"Ingested successfully!, '{collection_name}'")


def ingest_pdf_to_qdrant():
    base_path = Path(__file__).parent / "data"

    pdfs = {
        "nodejs.pdf": "nodejs-collection",
        "python.pdf": "python-collection",
    }

    for pdf_file, collection_name in pdfs.items():
        pdf_path = base_path / pdf_file
        ingest_pdf(pdf_path, collection_name)


if __name__ == "__main__":
    ingest_pdf_to_qdrant()
