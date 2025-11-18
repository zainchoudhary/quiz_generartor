import os
import chromadb
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document


google_api_key = os.environ["GOOGLE_API_KEY"]

PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "quiz_generator_documents"

try:

    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

except Exception as e:
    print(
        f"⚠️ RAG Pipeline Error: Failed to initialize embeddings or ChromaDB. Ensure GOOGLE_API_KEY is set. Error: {e}")
    embedding_model = None
    chroma_client = None



def index_document(text: str, file_hash: str) -> Optional[Chroma]:
    if not embedding_model or not chroma_client:
        return None

    doc_collection_name = f"{COLLECTION_NAME}_{file_hash}"

    try:
        if doc_collection_name in [c.name for c in chroma_client.list_collections()]:
            print(f"ChromaDB: Found existing index for document {file_hash}.")
            vector_store = Chroma(
                client=chroma_client,
                collection_name=doc_collection_name,
                embedding_function=embedding_model,
            )
            return vector_store

    except Exception as e:
        print(f"ChromaDB Check Error: {e}. Proceeding to re-index.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    doc = Document(page_content=text, metadata={"file_hash": file_hash})
    chunks = text_splitter.split_documents([doc])

    if not chunks:
        return None

    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            client=chroma_client,
            collection_name=doc_collection_name,
        )
        print(f"ChromaDB: Created new index for document {file_hash} with {len(chunks)} chunks.")
        return vector_store
    except Exception as e:
        print(f"ChromaDB Indexing Error: {e}")
        return None


def retrieve_context(vector_store: Chroma, topic: str, k: int = 5) -> str:

    if not vector_store:
        return ""

    docs = vector_store.similarity_search(topic, k=k)

    context = "\n---\n".join([doc.page_content for doc in docs])
    return context


def run_rag_pipeline(text: str, topic: str, file_hash: str) -> str:
    """
    Main function to run the RAG process with persistence check.
    """
    if not text and topic:
        # User entered a manual topic
        return topic

    if text:
        # 1. Index the document (will retrieve existing if hash matches)
        vector_store = index_document(text, file_hash)

        if vector_store:
            # 2. Retrieve relevant context
            query = topic if topic else text[:100]
            retrieved_context = retrieve_context(vector_store, query, k=5)

            if retrieved_context:
                print(f"RAG: Successfully retrieved {len(retrieved_context.split('---'))} context chunks.")
                return retrieved_context
            else:
                print("RAG: Retrieval failed, falling back to full text (max 4000 chars).")
                return text[:4000]  # Fallback
        else:
            print("RAG: Indexing failed, falling back to full text (max 4000 chars).")
            return text[:4000]  # Fallback

    return ""