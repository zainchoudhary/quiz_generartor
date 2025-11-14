import re
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

api_key = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_output_tokens=512,
    timeout=120
)

import time

def safe_chat(prompt):
    retries = 3
    delay = 10
    for i in range(retries):
        try:
            return llm.predict(prompt)
        except Exception as e:
            print(f"Retry {i+1} failed: {e}")
            time.sleep(delay)
    raise Exception("All retries failed")


class RAG:
    """
    RAG pipeline components: Chunking, Embedding, Indexing, Retrieval, and Generation.
    """

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", chunk_size = 200):
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.chunk_size = chunk_size
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None  # FAISS Index

    def clear_documents(self):
        """Clears all stored documents, embeddings, and the FAISS index."""
        self.documents = []
        self.embeddings = []
        self.index = None

    @staticmethod
    def _chunk_text(text: str, size: int) -> List[str]:
        """Split text into chunks of roughly `size` words."""
        if not text.strip():
            return []
        words = text.split()
        chunks = []
        for i in range(0, len(words), size):
            chunks.append(" ".join(words[i:i + size]))
        return chunks

    def _embed_chunks(self, chunks: List[str]) -> List[np.ndarray]:
        """Convert a list of text chunks into embeddings."""
        if not chunks:
            return []
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        return embeddings.tolist()

    def _build_index(self):
        """Build FAISS index for fast similarity search."""
        if self.embeddings:
            embeddings_array = np.array(self.embeddings, dtype='float32')
            dim = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings_array)

    def add_document(self, text: str):
        """
        Process a single document: Chunk -> Embed -> Index.
        """
        if not text.strip():
            return

        # 1. Chunking
        new_chunks = self._chunk_text(text, self.chunk_size)

        # 2. Embedding
        new_embeddings = self._embed_chunks(new_chunks)

        # 3. Store and Index
        self.documents.extend(new_chunks)
        self.embeddings.extend(new_embeddings)
        self._build_index()

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve top_k most similar chunks from the FAISS index."""
        if not self.index or not self.documents:
            return ""

        query_emb = self.model.encode([query])[0].astype('float32')
        distances, indices = self.index.search(np.array([query_emb]), top_k)

        retrieved_docs = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        return "\n".join(retrieved_docs)

    def generate_mcqs(self, topic: str, num_mcqs: int = 5) -> List[dict]:
        """Generate MCQs using retrieved context and the LLM."""

        # Retrieval step
        context = self.retrieve(topic)

        # If no context is found, LLM will try to answer based on general knowledge
        if not context:
            prompt_context = f"Topic: {topic}"
        else:
            prompt_context = f"Context:\n{context[:4000]}"

        prompt = f"""
            Generate {num_mcqs} multiple-choice questions (MCQs) related to the given information.
            Follow the constraints strictly:
            1. Don't give any introductory or concluding sentences. Start directly with the first question.
            2. Do not use phrases like "provided text," "given text," "as in the text," "According to the text," or "as mentioned in the text."
            3. Instead, formulate the question naturally as if you are knowledgeable about the topic.
            4.Remove unnecessary context.
            5.Only send the text you need for quiz generation.
            6.Use shorter questions and answers when possible.

            Example format:
            Q1. What is AI?
            A) Option 1
            B) Option 2
            C) Option 3
            D) Option 4
            Answer: B
            Q2. ...

            {prompt_context}
            """
        try:
            result = llm.invoke(prompt)
            output_text = getattr(result, "content", None) or getattr(result, "output_text", "")
        except Exception:
            output_text = ""

        # Clean and parse the output text
        output_text = re.sub(r'(?i)q(uestion)?\s*\d*[.:]\s*', 'Q', output_text)
        output_text = output_text.replace("Option ", "").replace("Answer:", "Answer:")

        pattern = r"""Q\s*([\s\S]*?)\s*A[\)\.:]\s*([\s\S]*?)\s*B[\)\.:]\s*([\s\S]*?)\s*C[\)\.:]\s*([\s\S]*?)\s*D[\)\.:]\s*([\s\S]*?)\s*Answer[:\s]*([ABCD])"""
        matches = re.findall(pattern, output_text, re.IGNORECASE | re.VERBOSE)

        mcqs = []
        for q in matches:
            question_text, A, B, C, D, ans = q
            clean = lambda s: re.sub(r'\s+', ' ', s).strip()
            mcqs.append({
                "question": clean(question_text),
                "options": {"A": clean(A), "B": clean(B), "C": clean(C), "D": clean(D)},
                "answer": ans.strip().upper()
            })
        return mcqs