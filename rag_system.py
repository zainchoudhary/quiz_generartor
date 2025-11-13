import re
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class RAG:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", chunk_size: int = 500):
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.chunk_size = chunk_size
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None

    def add_document(self, text: str):
        """Split document into chunks and add to FAISS index"""
        if not text.strip():
            return
        chunks = self._chunk_text(text, self.chunk_size)
        for chunk in chunks:
            self.documents.append(chunk)
            emb = self.model.encode([chunk])[0]
            self.embeddings.append(emb)
        self._build_index()

    @staticmethod
    def _chunk_text(text: str, size: int) -> List[str]:
        """Split text into chunks of roughly `size` words"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), size):
            chunks.append(" ".join(words[i:i+size]))
        return chunks

    def _build_index(self):
        """Build FAISS index for fast similarity search"""
        if self.embeddings:
            dim = self.embeddings[0].shape[0]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(np.array(self.embeddings, dtype='float32'))

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve top_k most similar chunks"""
        if not self.index or not self.documents:
            return ""
        query_emb = self.model.encode([query])[0].astype('float32')
        distances, indices = self.index.search(np.array([query_emb]), top_k)
        retrieved_docs = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        return "\n".join(retrieved_docs)

    def generate_mcqs(self, topic: str, num_mcqs: int = 5) -> List[dict]:
        """Generate MCQs using retrieved context"""
        context = self.retrieve(topic)
        prompt = f"""
          Generate {num_mcqs} multiple-choice questions (MCQs) from the following text.
          Don't give
          Don't give extra information just start from the first question.
          also dont used such type of words like "provided text" and "given text" anf " as in the text?" and "According to the text" or "as mentioned in the text"
          etc , means it should not feel that yuo generate mcqs from provided text, response automatically by guessing which is asked by the user and used that topic name instead of say provided text or any other words like it

          Example format:
          Q1. What is AI?
          A) Option 1
          B) Option 2
          C) Option 3
          D) Option 4
          Answer: B

        Context:
        {context[:4000]}
        """
        try:
            result = llm.invoke(prompt)
            output_text = getattr(result, "content", None) or getattr(result, "output_text", "")
        except Exception:
            output_text = ""

        # Clean and parse
        output_text = re.sub(r'(?i)question\s*\d*[:.]', lambda m: f"Q", output_text)
        output_text = output_text.replace("Option ", "").replace("Answer:", "Answer:")

        pattern = r"""Q\d*[\.\)]?\s*([\s\S]*?)\s*A[\)\.:]\s*([\s\S]*?)\s*B[\)\.:]\s*([\s\S]*?)\s*C[\)\.:]\s*([\s\S]*?)\s*D[\)\.:]\s*([\s\S]*?)\s*Answer[:\s]*([ABCD])"""
        matches = re.findall(pattern, output_text, re.IGNORECASE | re.VERBOSE)

        mcqs = []
        for q in matches:
            question_text, A, B, C, D, ans = q
            clean = lambda s: re.sub(r'\s+', ' ', s)
            mcqs.append({
                "question": clean(question_text),
                "options": {"A": clean(A), "B": clean(B), "C": clean(C), "D": clean(D)},
                "answer": ans.strip().upper()
            })
        return mcqs