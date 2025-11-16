import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

class RAGEngine:
    def __init__(self):
        # Load lightweight embedding model (Render safe)
        self.embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")

        # Load FAISS index
        self.index = faiss.read_index("data/index.faiss")

        # Load text chunks
        with open("data/index.pkl", "rb") as f:
            self.texts = pickle.load(f)["texts"]

        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm = genai.GenerativeModel("gemini-pro")

    def query(self, question, top_k=3):
        # Embed question
        q_emb = self.embedder.encode([question])
        
        # Search FAISS
        distances, indices = self.index.search(q_emb, top_k)

        # Get relevant text
        retrieved_text = "\n\n".join([self.texts[i] for i in indices[0]])

        # Create prompt
        prompt = f"""
You are a food safety assistant (Pariposhan).  
Answer only from the context provided.

CONTEXT:
{retrieved_text}

QUESTION:
{question}

ANSWER:
"""

        # Generate using Gemini
        response = self.llm.generate_content(prompt)
        return response.text
