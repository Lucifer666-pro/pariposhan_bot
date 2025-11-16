import pickle
import faiss
import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load FAISS index
index = faiss.read_index("data/index.faiss")

# Load text chunks
with open("data/index.pkl", "rb") as f:
    store = pickle.load(f)

texts = store["texts"]

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_context(query):
    emb = embedder.encode([query])
    scores, ids = index.search(emb, k=3)
    return "\n\n".join(texts[i] for i in ids[0])


def answer_from_rag(question):
    context = retrieve_context(question)

    prompt = f"""
You are Pariposhan, a Food Safety and FSSAI assistant.
Use ONLY the provided context below.

If answer cannot be found, reply:
"Please check official FSSAI source."

Context:
{context}

Question: {question}

Answer:
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
