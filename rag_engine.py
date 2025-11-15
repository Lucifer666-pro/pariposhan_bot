import pickle
import faiss
import os
from groq import Groq
from sentence_transformers import SentenceTransformer

# Initialize Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load FAISS index
index = faiss.read_index("data/index.faiss")

# Load text chunks
with open("data/index.pkl", "rb") as f:
    store = pickle.load(f)

texts = store["texts"]

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_context(query):
    emb = embedder.encode([query])
    scores, ids = index.search(emb, k=3)

    context = "\n\n".join(texts[i] for i in ids[0])
    return context


def answer_from_rag(question):
    context = retrieve_context(question)

    prompt = f"""
You are Pariposhan, a food safety assistant.
Use ONLY the context below (from official FSSAI PDFs).
If answer is not available, say: "Please check official FSSAI source."

Context:
{context}

Question: {question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
