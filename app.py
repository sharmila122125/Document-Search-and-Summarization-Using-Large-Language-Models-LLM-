import streamlit as st
import faiss
import pickle
import subprocess
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
TOP_K = 3
CONTEXT_LIMIT = 2000

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Amazon River RAG", layout="centered")
st.title("🌊 Amazon River – RAG with Ollama")
st.caption("FAISS + Local LLM (Ollama)")

# ---------------- LOAD RESOURCES ----------------
@st.cache_resource
def load_resources():
    index = faiss.read_index("amazon_index.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, embed_model

index, chunks, embed_model = load_resources()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")
answer_style = st.sidebar.selectbox(
    "Answer style", ["Very short", "Short", "Detailed"]
)

# ---------------- USER INPUT ----------------
query = st.text_input("Ask a question about the Amazon River")

def clean_text(text):
    return text.encode("utf-8", errors="ignore").decode("utf-8")

# ---------------- RAG FUNCTION ----------------
def query_rag(question):
    q_embedding = embed_model.encode([question])
    _, indices = index.search(q_embedding, TOP_K)

    context = "\n\n".join([chunks[i] for i in indices[0]])
    context = clean_text(context)[:CONTEXT_LIMIT]

    if answer_style == "Very short":
        instruction = "Answer in 2–3 sentences."
    elif answer_style == "Short":
        instruction = "Answer in 4–5 sentences."
    else:
        instruction = "Explain clearly with important details."

    prompt = f"""
{instruction}

Use ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    return result.stdout.strip(), context

# ---------------- BUTTON ----------------
if st.button("🔍 Search & Summarize"):
    if not query.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking with Ollama..."):
            answer, sources = query_rag(query)

        st.subheader("📌 Answer")
        st.success(answer)

        with st.expander("📄 Retrieved Context"):
            st.write(sources)
