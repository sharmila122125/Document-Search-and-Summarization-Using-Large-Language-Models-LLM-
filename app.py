import streamlit as st
import faiss
import pickle
import subprocess
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config(page_title="Amazon River RAG", layout="centered")
st.title("🌊 Amazon River – RAG with Ollama")
st.caption("FAISS + Local LLM (Ollama)")

# Load resources
@st.cache_resource
def load_resources():
    index = faiss.read_index("amazon_index.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, embed_model

index, chunks, embed_model = load_resources()

# Sidebar
st.sidebar.header("⚙️ Settings")
answer_style = st.sidebar.selectbox(
    "Answer style", ["Very short", "Short", "Detailed"]
)

# User input
query = st.text_input("Ask a question about the Amazon River")

# RAG logic
def query_rag(question):
    q_embedding = embed_model.encode([question])
    _, indices = index.search(q_embedding, 2)

    context = "\n\n".join([chunks[i] for i in indices[0]])

    if answer_style == "Very short":
        instruction = "Answer in 2–3 sentences."
    elif answer_style == "Short":
        instruction = "Answer in 4–5 sentences."
    else:
        instruction = "Explain clearly with details."

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
        text=True,
        capture_output=True
    )

    return result.stdout, context

# Button
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
