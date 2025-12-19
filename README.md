🌊 Amazon River RAG Application

Document Search & Summarization using FAISS + Ollama + Streamlit

📌 Project Overview

This project is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about the Amazon River using a PDF document as the knowledge source.

The system:

Retrieves relevant document sections using FAISS vector search

Generates accurate answers using a local LLM powered by Ollama

Provides a clean and interactive Streamlit UI

⚡ Fully offline | No API limits | No cloud dependency

🧠 Architecture
PDF Document
     ↓
Text Chunking
     ↓
Sentence Embeddings
     ↓
FAISS Vector Index
     ↓
User Question
     ↓
Semantic Search (FAISS)
     ↓
Retrieved Context
     ↓
Ollama LLM (llama3)
     ↓
Final Answer (Streamlit UI)

🛠️ Tech Stack
Component	Technology
Frontend	Streamlit
Vector Search	FAISS
Embeddings	Sentence Transformers (all-MiniLM-L6-v2)
LLM	Ollama (llama3)
Language	Python
📁 Project Structure
GenAI_Project/
│
├── ingest.py              # PDF ingestion + FAISS index creation
├── app.py                 # Streamlit RAG application
├── Amazon_River.pdf       # Input document
├── amazon_index.faiss     # FAISS vector index
├── chunks.pkl             # Stored document chunks
└── README.md

⚙️ Setup Instructions
1️⃣ Create Virtual Environment
python -m venv .venv
source .venv/Scripts/activate   # Windows

2️⃣ Install Dependencies
pip install streamlit faiss-cpu sentence-transformers pypdf

🧠 Install Ollama (Local LLM)
Download Ollama

👉 https://ollama.com/download

Pull Model
ollama pull llama3

Verify
ollama run llama3

📥 Step 1: Build FAISS Index

Run the ingestion script to process the PDF:

python ingest.py


This will generate:

amazon_index.faiss

chunks.pkl

🚀 Step 2: Run the Application
streamlit run app.py


Open browser at:

http://localhost:8501
