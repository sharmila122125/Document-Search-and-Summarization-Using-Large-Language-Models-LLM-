from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load PDF
reader = PdfReader("Amazon_River.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# Chunking
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(text)

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "amazon_index.faiss")

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ FAISS index created")
