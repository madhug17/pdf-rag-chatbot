from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

# -----------------------------
# PART 1: BASIC SIMILARITY
# -----------------------------

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Python is a programming language",
    "Machine learning is a subset of AI",
    "Pizza is made with dough and cheese",
    "Neural networks are used in deep learning",
    "Italy is famous for pasta and pizza"
]

# Print documents
for i, doc in enumerate(documents, 1):
    print(f"{i}. {doc}")

# Encode documents
document_embeddings = model.encode(documents)

query = "What is artificial intelligence?"
query_embedding = model.encode(query)

# Cosine similarity
theta = cosine_similarity([query_embedding], document_embeddings)[0]

print("\nSimilarity Scores:")
for i, (doc, score) in enumerate(zip(documents, theta), 1):
    print(f"{i}. [{score:.3f}] {doc}")

# Top 2 results
top_indices = np.argsort(theta)[::-1][:2]

print("\nTop Results:")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. {documents[idx]}")
    print(f"   Similarity: {theta[idx]:.3f}")

# -----------------------------
# PART 2: VECTOR DATABASE (ChromaDB)
# -----------------------------

print("\n" + "="*60)
print("TASK 5: VECTOR DATABASE (ChromaDB)")
print("="*60)

client = chromadb.Client()
collection = client.create_collection("my_documents")

collection.add(
    embeddings=document_embeddings.tolist(),
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"Stored {collection.count()} documents in database")

query = "What is AI?"
print("\n" + "="*60)
print(f"SEARCH: '{query}'")
print("="*60)

query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=2
)

print("\nRetrieved Documents:")
for i, doc in enumerate(results["documents"][0], 1):
    print(f"{i}. {doc}")

print("\nYou just built a mini RAG system.")
