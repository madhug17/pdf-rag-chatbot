import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "pizza recipe"
]
print("INPUT TEXTS")
print("-" * 40)

for i, text in enumerate(texts, 1):
    print(f"{i}. '{text}'")
embeddings = model.encode(texts)

print("\nEMBEDDING INFO")
print("-" * 40)
print(f"Number of texts: {len(texts)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"First 5 numbers of first embedding: {embeddings[0][:5]}")

print("\nDETAILED OUTPUT")
print("-" * 40)

for i, (t, emb) in enumerate(zip(texts, embeddings), 1):
    print(f"{i}. Text: {t}")
    print(f"   First 5 numbers: {emb[:5]}")
