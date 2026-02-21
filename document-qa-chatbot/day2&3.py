import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

document = """ Machine learning is a subset of artificial intelligence that enables 
computers to learn from data without being explicitly programmed. It has 
revolutionized many industries including healthcare, finance, and technology.

Deep learning is a specialized branch of machine learning that uses neural 
networks with multiple layers. These networks can automatically learn 
hierarchical representations of data, making them powerful for tasks like 
image recognition and natural language processing.

Supervised learning is a type of machine learning where the model is trained 
on labeled data. Each training example has an input and a corresponding 
correct output. The model learns to map inputs to outputs by finding patterns 
in the training data.

Unsupervised learning works with unlabeled data. The algorithm must discover 
patterns and structure in the data without guidance. Common techniques include 
clustering and dimensionality reduction.

Neural networks are computing systems inspired by biological neural networks. 
They consist of interconnected nodes that process information through weighted 
connections. Modern deep learning relies heavily on neural network architectures.
"""

print(f"Document length: {len(document)} characters")

# ---------------- SPLITTING ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(document)

print(f"Created {len(chunks)} chunks")

# ---------------- EMBEDDING ----------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

print(f"Created {len(embeddings)} embeddings")
print(f"Each embedding size: {embeddings.shape[1]}")

# ---------------- CHROMA DB ----------------
client = chromadb.Client()
collection = client.create_collection("ml_document")

collection.add(
    embeddings=embeddings.tolist(),
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print("Chunks stored in ChromaDB")

# ---------------- QUERY ----------------
user_question = "what is supervised learning"
question_embedding = model.encode(user_question)

results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=2
)

print("\nTop Matches from DB:")
for i, doc in enumerate(results["documents"][0], 1):
    print(f"\nMatch {i}")
    print(doc)

# ---------------- MANUAL SIMILARITY ----------------
print("\nManual Similarity Check:")

question_emb_reshaped = question_embedding.reshape(1, -1)

for i, (chunk, emb) in enumerate(zip(chunks, embeddings), 1):
    sim = cosine_similarity(
        question_emb_reshaped,
        emb.reshape(1, -1)
    )[0][0]

    print(f"\nChunk {i}: [{sim:.3f}] {chunk[:60]}")

    if sim > 0.5:
        print("         HIGH SIMILARITY")
