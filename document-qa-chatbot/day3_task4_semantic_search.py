from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [
    "Python is a programming language",
    "Machine learning is a subset of AI",
    "Pizza is made with dough and cheese",
    "Neural networks are used in deep learning",
    "Italy is famous for pasta and pizza"
]
for i , doc in enumerate(documents,1):
    print(f"{i},'{doc}'")
doc_embeddings= model.encode(documents)
print(f"the len of documents {len(doc_embeddings)}")
query = "what is artifical intelligent ?"
query_embedding = model.encode(query)
sim = cosine_similarity([query_embedding],doc_embeddings)[0]
for i, (doc,score) in enumerate(zip(documents,sim),1):
    print(f"{i}.[{score:.3f}]{doc}")
top= np.argsort(sim)[::-1][:2]
for rank, idx in enumerate(top,1):
    print(f'{rank},{doc}')
    print(f"similarity:{sim[idx]:.3f}")