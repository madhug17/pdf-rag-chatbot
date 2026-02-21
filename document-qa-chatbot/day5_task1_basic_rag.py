from pypdf import PdfReader
import re
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- LOAD PDF ----------------
print("Processing PDF...")

pdf_path = r"data\ML_CaseStudy_AIMLRhinos_EvenSem_2025-26.pdf"
reader = PdfReader(pdf_path)

raw_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        raw_text += text

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)   # FIXED
    text = re.sub(r" \n", "\n", text)
    text = re.sub(r"\n ", "\n", text)
    return text.strip()

clean_text_results = clean_text(raw_text)

# ---------------- SPLIT ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(clean_text_results)
print(f"Created {len(chunks)} chunks")

# ---------------- EMBEDDINGS ----------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# ---------------- VECTOR DB ----------------
client = chromadb.Client()
collection = client.create_collection("rag_with_llm")

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))]  # removed space
)

print(f"Stored {collection.count()} chunks")

# ---------------- QUESTION ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

user_question = "What is this case study about?"
"What methods are discussed?"
"What are the key findings?"
"Who are the authors?"
"What is machine learning?"
"What is supervised learning?"
"What applications are mentioned?"

question_embedding = model.encode(user_question)

results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=3
)

relevant_chunks = results["documents"][0]   # FIXED shape

# ---------------- BUILD PROMPT ----------------
context = "\n\n".join(relevant_chunks)

prompt = f"""Based on the following context, answer the question clearly and concisely.

Context:
{context}

Question: {user_question}

Answer:"""

print("Sending to Gemini...")

response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)
