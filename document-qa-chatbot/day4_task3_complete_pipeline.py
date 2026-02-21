from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import re

print("="*70)
print("🚀 COMPLETE RAG PIPELINE - YOUR CASE STUDY")
print("="*70)

# ---------------- STEP 1: LOAD PDF ----------------
print("\n📄 STEP 1: Load PDF")
print("-"*70)

pdf_path = r"data\ML_CaseStudy_AIMLRhinos_EvenSem_2025-26.pdf"
reader = PdfReader(pdf_path)

print(f"✅ Loaded: {pdf_path}")
print(f"   Pages: {len(reader.pages)}")

# ---------------- STEP 2: EXTRACT TEXT ----------------
print("\n📄 STEP 2: Extract Text")
print("-"*70)

raw_text = ""

for page_num, page in enumerate(reader.pages, 1):
    text = page.extract_text()
    if text:
        raw_text += text
    print(f"   Extracted page {page_num}")

print(f"\n✅ Total characters: {len(raw_text)}")

# ---------------- STEP 3: CLEAN TEXT ----------------
print("\n🧹 STEP 3: Clean Text")
print("-"*70)

def clean_text(text):
    text = re.sub(r" +", ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' \n', '\n', text)
    text = re.sub(r'\n ', '\n', text)
    return text.strip()

clean_text_result = clean_text(raw_text)

print(f"✅ Cleaned text")
print(f"   Before: {len(raw_text)} chars")
print(f"   After: {len(clean_text_result)} chars")
print(f"   Removed: {len(raw_text)-len(clean_text_result)} chars")

# ---------------- STEP 4: SPLIT INTO CHUNKS ----------------
print("\n✂️ STEP 4: Split into Chunks")
print("-"*70)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=['\n\n','\n','. ',' ','']
)

chunks = splitter.split_text(clean_text_result)

print(f"✅ Created {len(chunks)} chunks")
print(f"   Chunk size: ~300 chars")
print(f"   Overlap: 50 chars")

print(f"\nFirst 3 chunks preview:")
for i, chunk in enumerate(chunks[:3], 1):
    print(f"\n   Chunk {i} ({len(chunk)} chars):")
    print(f"   {chunk[:100]}...")

# ---------------- STEP 5: CREATE EMBEDDINGS ----------------
print("\n🔢 STEP 5: Create Embeddings")
print("-"*70)

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded!")

print("\nConverting chunks to embeddings...")
embeddings = model.encode(chunks)

print(f"✅ Created {len(embeddings)} embeddings")
print(f"   Each embedding: {embeddings.shape[1]} numbers")
print(f"   Total numbers: {len(embeddings)} × {embeddings.shape[1]} = {len(embeddings) * embeddings.shape[1]}")

# ---------------- STEP 6: STORE IN VECTOR DB ----------------
print("\n💾 STEP 6: Store in Vector Database")
print("-"*70)

print("Creating ChromaDB database...")
client = chromadb.Client()
collection = client.create_collection("case_study_db")

print("Storing chunks with embeddings...")
collection.add(
    embeddings=embeddings.tolist(),
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"✅ Stored {collection.count()} chunks in database!")

# ---------------- STEP 7: SEARCH WITH QUESTION ----------------
print("\n🔍 STEP 7: Search with Question")
print("-"*70)

user_question = "what is this case study about?"

print(f"\n❓ Question: '{user_question}'")

print("\n1. Converting question to embedding...")
question_embedding = model.encode(user_question)
print(f"   ✅ Question embedding: {question_embedding.shape}")

print("\n2. Searching database...")
results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=3
)
print(f"   ✅ Found {len(results['documents'][0])} relevant chunks")

print("\n3. Top matching chunks:")
for i, doc in enumerate(results['documents'][0], 1):
    print(f"\n   📄 Match #{i}:")
    print(f"   {doc[:200]}...")

# ---------------- COMPLETE ----------------
print("\n" + "="*70)
print("🎉 COMPLETE RAG PIPELINE WORKING!")
print("="*70)

print(f"""
✅ Successfully processed YOUR case study!

Pipeline Summary:
1. ✅ Loaded PDF ({len(reader.pages)} pages)
2. ✅ Extracted text ({len(raw_text)} chars)
3. ✅ Cleaned text ({len(clean_text_result)} chars)
4. ✅ Split into {len(chunks)} chunks
5. ✅ Created {len(embeddings)} embeddings
6. ✅ Stored in vector database
7. ✅ Searched and found relevant chunks

🚀 What's Working:
   - PDF processing ✅
   - Text cleaning ✅
   - Chunking ✅
   - Embeddings ✅
   - Vector search ✅

📝 What's Missing (Day 5):
   - LLM to generate natural answers
   - Instead of showing chunks, generate response
   - "This case study is about..."

Tomorrow: Add Gemini LLM to complete RAG! 🎯
""")