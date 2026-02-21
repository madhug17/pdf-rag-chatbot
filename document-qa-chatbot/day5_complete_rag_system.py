from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import re
import chromadb
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("Complete RAG System - Production Ready")
print("=" * 70)

# ---------------- LOAD & PROCESS PDF ----------------
pdf_path = r"data\ML_CaseStudy_AIMLRhinos_EvenSem_2025-26.pdf"

try:
    reader = PdfReader(pdf_path)
    raw_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    def clean_text(text):
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" \n", "\n", text)
        return text.strip()

    clean_text_result = clean_text(raw_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(clean_text_result)
    print(f"Created {len(chunks)} chunks")

except Exception as e:
    print(f"Error processing PDF: {e}")
    exit()

# ---------------- EMBEDDINGS ----------------
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    client = chromadb.Client()
    collection = client.create_collection("complete_rag_system")

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    print(f"Stored {collection.count()} chunks")

except Exception as e:
    print(f"Error creating embeddings: {e}")
    exit()

# ---------------- LLM ----------------
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )
    print("LLM Ready")

except Exception as e:
    print(f"Error loading LLM: {e}")
    exit()

# ---------------- RAG FUNCTION ----------------
def ask_question(question, style="detailed", show_sources=True):

    question_embedding = model.encode(question)

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=4
    )

    rel_chunks = results["documents"][0]
    context = "\n\n".join(rel_chunks)

    if style == "bullets":
        instruction = "Answer using bullet points."
    elif style == "concise":
        instruction = "Answer concisely in 3-4 sentences."
    else:
        instruction = "Provide a detailed and comprehensive answer."

    prompt = f"""
Based on the following context, {instruction}

Context:
{context}

Question: {question}

Answer:
"""

    try:
        response = llm.invoke(prompt)
        answer = response.content

        if show_sources:
            answer += "\n\nSources:\n"
            for i in range(len(rel_chunks)):
                answer += f"- Chunk {i+1}\n"

        return answer

    except Exception as e:
        return f"Error generating answer: {e}"

# ---------------- INTERACTIVE MODE ----------------
def run_chat():
    current_style = "detailed"
    show_sources = True

    print("\nType your question. Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower().startswith("style"):
            style = user_input.split()[-1]
            if style in ["detailed", "concise", "bullets"]:
                current_style = style
                print(f"Switched to {style} mode")
            else:
                print("Invalid style")
            continue

        if user_input.lower() == "sources on":
            show_sources = True
            print("Sources enabled")
            continue

        if user_input.lower() == "sources off":
            show_sources = False
            print("Sources disabled")
            continue

        print("\nAssistant:")
        answer = ask_question(
            user_input,
            style=current_style,
            show_sources=show_sources
        )
        print(answer)

# ---------------- MAIN MENU ----------------
while True:
    print("\n1. Interactive Chat")
    print("2. Exit")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        run_chat()
    elif choice == "2":
        break
    else:
        print("Invalid choice")

print("\nSession Ended")
