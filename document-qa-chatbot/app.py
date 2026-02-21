import streamlit as st
from pypdf import PdfReader
import re
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

st.title("📄 PDF RAG Chatbot")
st.markdown("**Upload a PDF and ask questions!**")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("📁 Upload PDF", type=['pdf'])
    answer_style = st.selectbox("💬 Style", ["detailed", "concise", "bullets"])
    show_sources = st.checkbox("📚 Show Sources", value=True)
    if st.button("🗑️ Clear"):
        st.session_state.messages = []
        st.rerun()

def process_pdf(pdf_file):
    with st.spinner("Processing..."):
        try:
            reader = PdfReader(pdf_file)
            raw_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text
            
            clean_text = re.sub(r" +", " ", re.sub(r"\n{3,}", "\n\n", re.sub(r" \n", "\n", raw_text))).strip()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(clean_text)
            
            if st.session_state.model is None:
                st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")
            
            embeddings = st.session_state.model.encode(chunks)
            client = chromadb.Client()
            
            try:
                client.delete_collection("pdf_chatbot")
            except:
                pass
            
            collection = client.create_collection("pdf_chatbot")
            collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(),
                ids=[f"chunk_{i}" for i in range(len(chunks))]
            )
            
            st.session_state.collection = collection
            
            if st.session_state.llm is None:
                st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            
            st.session_state.processed = True
            return len(chunks)
        except Exception as e:
            st.error(f"Error: {e}")
            return None

def answer_question(question, style, sources):
    if not st.session_state.processed:
        return "Upload PDF first!"
    
    try:
        q_emb = st.session_state.model.encode(question)
        results = st.session_state.collection.query(
            query_embeddings=[q_emb.tolist()], n_results=3
        )
        chunks = results["documents"][0]
        context = "\n\n".join(chunks)
        
        inst = "Answer using bullet points." if style == "bullets" else (
            "Answer in 2-3 sentences." if style == "concise" else "Provide detailed answer."
        )
        
        prompt = f"Based on context, {inst}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        response = st.session_state.llm.invoke(prompt)
        answer = response.content
        
        if sources:
            answer += "\n\n**Sources:**\n" + "\n".join([f"- Chunk {i+1}: {c[:100]}..." for i, c in enumerate(chunks)])
        
        return answer
    except Exception as e:
        return f"Error: {e}"

if uploaded_file:
    if not st.session_state.processed:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        chunks = process_pdf(tmp_path)
        os.unlink(tmp_path)
        
        if chunks:
            st.success(f"✅ PDF processed! {chunks} chunks created.")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = answer_question(prompt, answer_style, show_sources)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👈 Upload a PDF to start!")
    st.markdown("### How to use:\n1. Upload PDF\n2. Ask questions\n3. Get answers!")

st.markdown("---")
st.markdown("Built with Streamlit • Powered by Gemini")