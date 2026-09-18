import os
import tempfile
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="DocMind", page_icon="🧠")

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def load_pdf_documents(file_path: str):
    documents = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(Document(page_content=text, metadata={"page": idx}))
    except Exception:
        reader = PdfReader(file_path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(Document(page_content=text, metadata={"page": idx}))
    return documents

def ingest_pdf_bytes(file_bytes: bytes):
    embeddings = get_embeddings()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        documents = load_pdf_documents(tmp_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        chunks = splitter.split_documents(documents)
        valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
        db = Chroma.from_documents(valid_chunks, embeddings)
    finally:
        os.unlink(tmp_path)
    return db, len(documents), len(valid_chunks)

st.title("DocMind - Document Analyzer")
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Processing..."):
        db, pages, chunks = ingest_pdf_bytes(uploaded_file.getvalue())
        st.success(f"Indexed {pages} pages ({chunks} chunks)")
