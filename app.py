import os
import re
import time
import uuid
import tempfile
import pdfplumber
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Helpers
FALLBACK_MODELS = ["gemini-3.6-flash"]

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def get_llm(model_name: str = "gemini-3.6-flash"):
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0
    )

def _extract_text(content) -> str:
    """Normalize Gemini response content to a plain string regardless of format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def invoke_with_model_fallback(prompt_str_or_question: str, db=None, is_rag: bool = False):
    """Tries primary model; if 429 (quota) or 503 (unavailable) occurs, automatically tries fallback models."""
    last_err = None
    for model_name in FALLBACK_MODELS:
        try:
            llm = get_llm(model_name)
            if is_rag:
                rag_chain, retriever = build_rag_chain(db, llm)
                answer = rag_chain.invoke(prompt_str_or_question)
                return _extract_text(answer), retriever
            else:
                response = llm.invoke(prompt_str_or_question)
                content = response.content if hasattr(response, "content") else response
                return type("R", (), {"content": _extract_text(content)})()
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "503" in err or "UNAVAILABLE" in err:
                last_err = e
                time.sleep(1)
                continue
            else:
                raise e
    if last_err:
        raise last_err


def load_pdf_documents(file_path: str):
    """Loads PDF pages using pdfplumber structured Markdown table extraction alongside layout text."""
    documents = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                page_parts = [f"=== PAGE {idx + 1} ==="]
                
                # 1. Extract structured grid tables as Markdown tables (100% exact column-header mapping)
                try:
                    tables = page.extract_tables()
                    if tables:
                        page_parts.append("--- DETECTED TABLES (MARKDOWN GRID) ---")
                        for t_idx, table in enumerate(tables):
                            if not table:
                                continue
                            cleaned_rows = []
                            for row in table:
                                cleaned_row = [" ".join(str(cell).split()) if cell is not None else "" for cell in row]
                                if any(cleaned_row):
                                    cleaned_rows.append(cleaned_row)
                            
                            if cleaned_rows and len(cleaned_rows) > 1:
                                header = cleaned_rows[0]
                                md = f"| {' | '.join(header)} |\n| {' | '.join(['---'] * len(header))} |\n"
                                for r in cleaned_rows[1:]:
                                    md += f"| {' | '.join(r)} |\n"
                                page_parts.append(md)
                except Exception:
                    pass

                # 2. Extract layout text & standard text
                try:
                    text_layout = page.extract_text(layout=True) or ""
                    if text_layout.strip():
                        page_parts.append("--- TEXT LAYOUT VIEW ---")
                        page_parts.append(text_layout)
                except Exception:
                    pass

                try:
                    text_std = page.extract_text() or ""
                    if text_std.strip():
                        page_parts.append("--- STANDARD TEXT VIEW ---")
                        page_parts.append(text_std)
                except Exception:
                    pass

                full_text = "\n\n".join(page_parts)
                if full_text.strip():
                    documents.append(Document(
                        page_content=full_text,
                        metadata={"page": idx, "source": file_path}
                    ))
    except Exception:
        # Fallback to pypdf if pdfplumber encounters an unparseable PDF
        reader = PdfReader(file_path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text(extraction_mode="layout") or page.extract_text() or ""
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"page": idx, "source": file_path}
                ))

    return documents


def ingest_pdf_bytes(file_bytes: bytes):
    """Ingest PDF bytes → in-memory ChromaDB with isolated collection. Returns (db, page_count, chunk_count)."""
    embeddings = get_embeddings()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        documents = load_pdf_documents(tmp_path)

        if not documents:
            raise ValueError("No readable text found in this PDF. Please make sure the PDF contains extractable text and is not an image-only scan.")

        # Large chunk size (3000) keeps full timetables/tables intact in single chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        chunks = splitter.split_documents(documents)

        # Filter out empty text chunks
        valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

        if not valid_chunks:
            raise ValueError("No extractable text chunks created from this PDF.")

        # Create isolated collection with unique name to prevent cross-document contamination
        collection_name = f"doc_{uuid.uuid4().hex}"
        db = Chroma.from_documents(valid_chunks, embeddings, collection_name=collection_name)
    finally:
        os.unlink(tmp_path)

    return db, len(documents), len(valid_chunks)



def build_rag_chain(db, llm):
    """Build a fresh RAG chain from the given ChromaDB and LLM."""
    retriever = db.as_retriever(search_kwargs={"k": 8})
    qa_prompt = PromptTemplate.from_template("""You are an intelligent document assistant. Answer the user's question using the provided context from the document.

Guidelines:
- Base your answer primarily on the context provided below.
- If the context contains the answer, give a clear, detailed, and accurate response.
- For tables or timetables: carefully read column headers and row labels before answering.
- If the context does not contain enough information to answer, say "The document doesn't seem to contain that information."
- Never make up information not present in the context.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


def generate_summary(db) -> str:
    # Fetch actual text chunks directly from the current document's Chroma collection
    raw_data = db.get()
    docs = raw_data.get("documents", []) if raw_data else []
    combined = "\n\n".join(docs[:20]) if docs else ""

    prompt = f"""You are an expert document analyst. Read the following document excerpts and produce a concise, structured summary.

Format your response as:
- Start with one sentence describing what the document is about.
- Then list 5 to 7 key points as bullet points starting with •
- Keep each bullet point short (1–2 lines max).

Document excerpts:
{combined}

Summary:"""
    response = invoke_with_model_fallback(prompt, is_rag=False)
    return response.content if hasattr(response, "content") else str(response)




def parse_markdown_formatting(text: str) -> str:
    """Parses markdown bold syntax (**text**) into HTML <strong> text and formats bullet lines cleanly."""
    if not text:
        return ""
    
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^[•\-\*]\s+', stripped):
            clean_item = re.sub(r'^[•\-\*]\s+', '', stripped)
            formatted_lines.append(f"• {clean_item}")
        else:
            formatted_lines.append(line)
            
    text = "\n".join(formatted_lines)
    # Convert **bold** to <strong>bold</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong style="font-weight: 700; color: #ffffff;">\1</strong>', text)
    # Convert remaining single *italic* or _italic_ to <em>italic</em>
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    return text


def render_summary_html(text) -> str:
    # Guard: ensure text is always a plain string
    if isinstance(text, list):
        text = "\n".join(str(p) for p in text)
    else:
        text = str(text)
    lines = text.strip().split("\n")
    items = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        is_bullet = line.startswith("•") or line.startswith("-") or line.startswith("*")
        raw_content = line.lstrip("•-* ").strip()
        clean_line = parse_markdown_formatting(raw_content)

        if is_bullet:
            items += f"<li style='margin-bottom:8px; line-height: 1.6;'>{clean_line}</li>"
        else:
            items += f"<p style='color:#a78bfa; margin-bottom:10px; line-height:1.6;'>{clean_line}</p>"
    return f"""<div class="summary-text" style="padding: 0.2rem 0.5rem;">
        <ul style='padding-left:1.2rem;margin:0;color:#cbd5e1;'>{items}</ul>
    </div>"""


# Session state defaults
defaults = {
    "file_id": None, "db": None, "summary": None,
    "messages": [], "doc_name": None, "doc_pages": 0, "doc_chunks": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# Upload Section
st.markdown('<p class="section-label">Upload Your PDF</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a PDF here or click to browse",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    if st.session_state.file_id != file_id:
        # New file → reset state
        st.session_state.update({
            "messages": [], "summary": None, "db": None,
            "file_id": file_id, "doc_name": uploaded_file.name,
        })
        with st.spinner(f"⚙️ Processing **{uploaded_file.name}**…"):
            try:
                db, pages, chunks = ingest_pdf_bytes(uploaded_file.getvalue())
                st.session_state.db = db
                st.session_state.doc_pages = pages
                st.session_state.doc_chunks = chunks
                st.success(
                    f"✅ **{uploaded_file.name}** ready — {pages} page(s), {chunks} chunks indexed.",
                    icon="📄"
                )
                st.rerun()
            except Exception as e:
                st.session_state.db = None
                st.error(f"⚠️ {e}")


# Document / Chat Display
if st.session_state.db is None:
    st.markdown("""
    <div class="upload-placeholder">
        <span class="upload-placeholder-icon">📄</span>
        <div class="upload-placeholder-title">No document loaded yet</div>
        <div class="upload-placeholder-hint">
            Upload a PDF above — you'll instantly get a summary<br>and can ask questions about it.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Active document badge
    st.markdown(f"""
    <div class="doc-badge">
        📄 {st.session_state.doc_name}
        &nbsp;·&nbsp; {st.session_state.doc_pages} pages
        &nbsp;·&nbsp; {st.session_state.doc_chunks} chunks
    </div>
    """, unsafe_allow_html=True)

    # Document Summary
    st.markdown('<p class="section-label">Document Summary</p>', unsafe_allow_html=True)

    with st.expander("View document summary", expanded=True):
        if st.session_state.summary is None:
            with st.spinner("Generating summary…"):
                try:
                    st.session_state.summary = generate_summary(st.session_state.db)
                except Exception as e:
                    err = str(e)
                    st.error(f"❌ Failed to generate summary: {err}")
                    if st.button("🔄 Retry Summary"):
                        st.rerun()
        if st.session_state.summary:
            st.markdown(render_summary_html(st.session_state.summary), unsafe_allow_html=True)

    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

    # Chat Section
    st.markdown('<p class="section-label">Ask Anything</p>', unsafe_allow_html=True)
    if st.session_state.messages:
        chat_html = ""
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="user-bubble-wrap">
                    <div class="user-bubble">{msg["content"]}</div>
                </div>"""
            else:
                formatted_ai_content = parse_markdown_formatting(msg["content"]).replace("\n", "<br>")
                chat_html += f"""
                <div class="ai-bubble-wrap">
                    <div class="ai-bubble">{formatted_ai_content}</div>
                </div>"""
                if "sources" in msg:
                    chat_html += f'<div class="source-badge">📄 {msg["sources"]}</div>'

        st.markdown(f'<div class="chat-container">{chat_html}</div>', unsafe_allow_html=True)

    # Chat input
    if question := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking…"):
            try:
                answer, retriever = invoke_with_model_fallback(question, st.session_state.db, is_rag=True)
                source_docs = retriever.invoke(question)
                pages = sorted(set(doc.metadata.get("page", 0) + 1 for doc in source_docs))
                source_text = f"Sources: page(s) {', '.join(str(p) for p in pages)}"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_text
                })

            except Exception as e:
                err_msg = str(e)
                st.error(f"❌ An error occurred: {err_msg}")
                st.session_state.messages.pop()

        st.rerun()
