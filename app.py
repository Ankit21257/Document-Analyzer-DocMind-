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

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@300..700&display=swap');

html, body {
    font-family: 'Inter', sans-serif;
    background-color: #000000 !important;
    background: #000000 !important;
    color-scheme: dark !important;
}

[class*="css"], [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"], section.main, .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #000000 !important;
    background: #000000 !important;
}

.stApp {
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }

/* Hero / Header */
.hero-header { text-align: center; padding: 2.5rem 1rem 1rem 1rem; }
.hero-title {
    font-size: 2.5rem; font-weight: 700; color: #ffffff;
    margin: 0; letter-spacing: -0.025em; line-height: 1.2;
}
.hero-subtitle { color: #737373; font-size: 0.95rem; margin-top: 0.5rem; }

/* Divider */
.gradient-divider {
    height: 1px;
    background: #262626;
    margin: 1.5rem 0; border: none;
}

/* No-doc placeholder */
.upload-placeholder {
    border: 1px dashed #404040;
    border-radius: 16px; padding: 3rem 1.5rem;
    text-align: center; background: rgba(23, 23, 23, 0.75);
    backdrop-filter: blur(12px);
    margin-top: 0.5rem;
}
.upload-placeholder-icon { font-size: 2.5rem; display: block; margin-bottom: 0.8rem; color: #a3a3a3; }
.upload-placeholder-title { color: #ffffff; font-size: 1rem; font-weight: 600; margin-bottom: 0.4rem; }
.upload-placeholder-hint  { color: #737373; font-size: 0.85rem; }

/* Active doc badge */
.doc-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(23, 23, 23, 0.75); border: 1px solid #262626;
    backdrop-filter: blur(12px);
    border-radius: 9999px; padding: 0.35rem 1rem;
    color: #d4d4d4; font-size: 0.82rem; font-weight: 500;
    margin-bottom: 1rem;
}

/* Summary text */
.summary-card {
    padding: 0.5rem 0.2rem;
    font-family: 'Quicksand', sans-serif !important;
    font-optical-sizing: auto;
}
.summary-text, .summary-text * {
    color: #d4d4d4; font-size: 0.92rem; line-height: 1.8;
    font-family: 'Quicksand', sans-serif !important;
    font-optical-sizing: auto;
}

/* Section label */
.section-label {
    color: #a3a3a3; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.6rem;
}

/* Chat container */
.chat-container {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin-bottom: 1rem !important;
    min-height: 0 !important;
    max-height: none !important;
    overflow: visible !important;
    font-family: 'Quicksand', sans-serif !important;
    font-optical-sizing: auto;
}

/* Message Styling (Rounded box for User Question, No box for AI Answer) */
.user-bubble-wrap { display: flex; justify-content: flex-end; margin: 1.8rem 0 1.2rem 0; }
.user-bubble {
    background: rgba(39, 39, 42, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 14px 14px 4px 14px !important;
    padding: 0.5rem 0.95rem !important;
    color: #ffffff !important;
    font-size: 0.9rem; line-height: 1.5;
    font-weight: 500;
    max-width: 75%;
    width: fit-content !important;
    text-align: left;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    font-family: 'Quicksand', sans-serif !important;
    font-optical-sizing: auto;
}
.ai-bubble-wrap {
    display: flex; flex-direction: column; align-items: flex-start;
    margin: 0.4rem 0 2.2rem 0;
}
.ai-bubble {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    color: #e4e4e7 !important;
    font-size: 0.95rem; line-height: 1.6;
    max-width: 100%;
    font-family: 'Quicksand', sans-serif !important;
    font-optical-sizing: auto;
}
.source-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 9999px;
    background: rgba(23, 23, 23, 0.75); border: 1px solid #262626;
    backdrop-filter: blur(8px);
    color: #a3a3a3; font-family: 'Quicksand', sans-serif !important; font-size: 0.75rem;
    margin-top: 6px; margin-left: 0rem;
}

/* Viewport & Background Base */
html, body {
    background-color: #030509 !important;
    background: #030509 !important;
    color-scheme: dark !important;
}

#root, .stApp, 
[data-testid="stAppViewContainer"], 
[data-testid="stAppViewBlockContainer"],
[data-testid="stHeader"], 
[data-testid="stToolbar"],
[data-testid="stMain"], 
[data-testid="stMainBlockContainer"],
section.main {
    background-color: #030509 !important;
    background: #030509 !important;
}

/* Fixed Bottom Container for Question Textbox (Transparent Background) */
[data-testid="stBottom"],
[data-testid="stBottom"] *,
[data-testid="stBottomBlockContainer"],
[data-testid="stBottomBlockContainer"] * {
    background-color: transparent !important;
    background: transparent !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border-top: none !important;
    box-shadow: none !important;
}

[data-testid="stBottom"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 9999 !important;
    padding: 10px 0 16px 0 !important;
}

[data-testid="stBottomBlockContainer"] {
    max-width: 736px !important;
    margin: 0 auto !important;
    padding: 0 1rem !important;
}

.stApp, [data-testid="stMainBlockContainer"], section.main {
    padding-bottom: 120px !important;
}

/* Keep Only Textbox Styled */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea {
    background-color: #000000 !important;
    background: #000000 !important;
}

[data-testid="stChatInput"] {
    border: 1px solid #27272a !important;
    border-radius: 16px !important;
    padding: 8px 14px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.8) !important;
    outline: none !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

/* Click / Focus state: sleek dark gray border, NO red box or outline */
[data-testid="stChatInput"]:focus,
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"]:active,
[data-testid="stChatInput"]:focus-visible {
    border-color: #3f3f46 !important;
    outline: none !important;
    box-shadow: 0 0 0 1px #3f3f46 !important;
}

/* Textarea & Inner Elements */
[data-testid="stChatInput"] div,
[data-testid="stChatInput"] textarea {
    background-color: #000000 !important;
    background: #000000 !important;
    color: #f4f4f5 !important;
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}

[data-testid="stChatInput"] textarea {
    font-size: 0.93rem !important;
    line-height: 1.5 !important;
    color: #ffffff !important;
    font-family: 'Quicksand', sans-serif !important;
    font-optical-sizing: auto;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #71717a !important;
    font-size: 0.93rem !important;
    font-family: 'Quicksand', sans-serif !important;
    font-optical-sizing: auto;
}

/* Remove any red outline/ring from inner elements */
[data-testid="stChatInput"] *:focus,
[data-testid="stChatInput"] *:focus-within,
[data-testid="stChatInput"] *:active,
[data-testid="stChatInput"] *:focus-visible {
    outline: none !important;
    box-shadow: none !important;
    border-color: transparent !important;
}

/* Dark Circular Send Button */
[data-testid="stChatInputSubmitButton"] button {
    background-color: #27272a !important;
    color: #a1a1aa !important;
    border-radius: 50% !important;
    width: 32px !important;
    height: 32px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: none !important;
    transition: background-color 0.2s ease, transform 0.2s ease !important;
}

[data-testid="stChatInputSubmitButton"] button:hover {
    background-color: #3f3f46 !important;
    color: #ffffff !important;
    transform: scale(1.05) !important;
}

[data-testid="stChatInputSubmitButton"] svg {
    fill: #a1a1aa !important;
    color: #a1a1aa !important;
    width: 16px !important;
    height: 16px !important;
}

[data-testid="stChatInputSubmitButton"] button:hover svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

.stSpinner > div { border-top-color: #38bdf8 !important; }
[data-testid="stExpander"] {
    border: 1px solid #27272a !important;
    border-radius: 16px !important;
    background: #18181b !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #18181b !important;
    border: 1px dashed #3f3f46 !important;
    border-radius: 16px !important;
}

/* Full-Width Hero Breakout */
[data-testid="stCustomComponentV1"],
iframe[title*="components.html"] {
    width: 100vw !important;
    position: relative !important;
    left: 50% !important;
    right: 50% !important;
    margin-left: -50vw !important;
    margin-right: -50vw !important;
    border: none !important;
}

/* Minimalist White Loading Symbol on Pure Black */
#loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: #000000;
    z-index: 9999999;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: none;
    animation: fadeOutLoader 0.4s ease-out 0.7s forwards;
}

@keyframes fadeOutLoader {
    0% { opacity: 1; visibility: visible; }
    90% { opacity: 0.1; visibility: visible; }
    100% { opacity: 0; visibility: hidden; display: none; }
}

.white-spinner {
    width: 32px;
    height: 32px;
    border: 2.5px solid rgba(255, 255, 255, 0.12);
    border-top-color: #ffffff;
    border-radius: 50%;
    animation: spinWhite 0.7s linear infinite;
}

@keyframes spinWhite {
    to { transform: rotate(360deg); }
}
</style>

<!-- Loading Screen Overlay (White Symbol on Pure Black) -->
<div id="loading-overlay">
    <div class="white-spinner"></div>
</div>
""", unsafe_allow_html=True)

# Hero header component
components.html("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body {
    background: #000000;
    width: 100vw;
    height: 100%;
    margin: 0;
    padding: 0;
    overflow: hidden;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
  }
  .hero-container {
    background: #000000;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    width: 100vw;
    text-align: center;
    position: relative;
    padding-top: 8px;
  }
  .hero-title {
    font-size: 2.85rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.03em;
    position: relative;
    z-index: 20;
    margin-bottom: 0px;
  }
  .sparkles-box {
    background: #000000;
    width: 100vw;
    height: 160px;
    position: relative;
    overflow: hidden;
    margin-top: 2px;
  }
  /* Gradients across 100% full screen width */
  .gradient-indigo-blur {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(to right, transparent, #6366f1 10%, #6366f1 90%, transparent);
    filter: blur(4px);
    z-index: 10;
  }
  .gradient-indigo-sharp {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 1px;
    background: linear-gradient(to right, transparent, #6366f1 10%, #6366f1 90%, transparent);
    z-index: 10;
  }
  .gradient-sky-blur {
    position: absolute;
    top: 0;
    left: 5%;
    width: 90%;
    height: 5px;
    background: linear-gradient(to right, transparent, #0ea5e9 15%, #0ea5e9 85%, transparent);
    filter: blur(5px);
    z-index: 10;
  }
  .gradient-sky-sharp {
    position: absolute;
    top: 0;
    left: 5%;
    width: 90%;
    height: 1px;
    background: linear-gradient(to right, transparent, #0ea5e9 15%, #0ea5e9 85%, transparent);
    z-index: 10;
  }
  canvas {
    display: block;
    width: 100%;
    height: 100%;
    position: absolute;
    top: 0;
    left: 0;
    z-index: 5;
  }
  .hero-subtitle {
    position: absolute;
    bottom: 14px;
    left: 0;
    right: 0;
    text-align: center;
    color: #737373;
    font-size: 0.95rem;
    z-index: 25;
    font-weight: 400;
  }
  .special-text {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    color: #a1a1aa;
    font-size: 0.88rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    display: inline-block;
    min-height: 1.25rem;
    white-space: pre-wrap;
    text-shadow: 0 0 12px rgba(0, 0, 0, 0.9);
  }
</style>
</head>
<body>
<div class="hero-container">
  <h1 class="hero-title">DocMind</h1>
  <div class="sparkles-box">
    <!-- Gradients across full screen width -->
    <div class="gradient-indigo-blur"></div>
    <div class="gradient-indigo-sharp"></div>
    <div class="gradient-sky-blur"></div>
    <div class="gradient-sky-sharp"></div>
    
    <!-- Sparkles Canvas -->
    <canvas id="sparkles-canvas"></canvas>
    
    <!-- Subtitle placed directly over the sparkles field -->
    <p class="hero-subtitle"><span id="special-text-subtitle" class="special-text"></span></p>
  </div>
</div>

<script>
  const canvas = document.getElementById('sparkles-canvas');
  const ctx = canvas.getContext('2d');
  const box = document.querySelector('.sparkles-box');
  let W, H, particles = [];
  const DPR = window.devicePixelRatio || 1;

  function resize() {
    W = window.innerWidth || document.documentElement.clientWidth;
    H = box.clientHeight;
    canvas.width = W * DPR;
    canvas.height = H * DPR;
    ctx.scale(DPR, DPR);
  }

  function rand(min, max) { return min + Math.random() * (max - min); }

  function createParticle() {
    return {
      x: rand(0, W),
      y: rand(0, H),
      r: rand(0.4, 1.4),
      alpha: rand(0.1, 1.0),
      speed: rand(0.006, 0.028),
      dir: Math.random() < 0.5 ? 1 : -1,
      dx: rand(-0.3, 0.3),
      dy: rand(-0.3, 0.3)
    };
  }

  function init() {
    // High density filling 100% screen width
    const count = Math.min(950, Math.floor((W * H) / 70));
    particles = Array.from({ length: count }, createParticle);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x += p.dx;
      p.y += p.dy;
      p.alpha += p.speed * p.dir;
      if (p.alpha <= 0.05) { p.alpha = 0.05; p.dir = 1; }
      if (p.alpha >= 1) { p.alpha = 1; p.dir = -1; }
      if (p.x < -4 || p.x > W + 4 || p.y < -4 || p.y > H + 4) {
        particles[i] = createParticle();
      }
      ctx.save();
      ctx.globalAlpha = p.alpha;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = '#FFFFFF';
      ctx.fill();
      ctx.restore();
    }
    requestAnimationFrame(draw);
  }

  // Subtitle scramble animation
  const RANDOM_CHARS = "_!X$0-+*#";

  function getRandomChar(prevChar) {
    let char;
    do {
      char = RANDOM_CHARS[Math.floor(Math.random() * RANDOM_CHARS.length)];
    } while (char === prevChar);
    return char;
  }

  function startSpecialTextAnimation() {
    const el = document.getElementById('special-text-subtitle');
    if (!el) return;
    const text = "Upload any document. Get intelligent insights & instant answers.";
    const speed = 20;
    let animationStep = 0;
    let currentPhase = "phase1";
    let interval = null;

    function runPhase1() {
      const maxSteps = text.length * 2;
      const currentLength = Math.min(animationStep + 1, text.length);
      const chars = [];
      for (let i = 0; i < currentLength; i++) {
        const prevChar = i > 0 ? chars[i - 1] : undefined;
        chars.push(getRandomChar(prevChar));
      }
      for (let i = currentLength; i < text.length; i++) {
        chars.push('\u00A0');
      }
      el.textContent = chars.join('');

      if (animationStep < maxSteps - 1) {
        animationStep++;
      } else {
        currentPhase = "phase2";
        animationStep = 0;
      }
    }

    function runPhase2() {
      const revealedCount = Math.floor(animationStep / 2);
      const chars = [];
      for (let i = 0; i < revealedCount && i < text.length; i++) {
        chars.push(text[i]);
      }
      if (revealedCount < text.length) {
        if (animationStep % 2 === 0) {
          chars.push('_');
        } else {
          chars.push(getRandomChar());
        }
      }
      for (let i = chars.length; i < text.length; i++) {
        chars.push(getRandomChar());
      }
      el.textContent = chars.join('');

      if (animationStep < text.length * 2 - 1) {
        animationStep++;
      } else {
        el.textContent = text;
        if (interval) {
          clearInterval(interval);
          interval = null;
        }
      }
    }

    interval = setInterval(() => {
      if (currentPhase === "phase1") {
        runPhase1();
      } else {
        runPhase2();
      }
    }, speed);
  }

  window.addEventListener('resize', () => { resize(); init(); });
  resize();
  init();
  draw();
  startSpecialTextAnimation();
</script>
</body>
</html>
""", height=230)

st.markdown('<hr class="gradient-divider" style="margin-top: 0.2rem;">', unsafe_allow_html=True)


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
