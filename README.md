# DocMind

A document analysis application built with Streamlit, LangChain, and Google Gemini.

## Features
- Upload any PDF document to generate structured summaries
- Interactive RAG (Retrieval-Augmented Generation) chat to ask questions about document content
- Dark theme UI with custom animated canvas header

## Quick Start

1. **Clone the repository:**
   `ash
   git clone https://github.com/YOUR_USERNAME/DocMind.git
   cd DocMind
   `

2. **Set up a virtual environment:**
   `ash
   python -m venv venv
   # On Windows:
   venv\\Scripts\\activate
   # On macOS/Linux:
   source venv/bin/activate
   `

3. **Install dependencies:**
   `ash
   pip install -r requirements.txt
   `

4. **Configure environment variables:**
   Create a .env file in the root directory and add your Google API key:
   `env
   GOOGLE_API_KEY=your_google_api_key_here
   `

5. **Run the application:**
   `ash
   streamlit run app.py
   `

## Tech Stack
- **Frontend / App Framework:** Streamlit
- **RAG Pipeline:** LangChain
- **Vector Store:** ChromaDB
- **LLM & Embeddings:** Google Gemini (gemini-3.6-flash, gemini-embedding-001)
- **PDF Extraction:** pdfplumber & pypdf
