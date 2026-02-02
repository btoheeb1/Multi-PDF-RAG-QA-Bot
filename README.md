## Features

- ✅ Upload **multiple PDFs** at once (Streamlit UI)
- ✅ Automatic **text extraction → chunking → embedding**
- ✅ Stores embeddings in a **Chroma** vector database (persistent)
- ✅ **Retrieves across all uploaded PDFs** for each query
- ✅ Displays **source file names** used to generate the answer
- ✅ Uses **Groq + Llama-3.3-70B** for fast, high-quality responses

---

## Tech Stack

- **Python**
- **Streamlit** (UI)
- **LangChain** (RAG pipeline)
- **ChromaDB** (vector store)
- **HuggingFace Embeddings** (text embeddings)
- **Groq** (LLM inference, Llama-3.3-70B)

---

## Setup

1) Clone Repo
```bash
git clone https://github.com/btoheeb1/Multi-PDF-RAG-QA-Bot.git
cd Multi-PDF-RAG-QA-Bot

2) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows

3) Install dependencies
pip install -r requirements.txt

4) Add your Groq API key

Create a .env file in the project root:

GROQ_API_KEY=your_key_here

Run the App
streamlit run app.py
