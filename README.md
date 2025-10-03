---

````markdown
# LLM + RAG + GAR Demos

This repo contains simple scripts to demonstrate:
- Chatting with local LLMs via Ollama
- Adding LangChain wrappers
- Using Chainlit as a frontend
- Basic RAG (Retrieval Augmented Generation) with Chroma
- Basic GAR (Generative AI Augmented Retrieval) on documents
- Handling larger PDFs via RAG with Chainlit

---

## 🔧 Setup

Make sure you have:

1. **Python 3.10+**  
2. **Ollama installed and running**  
   ```bash
   ollama serve
````

3. Pull required models:

   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```

### 📦 Install Python dependencies

Install these into your virtual environment:

```bash
pip install ollama chainlit langchain langchain-core langchain-ollama langchain-chroma langchain-text-splitters chromadb pypdf2 pdfplumber
```

---

## 📂 Scripts

### 1. `1. base_llm_cmd_line.py`

Run a simple command-line chat with an LLM of your choice.

```bash
python 1.\ base_llm_cmd_line.py
```

---

### 2. `2. base_llm_cmd_line_with_langchain.py`

Same as (1) but uses **LangChain** to interact with the LLM.

```bash
python 2.\ base_llm_cmd_line_with_langchain.py
```

---

### 3. `3. base_llm_chainlit.py`

Wraps LangChain chat into a **Chainlit frontend**.

Run with:

```bash
python -m chainlit run ".\3. base_llm_chainlit.py" -w --port 5000
```

Then open [http://localhost:5000](http://localhost:5000).

---

### 4.1 `4.1 simple_rag_chromadb.py`

Builds a **simpleRAG collection** in ChromaDB from a text document.

```bash
python 4.1\ simple_rag_chromadb.py docs.txt
```

---

### 4.2 `4.2 simple_rag_chmd_line.py`

Chats with an LLM using **only the `simpleRAG` collection** as context.

```bash
python 4.2\ simple_rag_chmd_line.py
```

---

### 5. `5. simple_gar.py`

A **basic GAR (Generative AI Augmented Retrieval)** implementation over a text document.
Loads the document, augments queries with the LLM, and answers.

```bash
python 5.\ simple_gar.py
```

---

### 6.1 `6.1 peak_rag_chromadb.py`

Vectorizes and stores embeddings from **PDF files** into a ChromaDB collection named **`peakRAG`**.

```bash
python 6.1\ peak_rag_chromadb.py "yourfile.pdf"
```

---

### 6.2 `6.2 peak_rag_chainlit.py`

Chainlit app for chatting with an LLM using **PDF context** via the `peakRAG` collection.

```bash
python -m chainlit run ".\6.2 peak_rag_chainlit.py" -w --port 5000
```

---

## 📝 Notes

* If Chainlit is not recognized, use:

  ```bash
  python -m chainlit run ...
  ```
* For PDF indexing, prefer **pdfplumber** (more reliable than PyPDF2).
* If you see "Can't find in docs", the script will fall back to a default open-domain LLM answer.

---

```

---
