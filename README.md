# LLM Chat & RAG Implementation Suite

A collection of Python scripts demonstrating various LLM integration patterns, from basic command-line interfaces to advanced RAG (Retrieval Augmented Generation) systems with Chainlit frontends.

## 📋 Prerequisites

```bash
# Install required packages
pip install langchain langchain-community chainlit chromadb pypdf sentence-transformers
```

## 🗂️ Project Structure

### Basic LLM Integration
- **1. base_llm_cmd_line.py** - Direct LLM chat interface from command line
- **2. base_llm_cmd_line_with_langchain.py** - Command line chat using LangChain framework
- **3. base_llm_chainlit.py** - LangChain chat with Chainlit web interface

### Simple RAG (Retrieval Augmented Generation)
- **4.1 simple_rag_chromadb.py** - Generate and populate simpleRAG collection in ChromaDB
- **4.2 simple_rag_chmd_line.py** - Command line chat with simpleRAG context

### GAR (Generation Augmented Retrieval)
- **5. simple_gar.py** - Basic GAR implementation on text documents

### Peak RAG (Advanced PDF Processing)
- **6.1 peak_rag_chromadb.py** - Vectorize and store PDF embeddings in peakRAG collection
- **6.2 peak_rag_chainlit.py** - Chat with large PDFs via Chainlit interface

## 🚀 Usage

### Basic LLM Chat

**Script 1: Direct Command Line Chat**
```bash
python "1. base_llm_cmd_line.py"
```

**Script 2: LangChain Command Line Chat**
```bash
python "2. base_llm_cmd_line_with_langchain.py"
```

**Script 3: Chainlit Web Interface**
```bash
python -m chainlit run "3. base_llm_chainlit.py" -w --port 5000
```
Access at: `http://localhost:5000`

### Simple RAG Implementation

**Step 1: Create ChromaDB Collection**
```bash
python "4.1 simple_rag_chromadb.py"
```

**Step 2: Chat with RAG Context**
```bash
python "4.2 simple_rag_chmd_line.py"
```

### GAR Implementation

**Run GAR on Text Documents**
```bash
python "5. simple_gar.py"
```

### Peak RAG (PDF Processing)

**Step 1: Vectorize PDFs into ChromaDB**
```bash
python "6.1 peak_rag_chromadb.py"
```

**Step 2: Chat with PDF Context**
```bash
python -m chainlit run "6.2 peak_rag_chainlit.py" -w --port 5000
```
Access at: `http://localhost:5000`

## 📝 Notes

- **ChromaDB** is used as the vector database for RAG implementations
- **Chainlit** provides an interactive web interface for chatting with LLMs
- **Simple RAG** works with basic text collections
- **Peak RAG** is optimized for large PDF document processing
- Port 5000 is used by default for Chainlit apps (can be changed via `--port` flag)

## 🔄 Workflow

### For Simple RAG:
1. Run `4.1` to create your vector database
2. Run `4.2` to chat with the stored context

### For Peak RAG:
1. Run `6.1` to process and store your PDFs
2. Run `6.2` to launch the chat interface

## 💡 Tips

- The `-w` flag in Chainlit commands enables auto-reload on file changes
- Make sure ChromaDB collections are created before running chat scripts
- For best results, ensure your PDFs are text-based (not scanned images)

## 🐛 Troubleshooting

If you encounter module import errors:
```bash
pip install --upgrade langchain langchain-community langchain-openai
```

If ChromaDB fails to initialize:
```bash
pip install --upgrade chromadb
```
