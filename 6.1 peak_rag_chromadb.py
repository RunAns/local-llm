# build_index.py
# Create a Chroma DB from a TXT file using LangChain + Ollama embeddings.
# Usage:
#   python build_index.py docs.txt

import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from PyPDF2 import PdfReader

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


DOC_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("c. media-kit.pdf")
PERSIST_DIR = "./chroma_db"

def main():
    if not DOC_PATH.exists():
        raise FileNotFoundError(f"File not found: {DOC_PATH}")

    if DOC_PATH.suffix.lower() == ".pdf":
        text = load_pdf(DOC_PATH)
        print(text)
    else:
        text = DOC_PATH.read_text(encoding="utf-8", errors="ignore")


    # Simple chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(text)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create / overwrite a local Chroma DB
    vs = Chroma(
        collection_name="peakRAG",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # Clear old data if collection already exists
    try:
        vs.delete_collection()
    except Exception:
        pass

    vs = Chroma(
        collection_name="peakRAG",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    vs.add_texts(chunks)

    print(f"✅ Indexed {len(chunks)} chunks from {DOC_PATH} into {PERSIST_DIR}")

if __name__ == "__main__":
    main()
