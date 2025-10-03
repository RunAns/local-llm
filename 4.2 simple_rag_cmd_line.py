# rag_qa.py
# Retrieve from Chroma; answer ONLY from context. If not found,
# print: "Can't find in docs but answer is - <default LLM answer>"
#
# Usage (interactive):
#   python rag_qa.py
#
# Requires:
#   pip install langchain-chroma langchain-ollama langchain-text-splitters chromadb ollama

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ollama import Client

PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"  # `ollama pull nomic-embed-text`
LLM_MODEL = "llama3.1"            # `ollama pull llama3.1`

# Simple guardrail prompt: answer ONLY from context or say "Not found in docs."
RAG_PROMPT = """You must answer using only the context below.
If the answer is not in the context, reply exactly: Not found in docs.

Context:
{context}

Question: {question}
Answer:"""

def get_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        collection_name="simpleRAG",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

def format_context(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

def main():
    client = Client(host="http://localhost:11434")  # Ollama server
    vs = get_vectorstore()

    print(f"RAG ready. Using DB at {PERSIST_DIR}. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # 1) Retrieve top-k with scores so we can check if they are relevant
        results = vs.similarity_search_with_score(q, k=3)

        # Keep only reasonably close matches (smaller score = closer for cosine distance)
        THRESHOLD = 0.35
        filtered = [doc for (doc, score) in results if score is not None and score < THRESHOLD]

        if not filtered:
            # No relevant chunks → fall back: show default model answer
            default = client.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": q}],
            )["message"]["content"]
            print(f"AI: Can't find in docs but answer is - {default}\n")
            continue

        # 2) Build RAG prompt with context
        ctx = format_context([d for (d, _) in results][:3])  # you can also use filtered
        prompt = RAG_PROMPT.format(context=ctx, question=q)

        # 3) Ask the model with the RAG prompt
        rag_answer = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )["message"]["content"].strip()

        if rag_answer == "Not found in docs.":
            # Model decided it isn't in the provided context
            default = client.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": q}],
            )["message"]["content"]
            print(f"AI: Can't find in docs but answer is - {default}\n")
        else:
            print(f"AI: {rag_answer}\n")

if __name__ == "__main__":
    main()
