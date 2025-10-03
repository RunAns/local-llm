# rag_chainlit.py
# Chainlit app for simple RAG with Chroma + Ollama

import chainlit as cl
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ollama import Client

PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"   # ollama pull nomic-embed-text
LLM_MODEL = "llama3.1"             # ollama pull llama3.1

RAG_PROMPT = """You must answer using only the context below.
If the answer is not in the context, reply exactly: Not found in docs.

Context:
{context}

Question: {question}
Answer:"""

# --- helpers ---
def get_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        collection_name="peakRAG",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

def format_context(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

# --- chainlit hooks ---
@cl.on_chat_start
async def start():
    cl.user_session.set("client", Client(host="http://localhost:11434"))
    cl.user_session.set("vs", get_vectorstore())
    await cl.Message(content=f"✅ RAG ready. Using DB at {PERSIST_DIR}").send()

@cl.on_message
async def rag_reply(message: cl.Message):
    client: Client = cl.user_session.get("client")
    vs: Chroma = cl.user_session.get("vs")

    q = message.content.strip()
    if not q:
        await cl.Message(content="❌ Please enter a question").send()
        return

    # 1) Retrieve
    # results = vs.similarity_search_with_score(q, k=3)
    # THRESHOLD = 0.35
    # filtered = [doc for (doc, score) in results if score is not None and score < THRESHOLD]
    results = vs.similarity_search_with_score(q, k=3)
    # Always keep top 3, even if scores are higher
    filtered = [doc for (doc, _) in results]


    if not filtered:
        default = client.chat(
            model=LLM_MODEL, messages=[{"role": "user", "content": q}]
        )["message"]["content"]
        await cl.Message(content=f"AI: Can't find in docs but answer is - {default}").send()
        return

    # 2) Build context
    ctx = format_context([d for (d, _) in results][:3])
    prompt = RAG_PROMPT.format(context=ctx, question=q)

    # 3) Ask LLM with context
    rag_answer = client.chat(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}]
    )["message"]["content"].strip()

    if rag_answer == "Not found in docs.":
        default = client.chat(
            model=LLM_MODEL, messages=[{"role": "user", "content": q}]
        )["message"]["content"]
        await cl.Message(content=f"AI: Can't find in docs but answer is - {default}").send()
    else:
        await cl.Message(content=f"AI: {rag_answer}").send()
