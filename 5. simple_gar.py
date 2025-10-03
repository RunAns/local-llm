# gar_simple.py
# Super-simple "Generative AI Augmented Retrieval" demo
# Just loads a text file and lets the LLM answer using that context.

from ollama import Client

DOC_PATH = "a. simple rag input doc.txt"
MODEL = "llama3.1"

RAG_PROMPT = """You must answer using only the context below.
If the answer is not in the context, reply exactly: Not found in docs.

Context:
{context}

Question: {question}
Answer:"""

def main():
    # Load the document
    with open(DOC_PATH, "r", encoding="utf-8", errors="ignore") as f:
        context = f.read()

    client = Client(host="http://localhost:11434")

    print(f"GAR ready on {DOC_PATH}. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Ask model to answer using only the doc
        prompt = RAG_PROMPT.format(context=context, question=q)
        answer = client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])["message"]["content"]

        if answer.strip() == "Not found in docs.":
            # Fall back to default answer (no context restriction)
            default = client.chat(model=MODEL, messages=[{"role": "user", "content": q}])["message"]["content"]
            print(f"AI: Can't find in docs but answer is - {default}\n")
        else:
            print(f"AI: {answer}\n")

if __name__ == "__main__":
    main()
