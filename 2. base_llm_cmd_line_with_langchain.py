# lc_cli_chat_fixed.py
# Recommended pins (PowerShell):
# pip install -U "pydantic>=2.7,<3" "langchain>=0.2,<0.4" "langchain-core>=0.2,<0.4" "langchain-ollama>=0.1.3"

import argparse

# --- bring forward-ref types into scope BEFORE model_rebuild ---
from langchain_core.caches import BaseCache  # noqa: F401
from langchain_core.callbacks import Callbacks  # noqa: F401

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Rebuild with an explicit types namespace so Pydantic can resolve refs
ChatOllama.model_rebuild(
    _types_namespace={
        "BaseCache": BaseCache,
        "Callbacks": Callbacks,
    }
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="You are a concise, friendly assistant.")
    # ap.add_argument("--system", default="You are a translator. Simply translate the input text to old English")

    ap.add_argument("--model", default="llama3.1", help="Ollama model, e.g. llama3.1:8b")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--once", help="One-shot message (skip interactive loop)")
    args = ap.parse_args()

    llm = ChatOllama(model=args.model, temperature=args.temperature)

    prompt = ChatPromptTemplate.from_messages([
        ("system", args.system),
        ("human", "{input}")
    ])
    chain = prompt | llm

    if args.once:
        resp = chain.invoke({"input": args.once})
        print(resp.content)
        return

    print(f"Chatting with {args.model}\nSystem: {args.system}\nType 'exit' to quit.\n")
    while True:
        user = input("You: ")
        if user.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        resp = chain.invoke({"input": user})
        print(f"Llama: {resp.content}\n")

if __name__ == "__main__":
    main()
