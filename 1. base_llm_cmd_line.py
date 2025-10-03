import sys, os
from ollama import Client

HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

client = Client(host=HOST)

def chat_once(prompt: str):
    # non-streaming:
    resp = client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]

def chat_loop():
    print(f"Chatting with {MODEL} at {HOST}. Type 'exit' to quit.\n")
    messages = []
    while True:
        user = input("You: ")
        if user.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user})
        # streaming reply
        stream = client.chat(model=MODEL, messages=messages, stream=True)
        print("Llama:", end=" ", flush=True)
        full = []
        for chunk in stream:
            piece = chunk["message"]["content"]
            full.append(piece)
            print(piece, end="", flush=True)
        print("\n")
        messages.append({"role": "assistant", "content": "".join(full)})

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(chat_once(" ".join(sys.argv[1:])))
    else:
        chat_loop()
