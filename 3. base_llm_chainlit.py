import chainlit as cl

from langchain_core.caches import BaseCache  # noqa: F401
from langchain_core.callbacks import Callbacks  # noqa: F401
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

ChatOllama.model_rebuild(_types_namespace={"BaseCache": BaseCache, "Callbacks": Callbacks})

MODEL_NAME = "llama3.1"
SYSTEM_PROMPT = "You are a clear, friendly assistant. Keep answers concise."

@cl.on_chat_start
async def start():
    await cl.Message(
        content=f"✅ Ready! Model: **{MODEL_NAME}**\n"
                f"System: {SYSTEM_PROMPT}"
    ).send()
    cl.user_session.set("history", [SystemMessage(content=SYSTEM_PROMPT)])

@cl.on_message
async def handle(message: cl.Message):
    history = cl.user_session.get("history", [SystemMessage(content=SYSTEM_PROMPT)])
    llm = ChatOllama(model=MODEL_NAME, temperature=0.7)

    # add user message
    history.append(HumanMessage(content=message.content))

    # stream reply
    out = cl.Message(author="AI", content="")
    try:
        async for chunk in llm.astream(history):
            if hasattr(chunk, "content") and chunk.content:
                await out.stream_token(chunk.content)
        await out.send()
        # store assistant reply
        history.append(AIMessage(content=out.content))
        cl.user_session.set("history", history)
    except Exception as e:
        await cl.Message(content=f"❌ Error: {e}").send()

if __name__ == "__main__":
    cl.run()
