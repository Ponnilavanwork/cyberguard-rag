import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from rag import load_chunks, Retriever, Agent
agent = Agent(Retriever(load_chunks()))
print("CyberGuard-RAG (type 'quit' to exit)")
while True:
    q = input("\n> ").strip()
    if q.lower() in {"quit", "exit"}: break
    o = agent.run(q); print(o["answer"]); print("trace:", o["trace"])
