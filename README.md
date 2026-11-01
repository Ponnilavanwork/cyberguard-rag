# CyberGuard-RAG
Retrieval-Augmented cybersecurity advisor with a small tool-using agent (URL risk checker).
Capstone – *From Tokens to Transformers: GenAI x LLM Mastery* (IIT Guwahati, Consulting & Analytics Club).

## Run
```
pip install -r requirements.txt
python src/evaluate.py     # retrieval evaluation + demo queries -> outputs/results.json
python src/app.py          # interactive CLI
```
## Structure
- `kb/` curated knowledge base (8 documents)
- `src/rag.py` chunking, TF-IDF retriever, URL-risk tool, agent, grounded answer composer
- `src/evaluate.py` 18-question gold-labelled evaluation (hit@1 / hit@3)
- To use an LLM: `Agent(retriever, llm=lambda prompt: ...)` – the prompt already enforces context-only answers.
