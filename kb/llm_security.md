# LLM application security
Prompt injection is when malicious instructions inside user input or retrieved documents override the system prompt. Mitigations: treat retrieved text as data, restrict tool permissions, validate outputs and keep a human in the loop for sensitive actions.
Retrieval-augmented generation reduces hallucination by grounding answers in retrieved documents, but poisoned documents in the knowledge base can still mislead the model, so sources should be curated and cited.
