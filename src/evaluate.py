import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from rag import load_chunks, Retriever, Agent

TESTS = [
 ("How do I recognise a phishing email?", "phishing.md"),
 ("Do I need to enter my PIN to receive money on UPI?", "upi_fraud.md"),
 ("Which number do I call to report cyber fraud in India?", "upi_fraud.md"),
 ("Why is SMS OTP weaker than an authenticator app?", "password_mfa.md"),
 ("What is credential stuffing?", "password_mfa.md"),
 ("What is the 3-2-1 backup rule?", "malware_ransomware.md"),
 ("What should I do if ransomware infects my PC?", "malware_ransomware.md"),
 ("Is public Wi-Fi safe?", "network_security.md"),
 ("How can I secure my home router?", "network_security.md"),
 ("How do I prevent SQL injection?", "web_vulns.md"),
 ("What is XSS and how is it prevented?", "web_vulns.md"),
 ("A caller says I am under digital arrest, what is this?", "social_engineering.md"),
 ("What is vishing?", "social_engineering.md"),
 ("What is prompt injection in LLM apps?", "llm_security.md"),
 ("How does RAG reduce hallucination?", "llm_security.md"),
 ("Is passkey phishing resistant?", "password_mfa.md"),
 ("How do scammers misuse QR codes for payments?", "upi_fraud.md"),
 ("What are the OWASP Top 10?", "web_vulns.md"),
]
if __name__ == "__main__":
    chunks = load_chunks(); r = Retriever(chunks); agent = Agent(r)
    h1 = h3 = 0; rows = []
    for q, gold in TESTS:
        res = r.search(q, 3); srcs = [x["source"] for x in res]
        h1 += srcs[0] == gold; h3 += gold in srcs
        rows.append({"q": q, "gold": gold, "top1": srcs[0], "hit@1": srcs[0] == gold, "hit@3": gold in srcs})
    # baseline: random chance = 1/8 sources
    summary = {"n_docs": len({c['source'] for c in chunks}), "n_chunks": len(chunks), "n_queries": len(TESTS),
               "hit@1": h1 / len(TESTS), "hit@3": h3 / len(TESTS), "random_hit@1": 1/8}
    demos = ["Is http://paytm-kyc-update.xyz/login safe? I got an SMS asking for my OTP",
             "What is the best way to cook biryani?",
             "I got a call saying I am under digital arrest, what do I do?"]
    demo_out = []
    for d in demos:
        o = agent.run(d); demo_out.append({"q": d, "answer": o["answer"], "trace": o["trace"]})
    json.dump({"summary": summary, "rows": rows, "demos": demo_out}, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "results.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    for r_ in rows:
        if not r_["hit@1"]: print("MISS@1:", r_)
    for d in demo_out: print("\nQ:", d["q"], "\nA:", d["answer"], "\n", d["trace"])
