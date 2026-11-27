"""Run from repository root: python -m scripts.check_models"""
"""
00 Model Clients: Check hosted Qwen, embeddings, and optional Ollama readiness
Module: 00 Model Clients
Author: Debanjali Ghosh

"""

from src.model_clients import qwen_chat, embed_texts, local_llama_chat
for name, fn in [
    ("Qwen hosted", lambda: qwen_chat([{"role":"user","content":"Reply exactly: ready"}])),
    ("BGE embedding", lambda: f"vectors={len(embed_texts(['test']))}"),
    ("Ollama optional", lambda: local_llama_chat([{"role":"user","content":"Reply exactly: ready"}]))]:
    try: print(name, "OK", str(fn())[:100])
    except Exception as exc: print(name, "NOT READY", type(exc).__name__, str(exc)[:160])
