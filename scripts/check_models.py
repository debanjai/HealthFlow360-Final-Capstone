"""Run from repository root: python -m scripts.check_models"""
from src.model_clients import qwen_chat, embed_texts, local_llama_chat
for name, fn in [
    ("Qwen hosted", lambda: qwen_chat([{"role":"user","content":"Reply exactly: ready"}])),
    ("BGE embedding", lambda: f"vectors={len(embed_texts(['test']))}"),
    ("Ollama optional", lambda: local_llama_chat([{"role":"user","content":"Reply exactly: ready"}]))]:
    try: print(name, "OK", str(fn())[:100])
    except Exception as exc: print(name, "NOT READY", type(exc).__name__, str(exc)[:160])
