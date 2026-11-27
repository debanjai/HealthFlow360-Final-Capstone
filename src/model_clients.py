"""HealthFlow 360 clients: hosted Qwen text, Ollama fallback, embeddings."""
"""
00 Model Clients: Call Hosted Qwen text generation, Ollama fallback, and embeddings Models
Module: 00 Model Clients
Author: Debanjali Ghosh
"""
from __future__ import annotations

import os
from pathlib import Path
import time

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / '.env', override=True)


from .observability import traced, span, update

def _qwen_chat(messages: list[dict], temperature: float = 0.0,
              max_tokens: int | None = None, observation=None) -> str:
    """Text generation via the Track B hosted Qwen3-8B chat-completions API.

    Keeps the existing HealthFlow 360 function signature and string return type.
    """
    if not messages or not any(m.get('role') == 'user' for m in messages):
        raise ValueError('At least one user message is required')

    base_url = os.getenv('INFERENCE_API_BASE_URL', '').rstrip('/')
    api_key = os.getenv('INFERENCE_API_KEY', '')
    model_id = os.getenv('INFERENCE_MODEL', 'Qwen/Qwen3-8B')
    cert_setting = os.getenv('INFERENCE_CA_CERT', '')
    if not base_url or not api_key or not cert_setting:
        raise RuntimeError('Set INFERENCE_API_BASE_URL, INFERENCE_API_KEY and INFERENCE_CA_CERT in .env')
    if not base_url.startswith('https://'):
        raise ValueError('Hosted inference requires an HTTPS base URL')

    cert_path = Path(cert_setting).expanduser()
    if not cert_path.is_absolute():
        cert_path = PROJECT_ROOT / cert_path
    if not cert_path.is_file():
        raise FileNotFoundError(f'Inference CA certificate not found: {cert_path}')

    limit = 512 if max_tokens is None else int(max_tokens)
    if not 1 <= limit <= 1024:
        raise ValueError('Hosted Qwen max_tokens must be between 1 and 1024')

    payload = {
        'model': model_id,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': limit,
    }
    url = f'{base_url}/chat/completions'
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

    for attempt in range(4):
        try:
            response = requests.post(
                url, headers=headers, json=payload,
                verify=str(cert_path), timeout=120,
            )
        except requests.RequestException as exc:
            # Never include request headers or patient text in exception messages.
            raise RuntimeError(f'Hosted inference connection failed ({type(exc).__name__})') from None

        if response.status_code in (429, 503) and attempt < 3:
            time.sleep((2, 4, 8)[attempt])
            continue
        if not response.ok:
            raise RuntimeError(f'Hosted inference returned HTTP {response.status_code}')
        try:
            body = response.json()
            content = body['choices'][0]['message']['content']
            usage = body.get('usage') or {}
            counts = {dst: usage[src] for src, dst in [('prompt_tokens','input'), ('completion_tokens','output'), ('total_tokens','total')] if isinstance(usage.get(src), int)}
            if counts:
                update(observation, usage_details=counts)
        except (ValueError, KeyError, IndexError, TypeError):
            raise RuntimeError('Hosted inference returned an unexpected response structure') from None
        if not isinstance(content, str):
            raise RuntimeError('Hosted inference returned non-text content')
        return content.strip()

    raise RuntimeError('Hosted inference retries exhausted')


def local_llama_chat(messages: list[dict], temperature: float = 0.0,
                     max_tokens: int = 600) -> str:
    """Native Ollama /api/chat fallback, unchanged."""
    base = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')
    if base.endswith('/v1'):
        base = base[:-3]
    response = requests.post(
        f'{base}/api/chat',
        json={'model': os.getenv('OLLAMA_MODEL', 'llama3.2:3b'), 'messages': messages,
              'stream': False, 'options': {'temperature': temperature, 'num_predict': max_tokens}},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()['message']['content'] or ''


@traced("embed_texts")
def embed_texts(texts):
    """Generate embeddings over verified HTTPS."""
    base_url = os.getenv('EMBEDDING_API_BASE_URL', '').rstrip('/')
    key = os.getenv('EMBEDDING_API_KEY')
    cert_setting = os.getenv('EMBEDDING_CA_CERT', '')
    if not base_url or not key or not cert_setting:
        raise RuntimeError('Set EMBEDDING_API_BASE_URL, EMBEDDING_API_KEY and EMBEDDING_CA_CERT in .env')
    if not base_url.startswith('https://'):
        raise ValueError('Embedding API requires HTTPS')
    cert_path = Path(cert_setting).expanduser()
    if not cert_path.is_absolute():
        cert_path = PROJECT_ROOT / cert_path
    if not cert_path.is_file():
        raise FileNotFoundError('Embedding CA certificate not found')
    from .ingest import validate_embedding_texts
    import math
    import ssl
    import httpx
    from openai import OpenAI, APIStatusError
    texts = list(texts)
    if not texts:
        return []
    model = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
    validate_embedding_texts(texts, model)
    size = int(os.getenv('EMBEDDING_BATCH_SIZE', '16'))
    retries = int(os.getenv('EMBEDDING_MAX_RETRIES', '3'))
    delay = float(os.getenv('EMBEDDING_RETRY_DELAY', '2'))
    if not 1 <= size <= 16 or not 0 <= retries <= 10 or not math.isfinite(delay) or delay < 0:
        raise ValueError('Invalid embedding batch or retry settings')
    vectors = []
    ssl_context = ssl.create_default_context(cafile=str(cert_path))
    with httpx.Client(verify=ssl_context, timeout=30.0) as http_client:
        with OpenAI(base_url=base_url, api_key=key, timeout=30.0,
                    max_retries=0, http_client=http_client) as client:
            for offset in range(0, len(texts), size):
                batch = texts[offset:offset + size]
                with span('embedding_batch', metadata={'offset': offset, 'count': len(batch), 'model': model}) as obs:
                    for attempt in range(retries + 1):
                        try:
                            result = client.embeddings.create(model=model, input=batch)
                            rows = sorted(result.data, key=lambda r: r.index)
                            if [r.index for r in rows] != list(range(len(batch))):
                                raise ValueError('Invalid embedding response indices')
                            if any(len(r.embedding) != 384 or not all(math.isfinite(v) for v in r.embedding) for r in rows):
                                raise ValueError('Invalid embedding vector')
                            vectors.extend(r.embedding for r in rows)
                            update(obs, output={'count': len(rows), 'retries': attempt})
                            break
                        except APIStatusError as exc:
                            if exc.status_code in (429, 503) and attempt < retries:
                                time.sleep(delay * 2 ** attempt)
                                continue
                            raise RuntimeError(f'Embedding batch offset {offset}: HTTP {exc.status_code}') from None
                        except Exception as exc:
                            raise RuntimeError(f'Embedding batch offset {offset}: {type(exc).__name__}') from None
    return vectors


def qwen_chat(messages, temperature=0.0, max_tokens=None):
    with span("qwen_generation", kind="generation", model=os.getenv("INFERENCE_MODEL", "Qwen/Qwen3-8B")) as obs:
        return _qwen_chat(messages, temperature, max_tokens, observation=obs)
