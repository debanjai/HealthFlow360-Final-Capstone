"""src/ingest.py - Data ingestion and chunking module for HealthFlow 360.

01 Ingest: Data ingestion and chunking for HealthFlow 360
Module: 01 Ingest
Author: Debanjali Ghosh
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from src.schemas import DocumentChunk


def clean_text(raw_text: str | None) -> str:
    """Normalize whitespace and strip non-informative noise."""
    if not raw_text:
        return ""
    # Collapse irregular whitespace, newlines, and tabs into single spaces
    return " ".join(raw_text.strip().split())


from functools import lru_cache
import os

@lru_cache(maxsize=4)
def embedding_tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, trust_remote_code=False, use_fast=True)


def embedding_limit():
    limit = int(os.getenv('EMBEDDING_MAX_TOKENS', '512'))
    if not 1 <= limit <= 512:
        raise ValueError('EMBEDDING_MAX_TOKENS must be 1..512')
    return limit


def validate_embedding_texts(texts, model=None):
    tokenizer = embedding_tokenizer(model or os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5'))
    limit = embedding_limit()
    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Empty or invalid embedding text at index {i}')
        count = len(tokenizer.encode(text, add_special_tokens=True, truncation=False, verbose=False))
        if count > limit:
            raise ValueError(f'Embedding text index {i} has {count} tokens; limit {limit}. Rechunk before indexing.')


def chunk_text_tokens(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split on model-token offsets, preserving original text and overlap."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    tokenizer = embedding_tokenizer(os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5'))
    budget = min(chunk_size, embedding_limit() - tokenizer.num_special_tokens_to_add(pair=False))
    if not 0 <= overlap < budget:
        raise ValueError('Overlap must be smaller than chunk token budget')
    offsets = tokenizer(cleaned, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)['offset_mapping']
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + budget, len(offsets))
        while end > start:
            chunk = cleaned[offsets[start][0]:offsets[end-1][1]].strip()
            if chunk and len(tokenizer.encode(chunk, add_special_tokens=True, truncation=False, verbose=False)) <= embedding_limit():
                break
            end -= 1
        if end <= start:
            raise ValueError('Unable to create valid embedding chunk')
        chunks.append(chunk)
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_document(
    doc_id: str,
    title: str,
    url: str,
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[DocumentChunk]:
    """Convert a single document into stable, schema-validated DocumentChunk objects.

    Args:
        doc_id: Unique identifier for the parent source document.
        title: Title of the guideline or document.
        url: Source reference URL.
        text: Raw body text.
        chunk_size: Model-token window size.
        overlap: Model-token overlap window.

    Returns:
        List of DocumentChunk instances with deterministic IDs (e.g., 'doc_0').
    """
    raw_slices = chunk_text_tokens(text, chunk_size=chunk_size, overlap=overlap)
    chunks: List[DocumentChunk] = []

    for idx, slice_text in enumerate(raw_slices):
        if not slice_text.strip():
            continue
        chunks.append(
            DocumentChunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_{idx}",
                title=title,
                url=url,
                text=slice_text,
            )
        )

    return chunks


def build_corpus(
    raw_docs: Sequence[Dict[str, Any]],
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[DocumentChunk]:
    """Ingest, clean, deduplicate, and chunk an entire batch of raw documents.

    Args:
        raw_docs: Iterable of dicts containing 'doc_id', 'title', 'url', and 'text'.
        chunk_size: Model-token window size.
        overlap: Model-token overlap size.

    Returns:
        Deduplicated list of DocumentChunk instances.
    """
    corpus: List[DocumentChunk] = []
    seen_chunk_ids = set()

    for doc in raw_docs:
        doc_id = str(doc.get("doc_id", "")).strip()
        title = str(doc.get("title", "")).strip()
        url = str(doc.get("url", "")).strip()
        text = str(doc.get("text", "")).strip()

        if not doc_id or not text:
            continue

        doc_chunks = chunk_document(
            doc_id=doc_id,
            title=title,
            url=url,
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in doc_chunks:
            if chunk.chunk_id not in seen_chunk_ids and chunk.text.strip():
                seen_chunk_ids.add(chunk.chunk_id)
                corpus.append(chunk)

    return corpus


def save_corpus(
    chunks: Sequence[DocumentChunk],
    output_path: Union[str, Path] = "data/corpus.jsonl",
) -> Path:
    """Export chunks to JSON Lines format for downstream retrieval modules."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")

    return path


def load_corpus(
    corpus_path: Union[str, Path, None] = None,
) -> List[DocumentChunk]:
    """Load and validate a configurable JSONL corpus."""

    if corpus_path is None:
        corpus_path = os.getenv(
            "CORPUS_PATH",
            "data/corpus.jsonl"
        )

    path = Path(corpus_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {path.resolve()}"
        )

    loaded_chunks: List[DocumentChunk] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()

            if not line_str:
                continue

            data = json.loads(line_str)
            loaded_chunks.append(DocumentChunk(**data))

    return loaded_chunks


# Alias for backward and cross-team compatibility
load_corpus_jsonl = load_corpus
