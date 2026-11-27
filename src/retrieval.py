"""Hybrid retrieval (BM25 + dense embeddings)."""
"""
02 Retrieval: Hybrid retrieval (BM25 + dense embeddings) for HealthFlow 360
Module: 02 Retrieval
Author: Debanjali Ghosh
"""


from typing import Optional
import os
import time
from threading import RLock
from rank_bm25 import BM25Okapi
import numpy as np
from .schemas import DocumentChunk, Evidence
from .model_clients import embed_texts
from .ingest import load_corpus

_corpus: list[DocumentChunk] = []
_bm25: Optional[BM25Okapi] = None
_embeddings: Optional[np.ndarray] = None

from .observability import traced, span, update

_index_lock = RLock()
_index_model = None
_retry_after = 0.0
_reranker = None
_reranker_key = None
_reranker_retry = 0.0
_reranker_lock = RLock()


def _ensure_index():
    global _corpus, _bm25, _embeddings, _index_model, _retry_after
    with _index_lock:
        if not _corpus:
            _corpus = load_corpus()
            if not _corpus:
                return
            _bm25 = BM25Okapi([doc.text.lower().split() for doc in _corpus])
        model = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
        if _index_model != model:
            _embeddings = None
            _retry_after = 0.0
            _index_model = model
        if _embeddings is not None or time.monotonic() < _retry_after:
            return
        with span('dense_index_build', metadata={'model': model, 'count': len(_corpus)}) as obs:
            try:
                matrix = np.asarray(embed_texts([doc.text for doc in _corpus]), dtype=np.float32)
                if matrix.shape != (len(_corpus), 384) or not np.isfinite(matrix).all():
                    raise ValueError('Invalid dense index')
                _embeddings = matrix
                update(obs, output={'status': 'ready', 'count': len(matrix)})
            except Exception as exc:
                _embeddings = None
                _retry_after = time.monotonic() + 60
                update(obs, level='ERROR', status_message=type(exc).__name__, output={'status': 'bm25_fallback'})

def _rerank(query, candidates):
    global _reranker, _reranker_key, _reranker_retry
    key = (os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-base'),
           os.getenv('RERANKER_DEVICE', 'cpu'))
    with _reranker_lock:
        if key != _reranker_key:
            _reranker, _reranker_retry, _reranker_key = None, 0.0, key
        if time.monotonic() < _reranker_retry:
            raise RuntimeError('Reranker cooldown')
        try:
            if _reranker is None:
                with span('reranker_load', metadata={'model': key[0], 'device': key[1]}):
                    from sentence_transformers import CrossEncoder
                    _reranker = CrossEncoder(key[0], device=key[1], max_length=512)
            scores = np.asarray(_reranker.predict(
                [(query, _corpus[j].text) for j in candidates], show_progress_bar=False))
            if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
                raise ValueError('Invalid reranker scores')
            return candidates[np.argsort(scores)[::-1]]
        except Exception:
            _reranker = None
            _reranker_retry = time.monotonic() + 60
            raise


@traced("hybrid_search")
def hybrid_search(query: str, k: int = 5) -> list[Evidence]:
    """Hybrid BM25 + embedding search. Falls back to BM25 if embeddings unavailable."""
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    _ensure_index()
    if not _corpus or _bm25 is None:
        return []

    # BM25 scores
    with span("bm25_search"):
        bm25_scores = _bm25.get_scores(query.lower().split())
    bm25_norm = bm25_scores / (bm25_scores.max() + 1e-8) if bm25_scores.max() > 0 else bm25_scores

    mode = "bm25_only"
    # Embedding scores (if available)
    if _embeddings is not None:
        try:
            q_emb = np.array(embed_texts([query])[0])
            if q_emb.shape != (384,) or not np.isfinite(q_emb).all():
                raise ValueError("Invalid query embedding")
            emb_scores = np.dot(_embeddings, q_emb) / (
                np.linalg.norm(_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
            )
            emb_norm = (emb_scores - emb_scores.min()) / (emb_scores.max() - emb_scores.min() + 1e-8)
            combined = 0.4 * bm25_norm + 0.6 * emb_norm
            mode = "hybrid"
        except Exception as exc:
            with span("dense_query_fallback") as obs:
                update(obs, level="ERROR", status_message=type(exc).__name__)
            combined = bm25_norm
    else:
        combined = bm25_norm

    order = np.argsort(combined)[::-1]
    rerank_status = 'disabled'
    if os.getenv('RERANKER_ENABLED', 'false').lower() == 'true':
        rerank_status = 'skipped_no_dense'
        if mode == 'hybrid':
            with span('cross_encoder_rerank') as obs:
                try:
                    n = int(os.getenv('RERANKER_TOP_N', '20'))
                    if not 1 <= n <= 100:
                        raise ValueError('RERANKER_TOP_N must be 1..100')
                    candidates = order[:max(k, n)]
                    order = _rerank(query, candidates)
                    mode = 'hybrid_reranked'
                    rerank_status = 'success'
                    update(obs, output={'status': 'success', 'candidates': len(candidates)})
                except Exception as exc:
                    rerank_status = 'fallback_hybrid'
                    update(obs, level='ERROR', status_message=type(exc).__name__, output={'status': rerank_status})
    top_idx = order[:k]
    results = []
    for idx in top_idx:
        doc = _corpus[idx]
        results.append(Evidence(
            doc_id=doc.doc_id,
            chunk_id=doc.chunk_id,
            title=doc.title,
            url=doc.url,
            text=doc.text,
            score=float(combined[idx])
        ))
    with span("retrieval_result") as obs:
        update(obs, output={"retrieval_mode": mode, "count": len(results), "dense_index_ready": _embeddings is not None, "reranker_status": rerank_status})
    return results

def get_source(doc_id: str) -> list[DocumentChunk]:
    _ensure_index()
    return [d for d in _corpus if d.doc_id == doc_id]
