"""Shared signatures. Implementations belong in owner modules."""
"""
11 Interfaces: Shared signatures for HealthFlow 360
Module: 11 Interfaces
Author: Gautam Singh
"""

from collections.abc import Sequence
from .schemas import DocumentChunk, Evidence, Answer, SafetyDecision

def ingest_documents(paths: Sequence[str]) -> list[DocumentChunk]:
    raise NotImplementedError

def hybrid_search(query: str, k: int = 5) -> list[Evidence]:
    raise NotImplementedError

def get_source(doc_id: str) -> list[DocumentChunk]:
    raise NotImplementedError

def get_history(session_id: str) -> list[dict]:
    raise NotImplementedError

def generate_answer(question: str, evidence: list[Evidence]) -> Answer:
    raise NotImplementedError

def assess_safety(question: str) -> SafetyDecision:
    raise NotImplementedError

def run_agent(question: str, session_id: str) -> Answer:
    raise NotImplementedError
