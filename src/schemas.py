"""
00 Schemas: Define data structures for document chunks, evidence, plans, safety decisions, and answers
Module: 00 Schemas
Author: Debanjali Ghosh
"""

from pydantic import BaseModel, Field
from typing import Literal

class DocumentChunk(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    url: str
    text: str
    source_date: str | None = None

class Evidence(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    url: str
    text: str
    score: float = 0.0

class Plan(BaseModel):
    intent: str
    queries: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)

class SafetyDecision(BaseModel):
    risk_level: Literal["emergency", "review", "normal"]
    reason: str
    escalate: bool = False

class Answer(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    status: Literal["answered", "abstained", "escalated", "error"] = "answered"
    session_id: str = ""
