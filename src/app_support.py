"""Lightweight Streamlit request safeguards; no model imports or sensitive payload logging."""
"""
11 App Support: Streamlit Request Safeguards and Observability
Module: 11 App Support
Author: Gautam Singh

"""
from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from typing import Callable, TypeVar

T = TypeVar('T')
logger = logging.getLogger('healthflow.app')


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


class RateLimiter:
    """Sliding-window limiter scoped to one Streamlit session (not a global quota)."""
    def __init__(self, max_calls: int = 10, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        if max_calls < 1 or window_s <= 0:
            raise ValueError('Invalid rate limit')
        self.max_calls, self.window_s, self.clock = max_calls, window_s, clock
        self.calls: deque[float] = deque()

    def allow(self) -> bool:
        now = self.clock()
        while self.calls and now - self.calls[0] >= self.window_s:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            return False
        self.calls.append(now)
        return True

    def retry_after(self) -> float:
        now = self.clock()
        while self.calls and now - self.calls[0] >= self.window_s:
            self.calls.popleft()
        return max(0.0, self.window_s - (now - self.calls[0])) if len(self.calls) >= self.max_calls else 0.0


def validate_question(question: str, max_chars: int = 2000) -> str | None:
    if not question or not question.strip():
        return 'Please enter a question.'
    if len(question) > max_chars:
        return f'Please keep questions to {max_chars:,} characters or fewer.'
    return None


def validate_report(report: str, max_chars: int = 15000) -> str | None:
    if not report or not report.strip():
        return 'Please paste a radiology report.'
    if len(report) > max_chars:
        return f'Please keep reports to {max_chars:,} characters or fewer.'
    return None


def measured_call(action: Callable[[], T], *, trace_id: str, mode: str) -> tuple[T, float]:
    """Measure work and log only trace, mode, status and elapsed time."""
    start = time.perf_counter()
    try:
        result = action()
    except Exception as exc:
        logger.error('trace=%s mode=%s outcome=error type=%s latency_s=%.2f',
                     trace_id, mode, type(exc).__name__, time.perf_counter() - start)
        raise
    elapsed = time.perf_counter() - start
    logger.info('trace=%s mode=%s outcome=completed latency_s=%.2f', trace_id, mode, elapsed)
    return result, elapsed


from contextlib import contextmanager
from .observability import span, update, trace_details, flush_traces

@contextmanager
def request_trace(mode: str, session_id: str, **metadata):
    request_id = new_trace_id()
    start = time.perf_counter()
    state = {'request_id': request_id, 'status': 'error'}
    try:
        with span('healthflow360-streamlit-request', metadata={
            'request_id': request_id, 'mode': mode, 'session_id': session_id,
            **metadata
        }) as obs:
            state['langfuse_trace_id'], state['trace_url'] = trace_details(obs)
            try:
                yield state
            finally:
                state['elapsed'] = time.perf_counter() - start
                output = {k: state[k] for k in ('status', 'urgency', 'citation_count', 'summary_available') if k in state}
                update(obs, output=output, **({'level': 'ERROR'} if state['status'] == 'error' else {}))
                logger.info('trace=%s mode=%s outcome=%s latency_s=%.2f', request_id, mode, state['status'], state['elapsed'])
    finally:
        flush_traces()
