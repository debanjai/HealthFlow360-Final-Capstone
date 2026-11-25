"""Bounded process-local session state; no disk persistence."""
from collections import OrderedDict
from copy import deepcopy
from threading import RLock
_sessions = OrderedDict()
_lock = RLock()

def get_history(session_id):
    with _lock:
        return deepcopy(_sessions.get(session_id, []))

def record(session_id, row):
    with _lock:
        history = _sessions.setdefault(session_id, [])
        history.append(deepcopy(row))
        del history[:-10]
        _sessions.move_to_end(session_id)
        while len(_sessions) > 256:
            _sessions.popitem(last=False)
