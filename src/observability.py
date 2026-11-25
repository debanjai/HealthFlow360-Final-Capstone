"""Opt-in metadata-only Langfuse tracing; never capture arguments or return text."""
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / '.env', override=True)

@contextmanager
def span(name, kind='span', **fields):
    obs = None
    cm = None
    if os.getenv('LANGFUSE_ENABLED', 'true').lower() == 'true' and os.getenv('LANGFUSE_PUBLIC_KEY') and os.getenv('LANGFUSE_SECRET_KEY'):
        try:
            from langfuse import get_client
            cm = get_client().start_as_current_observation(name=name, as_type=kind, **fields)
            obs = cm.__enter__()
        except Exception:
            cm = None
    try:
        yield obs
    except BaseException as exc:
        update(obs, level='ERROR', status_message=type(exc).__name__)
        raise
    finally:
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass

def update(obs, **fields):
    if obs is not None:
        try:
            obs.update(**fields)
        except Exception:
            pass

def traced(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with span(name) as obs:
                result = fn(*args, **kwargs)
                status = getattr(result, 'status', None)
                if status in ('answered', 'abstained', 'escalated', 'error'):
                    update(obs, output={'status': status}, **({'level':'ERROR'} if status == 'error' else {}))
                elif isinstance(result, list):
                    update(obs, output={'count': len(result)})
                return result
        return wrapped
    return decorate


def trace_details(obs):
    if obs is not None:
        try:
            from langfuse import get_client
            client = get_client()
            trace_id = client.get_current_trace_id()
            return trace_id, client.get_trace_url(trace_id=trace_id) if trace_id else None
        except Exception:
            pass
    return None, None


def flush_traces():
    if os.getenv('LANGFUSE_ENABLED', 'true').lower() == 'true' and os.getenv('LANGFUSE_PUBLIC_KEY') and os.getenv('LANGFUSE_SECRET_KEY'):
        try:
            from langfuse import get_client
            get_client().flush()
        except Exception:
            pass
