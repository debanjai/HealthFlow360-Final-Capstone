"""Allowlisted tool dispatch with bounded steps and metadata-only tracing."""
from .observability import span, update

class Executor:
    def __init__(self, registry, max_steps=10):
        self.registry = dict(registry)
        self.max_steps = min(10, max_steps)
        self.steps = 0

    def call(self, name, *args, **kwargs):
        if self.steps >= self.max_steps:
            raise RuntimeError('Step budget exhausted')
        if name not in self.registry:
            raise ValueError('Unknown tool')
        self.steps += 1
        with span('agent_tool', metadata={'tool':name,'step':self.steps}) as obs:
            result = self.registry[name](*args, **kwargs)
            update(obs, output={'completed':True})
            return result
