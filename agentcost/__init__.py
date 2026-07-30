"""agentcost — cost observability for LLM agent code.

Attributes spend to code paths and catches prompt-cache failures that are silent
by construction: a cached prefix below the model's minimum returns no error, it
simply never caches and pays full input price forever.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
