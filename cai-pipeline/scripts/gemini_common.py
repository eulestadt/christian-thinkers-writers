"""Deprecated shim. Pipeline LLM calls go through llm_common (deepseek-flash)."""
from llm_common import *  # noqa: F401,F403
from llm_common import LLMClient as GeminiPool  # noqa: F401
