"""
Reasoning Agent

Analyzes anomalies and historical context to determine root causes.

ARCHITECTURE DESIGN:
- Base class defines reasoning interface
- MockReasoningAgent: Rule-based reasoning (no LLM needed)
- LLMReasoningAgent: Claude/GPT integration (TODO: implement when credits available)

This allows swapping implementations without changing the rest of the system.
"""

from .reasoning import ReasoningAgent, MockReasoningAgent

# TODO: Add LLMReasoningAgent when Claude/GPT credits available
# from .reasoning_llm import LLMReasoningAgent

__all__ = ["ReasoningAgent", "MockReasoningAgent"]
