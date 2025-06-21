"""
Framework adapters for different AI frameworks.

This module provides adapters for integrating various AI frameworks
into the multi-agent system.
"""

from .base_adapter import BaseFrameworkAdapter
from .google_adk_adapter import GoogleADKAdapter
from .langgraph_adapter import LangGraphAdapter
from .crewai_adapter import CrewAIAdapter

__all__ = [
    "BaseFrameworkAdapter",
    "GoogleADKAdapter", 
    "LangGraphAdapter",
    "CrewAIAdapter",
]
