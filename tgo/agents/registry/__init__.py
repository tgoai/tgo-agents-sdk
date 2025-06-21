"""
Registry module for managing framework adapters.

This module provides the registry system for discovering and managing
different AI framework adapters in the multi-agent system.
"""

from .adapter_registry import AdapterRegistry, get_registry

__all__ = [
    "AdapterRegistry",
    "get_registry",
]
