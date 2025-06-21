"""
Multi-agent coordination module.

This module provides the coordination layer for managing multi-agent
task execution across different AI frameworks.
"""

from .multi_agent_coordinator import MultiAgentCoordinator
from .workflow_engine import WorkflowEngine
from .task_executor import TaskExecutor
from .result_aggregator import ResultAggregator

__all__ = [
    "MultiAgentCoordinator",
    "WorkflowEngine", 
    "TaskExecutor",
    "ResultAggregator",
]
