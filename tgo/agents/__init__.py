"""
Multi-agent system package.

This package provides a comprehensive multi-agent system with support for
multiple AI frameworks, different workflow types, and advanced coordination.
"""

# Core components
from .coordinator.multi_agent_coordinator import MultiAgentCoordinator
from .registry.adapter_registry import AdapterRegistry

# Framework adapters
from .adapters.google_adk_adapter import GoogleADKAdapter
# Note: Other adapters would be imported when implemented
# from .adapters.langgraph_adapter import LangGraphAdapter
# from .adapters.crewai_adapter import CrewAIAdapter

# Memory and session management
from .memory.in_memory_memory_manager import InMemoryMemoryManager
from .memory.in_memory_session_manager import InMemorySessionManager

# Core models and enums
from .core.models import (
    Task, Agent, AgentConfig, AgentInstance, MultiAgentConfig,
    WorkflowConfig, Session, SessionConfig, MemoryEntry, MemoryConfig,
    TaskResult, AgentExecutionResult, MultiAgentResult,
    ExecutionContext, ExecutionMetrics,
    ToolCallResult, KnowledgeBaseQueryResult
)

from .core.enums import (
    AgentType, AgentStatus, TaskType, TaskStatus, TaskPriority,
    SessionType, WorkflowType, ExecutionStrategy, FrameworkCapability
)

# Exceptions
from .core.exceptions import (
    MultiAgentError, FrameworkNotFoundError, AgentCreationError,
    WorkflowExecutionError, WorkflowConfigurationError, TaskExecutionError
)

__version__ = "1.0.0"

__all__ = [
    # Core components
    "MultiAgentCoordinator",
    "AdapterRegistry",

    # Framework adapters
    "GoogleADKAdapter",

    # Memory and session management
    "InMemoryMemoryManager",
    "InMemorySessionManager",

    # Core models
    "Task",
    "Agent",
    "AgentConfig",
    "AgentInstance",
    "MultiAgentConfig",
    "WorkflowConfig",
    "Session",
    "SessionConfig",
    "MemoryEntry",
    "MemoryConfig",
    "TaskResult",
    "AgentExecutionResult",
    "MultiAgentResult",
    "ExecutionContext",
    "ExecutionMetrics",
    "ToolCallResult",
    "KnowledgeBaseQueryResult",

    # Enums
    "AgentType",
    "AgentStatus",
    "TaskType",
    "TaskStatus",
    "TaskPriority",
    "SessionType",
    "WorkflowType",
    "ExecutionStrategy",
    "FrameworkCapability",

    # Exceptions
    "MultiAgentError",
    "FrameworkNotFoundError",
    "AgentCreationError",
    "WorkflowExecutionError",
    "WorkflowConfigurationError",
    "TaskExecutionError",
]
