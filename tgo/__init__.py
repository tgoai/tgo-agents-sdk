"""
TGO Multi-Agent System

A powerful, framework-agnostic multi-agent system that orchestrates AI agents 
across different frameworks with unified interfaces, memory management, and 
flexible workflow execution.

Example usage:
    from tgo.agents import MultiAgentCoordinator, AdapterRegistry, GoogleADKAdapter
    from tgo.agents.core.models import MultiAgentConfig, AgentConfig, Task
    from tgo.agents.core.enums import AgentType, WorkflowType
    
    # Initialize system
    registry = AdapterRegistry()
    registry.register("google-adk", GoogleADKAdapter())
    
    coordinator = MultiAgentCoordinator(registry)
    
    # Configure and execute
    config = MultiAgentConfig(...)
    task = Task(...)
    result = await coordinator.execute_task(config, task)
"""

__version__ = "1.0.0"
__author__ = "TGO Team"
__email__ = "tangtaoit@githubim.com"

# Re-export main components for convenience
from .agents import (
    # Core components
    MultiAgentCoordinator,
    AdapterRegistry,
    
    # Framework adapters
    GoogleADKAdapter,
    
    # Memory and session management
    InMemoryMemoryManager,
    InMemorySessionManager,
    
    # Core models
    Task,
    Agent, 
    AgentConfig,
    AgentInstance,
    MultiAgentConfig,
    WorkflowConfig,
    Session,
    SessionConfig,
    MemoryEntry,
    MemoryConfig,
    TaskResult,
    AgentExecutionResult, 
    MultiAgentResult,
    ExecutionContext,
    ExecutionMetrics,
    ToolCallResult,
    KnowledgeBaseQueryResult,
    
    # Enums
    AgentType,
    AgentStatus, 
    TaskType,
    TaskStatus,
    TaskPriority,
    SessionType,
    WorkflowType,
    ExecutionStrategy,
    FrameworkCapability,
    
    # Exceptions
    MultiAgentError,
    FrameworkNotFoundError,
    AgentCreationError, 
    WorkflowExecutionError,
    WorkflowConfigurationError,
    TaskExecutionError,
)

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
