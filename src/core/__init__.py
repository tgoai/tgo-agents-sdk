"""
Core abstractions for the multi-agent system.

This module provides the fundamental interfaces, data models, and exceptions
that define the architecture of the multi-agent system.
"""

from .interfaces import (
    # Core interfaces
    BaseFrameworkAdapter,
    MultiAgentCoordinator,
    WorkflowEngine,
    TaskExecutor,
    ResultAggregator,

    # Session and Memory interfaces
    SessionManager,
    MemoryManager,

    # Protocol definitions
    AgentFrameworkProtocol,
    WorkflowProtocol,

    # Feature interfaces
    StreamingCapable,
    BatchProcessingCapable,
    MonitoringCapable,
)

from .models import (
    # Core data models
    Task,
    Agent,
    AgentConfig,
    AgentInstance,
    MultiAgentConfig,
    WorkflowConfig,

    # Session and Memory models
    Session,
    SessionConfig,
    MemoryEntry,
    MemoryConfig,

    # Result models
    TaskResult,
    AgentExecutionResult,
    MultiAgentResult,

    # Execution models
    ExecutionContext,
    ExecutionMetrics,

    # Tool and KB models
    ToolCallResult,
    KnowledgeBaseQueryResult,
)

from .enums import (
    # Agent types and states
    AgentType,
    AgentStatus,

    # Task types and states
    TaskType,
    TaskStatus,
    TaskPriority,

    # Session types
    SessionType,

    # Workflow types and strategies
    WorkflowType,
    ExecutionStrategy,

    # Framework capabilities
    FrameworkCapability,
)

from .exceptions import (
    # Base exceptions
    MultiAgentError,
    
    # Framework exceptions
    FrameworkError,
    FrameworkNotFoundError,
    FrameworkInitializationError,
    FrameworkUnavailableError,
    
    # Agent exceptions
    AgentError,
    AgentNotFoundError,
    AgentCreationError,
    AgentExecutionError,
    
    # Task exceptions
    TaskError,
    TaskExecutionError,
    TaskTimeoutError,
    TaskValidationError,
    
    # Workflow exceptions
    WorkflowError,
    WorkflowExecutionError,
    WorkflowConfigurationError,
)

__all__ = [
    # Interfaces
    "BaseFrameworkAdapter",
    "MultiAgentCoordinator",
    "WorkflowEngine",
    "TaskExecutor",
    "ResultAggregator",
    "SessionManager",
    "MemoryManager",
    "AgentFrameworkProtocol",
    "WorkflowProtocol",
    "StreamingCapable",
    "BatchProcessingCapable",
    "MonitoringCapable",

    # Models
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
    "FrameworkError",
    "FrameworkNotFoundError",
    "FrameworkInitializationError",
    "FrameworkUnavailableError",
    "AgentError",
    "AgentNotFoundError",
    "AgentCreationError",
    "AgentExecutionError",
    "TaskError",
    "TaskExecutionError",
    "TaskTimeoutError",
    "TaskValidationError",
    "WorkflowError",
    "WorkflowExecutionError",
    "WorkflowConfigurationError",
]
