"""
Core interfaces for the multi-agent system.

This module defines the fundamental interfaces and protocols that components
must implement to participate in the multi-agent system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator, Protocol, runtime_checkable

from .models import (
    Task, AgentConfig, AgentInstance, MultiAgentConfig, WorkflowConfig,
    TaskResult, AgentExecutionResult, MultiAgentResult, ExecutionContext,
    ExecutionMetrics, ToolCallResult, KnowledgeBaseQueryResult,
    Session, MemoryEntry
)
from .enums import FrameworkCapability, WorkflowType, SessionType


class MemoryManager(ABC):
    """Abstract memory manager for handling conversation memory.

    This class manages conversation memory, providing storage, retrieval,
    and search capabilities for agent memory and context.
    """

    @abstractmethod
    async def store_memory(
        self,
        session_id: str,
        content: str,
        memory_type: str,
        session_type: SessionType = SessionType.SINGLE_CHAT,
        agent_id: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """Store a memory entry.

        Args:
            session_id: Session identifier
            content: Memory content
            memory_type: Type of memory (conversation, fact, preference, context)
            session_type: Type of session
            agent_id: Optional agent identifier
            importance: Importance score (0-1)
            tags: Optional memory tags
            metadata: Optional additional metadata

        Returns:
            Created memory entry
        """
        pass

    @abstractmethod
    async def retrieve_memories(
        self,
        session_id: str,
        session_type: SessionType = SessionType.SINGLE_CHAT,
        memory_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[MemoryEntry]:
        """Retrieve memories for a session.

        Args:
            session_id: Session identifier
            session_type: Type of session
            memory_type: Optional memory type filter
            agent_id: Optional agent identifier filter
            limit: Maximum number of memories to return
            min_importance: Minimum importance threshold

        Returns:
            List of memory entries
        """
        pass

    @abstractmethod
    async def search_memories(
        self,
        session_id: str,
        query: str,
        session_type: SessionType = SessionType.SINGLE_CHAT,
        memory_type: Optional[str] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[MemoryEntry]:
        """Search memories by content similarity.

        Args:
            session_id: Session identifier
            query: Search query
            session_type: Type of session
            memory_type: Optional memory type filter
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score

        Returns:
            List of matching memory entries
        """
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry.

        Args:
            memory_id: Memory identifier

        Returns:
            True if memory was deleted, False if not found
        """
        pass

    @abstractmethod
    async def cleanup_old_memories(
        self,
        retention_days: int = 30,
        min_importance: float = 0.1
    ) -> int:
        """Clean up old or low-importance memories.

        Args:
            retention_days: Number of days to retain memories
            min_importance: Minimum importance to retain

        Returns:
            Number of memories cleaned up
        """
        pass


@runtime_checkable
class AgentFrameworkProtocol(Protocol):
    """Protocol that all agent frameworks must implement."""
    
    async def initialize(self) -> None:
        """Initialize the framework."""
        ...
    
    async def cleanup(self) -> None:
        """Clean up framework resources."""
        ...
    
    async def create_agent(self, config: AgentConfig) -> AgentInstance:
        """Create an agent instance."""
        ...
    
    async def execute_task(self, agent_id: str, task: Task, context: ExecutionContext) -> AgentExecutionResult:
        """Execute a task with an agent."""
        ...


@runtime_checkable
class WorkflowProtocol(Protocol):
    """Protocol for workflow execution engines."""
    
    async def execute(
        self, 
        config: WorkflowConfig, 
        task: Task, 
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute a workflow."""
        ...


@runtime_checkable
class StreamingCapable(Protocol):
    """Protocol for components that support streaming execution."""
    
    async def execute_streaming(
        self,
        *args: Any,
        **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute with streaming results."""
        ...


@runtime_checkable
class BatchProcessingCapable(Protocol):
    """Protocol for components that support batch processing."""
    
    async def execute_batch(
        self,
        tasks: List[Task],
        *args: Any,
        **kwargs: Any
    ) -> List[TaskResult]:
        """Execute multiple tasks in batch."""
        ...


@runtime_checkable
class MonitoringCapable(Protocol):
    """Protocol for components that support monitoring."""
    
    def get_metrics(self) -> ExecutionMetrics:
        """Get execution metrics."""
        ...
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        ...


class BaseFrameworkAdapter(ABC):
    """Base class for all framework adapters.
    
    This class provides the common interface that all AI framework adapters
    must implement to integrate with the multi-agent system.
    """
    
    def __init__(self, framework_name: str, version: str):
        self.framework_name = framework_name
        self.version = version
        self._initialized = False
        self._capabilities: List[FrameworkCapability] = []
        self._memory_manager: Optional[MemoryManager] = None
    
    @property
    def name(self) -> str:
        """Get framework name."""
        return self.framework_name
    
    @property
    def version_info(self) -> str:
        """Get framework version."""
        return self.version
    
    @property
    def capabilities(self) -> List[FrameworkCapability]:
        """Get framework capabilities."""
        return self._capabilities.copy()
    
    @property
    def is_initialized(self) -> bool:
        """Check if framework is initialized."""
        return self._initialized
    
    @abstractmethod
    def set_memory_manager(self, memory_manager: Optional[MemoryManager]) -> None:
        """Set the memory manager."""
        pass

    @abstractmethod
    def set_mcp_tool_manager(self, mcp_tool_manager: Optional[Any]) -> None:
        """Set the MCP tool manager."""
        pass
    
    def supports_capability(self, capability: FrameworkCapability) -> bool:
        """Check if framework supports a specific capability."""
        return capability in self._capabilities
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the framework."""
        pass

    

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up framework resources."""
        pass
    
    @abstractmethod
    async def create_agent(self, config: AgentConfig,context: ExecutionContext) -> AgentInstance:
        """Create an agent instance."""
        pass
    
    @abstractmethod
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent instance."""
        pass
    
    @abstractmethod
    async def execute_task(
        self, 
        agent_id: str, 
        task: Task, 
        context: ExecutionContext
    ) -> AgentExecutionResult:
        """Execute a task with an agent."""
        pass
    
    
    @abstractmethod
    async def query_knowledge_base(
        self, 
        agent_id: str, 
        kb_id: str, 
        kb_name: str, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[ExecutionContext] = None
    ) -> KnowledgeBaseQueryResult:
        """Query a knowledge base through an agent."""
        pass
    
    # Optional methods for advanced capabilities
    async def execute_task_streaming(
        self, 
        agent_id: str, 
        task: Task, 
        context: ExecutionContext
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a task with streaming results (optional)."""
        raise NotImplementedError("Streaming not supported by this framework")
    
    async def execute_multi_agent_task(
        self, 
        agents: List[AgentInstance], 
        task: Task, 
        workflow_config: WorkflowConfig,
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute a multi-agent task (optional)."""
        raise NotImplementedError("Multi-agent execution not supported by this framework")


class MultiAgentCoordinator(ABC):
    """Abstract coordinator for multi-agent system execution.
    
    This class orchestrates the execution of tasks across multiple agents
    and different AI frameworks.
    """
    
    @abstractmethod
    async def execute_task(
        self, 
        config: MultiAgentConfig, 
        task: Task
    ) -> MultiAgentResult:
        """Execute a task using the multi-agent system."""
        pass
    
    @abstractmethod
    async def execute_task_stream(
        self, 
        config: MultiAgentConfig, 
        task: Task
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a task with streaming results."""
        pass
    
    @abstractmethod
    async def execute_batch_tasks(
        self, 
        config: MultiAgentConfig, 
        tasks: List[Task]
    ) -> List[MultiAgentResult]:
        """Execute multiple tasks in batch."""
        pass


class WorkflowEngine(ABC):
    """Abstract workflow execution engine.
    
    This class handles the execution of different workflow types
    (hierarchical, sequential, parallel, etc.).
    """
    
    @abstractmethod
    async def execute_workflow(
        self, 
        workflow_type: WorkflowType,
        config: WorkflowConfig, 
        task: Task, 
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute a workflow."""
        pass
    
    @abstractmethod
    def supports_workflow_type(self, workflow_type: WorkflowType) -> bool:
        """Check if workflow type is supported."""
        pass


class TaskExecutor(ABC):
    """Abstract task executor.
    
    This class handles the low-level execution of individual tasks.
    """
    
    @abstractmethod
    async def execute(
        self, 
        task: Task, 
        agent: AgentInstance, 
        context: ExecutionContext
    ) -> TaskResult:
        """Execute a single task."""
        pass


class ResultAggregator(ABC):
    """Abstract result aggregator.

    This class handles the aggregation of results from multiple agents.
    """

    @abstractmethod
    async def aggregate_results(
        self,
        task: Task,
        agent_results: Dict[str, AgentExecutionResult],
        strategy: str = "default"
    ) -> MultiAgentResult:
        """Aggregate results from multiple agents."""
        pass


class SessionManager(ABC):
    """Abstract session manager for handling user sessions.

    This class manages user sessions, providing session lifecycle management,
    context storage, and session-based operations.
    """

    @abstractmethod
    async def create_session(
        self,
        session_id: str,
        user_id: str,
        session_type: SessionType = SessionType.SINGLE_CHAT,
        **kwargs: Any
    ) -> Session:
        """Create a new session.

        Args:
            session_id: Unique session identifier
            user_id: User identifier
            session_type: Type of session (single chat or group chat)
            **kwargs: Additional session parameters

        Returns:
            Created session instance
        """
        pass

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session instance if found, None otherwise
        """
        pass

    @abstractmethod
    async def update_session(self, session_id: str, **updates: Any) -> bool:
        """Update session properties.

        Args:
            session_id: Session identifier
            **updates: Properties to update

        Returns:
            True if session was updated, False if not found
        """
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        pass

    @abstractmethod
    async def list_user_sessions(
        self,
        user_id: str,
        active_only: bool = True
    ) -> List[Session]:
        """List sessions for a user.

        Args:
            user_id: User identifier
            active_only: Whether to return only active sessions

        Returns:
            List of user sessions
        """
        pass

    @abstractmethod
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        pass

