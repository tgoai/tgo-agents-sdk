"""
Core data models for the multi-agent system.

This module defines the fundamental data structures used throughout
the multi-agent system, providing type safety and validation.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import uuid as uuid_lib
from mcp.types import Content, TextContent

from .enums import (
    AgentType, AgentStatus, TaskType, TaskStatus, TaskPriority,
    WorkflowType, ExecutionStrategy, SessionType
)

if TYPE_CHECKING:
    from typing import TYPE_CHECKING


class Task(BaseModel):
    """Core task model representing a unit of work to be executed."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    task_id: str = Field(
        default_factory=lambda: str(uuid_lib.uuid4()),
        description="Unique task identifier"
    )
    title: str = Field(..., description="Task title", min_length=1, max_length=200)
    description: Optional[str] = Field(None, description="Task description", max_length=1000)
    task_type: TaskType = Field(default=TaskType.SIMPLE, description="Type of task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    
    # Task data
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for the task")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data from task execution")
    
    # Execution configuration
    timeout_seconds: Optional[int] = Field(None, description="Task timeout in seconds", ge=1)
    max_retries: int = Field(default=0, description="Maximum retry attempts", ge=0)
    retry_count: int = Field(default=0, description="Current retry count", ge=0)
    
    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Task creation timestamp"
    )
    started_at: Optional[datetime] = Field(None, description="Task start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    
    # Relationships
    parent_task_id: Optional[str] = Field(None, description="Parent task ID for subtasks")
    subtask_ids: List[str] = Field(default_factory=list, description="List of subtask IDs")
    depends_on: List[str] = Field(default_factory=list, description="Task dependencies")
    
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
    
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == TaskStatus.RUNNING
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries and self.status == TaskStatus.FAILED
    
    def get_execution_duration(self) -> Optional[int]:
        """Get task execution duration in seconds."""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return None


class Agent(BaseModel):
    """Core agent model representing an AI agent."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    agent_id: str = Field(
        default_factory=lambda: str(uuid_lib.uuid4()),
        description="Unique agent identifier"
    )
    name: str = Field(..., description="Agent name", min_length=1, max_length=100)
    agent_type: AgentType = Field(..., description="Type of agent")
    description: Optional[str] = Field(None, description="Agent description", max_length=500)
    
    # Capabilities
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    model: str = Field(default="gemini-2.0-flash", description="AI model to use")
    
    # Configuration
    instructions: Optional[str] = Field(None, description="Agent instructions", max_length=2000)
    tools: List[str] = Field(default_factory=list, description="Available tools")
    knowledge_bases: List[str] = Field(default_factory=list, description="Available knowledge bases")
    
    # Runtime state
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Current agent status")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Agent creation timestamp"
    )
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    
    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return self.status == AgentStatus.IDLE
    
    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities
    
    def has_tool(self, tool: str) -> bool:
        """Check if agent has access to a specific tool."""
        return tool in self.tools


class AgentConfig(BaseModel):
    """Configuration for creating an agent."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    agent_id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent name", min_length=1, max_length=100)
    agent_type: AgentType = Field(..., description="Type of agent")
    description: Optional[str] = Field(None, description="Agent description", max_length=500)

    # Model configuration
    model: str = Field(default="gemini-2.0-flash", description="AI model to use")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    instructions: Optional[str] = Field(None, description="Agent instructions", max_length=2000)

    # Resources - tools can be functions or MCPTool objects
    tools: List[Any] = Field(default_factory=list, description="Available tools (functions or MCPTool objects)")
    knowledge_bases: List[str] = Field(default_factory=list, description="Available knowledge bases")

    # MCP (Model Context Protocol) configuration - deprecated, use tools array instead
    mcp_servers: List[str] = Field(default_factory=list, description="[Deprecated] Available MCP server IDs")
    mcp_tools: List[str] = Field(default_factory=list, description="[Deprecated] Specific MCP tools to enable")
    mcp_auto_approve: bool = Field(default=False, description="Auto-approve MCP tool calls")

    # Execution parameters
    max_iterations: int = Field(default=10, description="Maximum iterations", ge=1, le=100)
    timeout_seconds: Optional[int] = Field(None, description="Execution timeout", ge=1)
    temperature: float = Field(default=0.7, description="Model temperature", ge=0.0, le=2.0)

    # Framework-specific configuration
    framework_config: Dict[str, Any] = Field(default_factory=dict, description="Framework-specific config")

    def get_function_tools(self) -> List[Any]:
        """Get function tools from the tools list."""
        return [tool for tool in self.tools if callable(tool) and not hasattr(tool, 'server_id')]

    def get_mcp_tools(self) -> List[Any]:
        """Get MCP tools from the tools list."""
        return [tool for tool in self.tools if hasattr(tool, 'server_id') and hasattr(tool, 'name')]

    def get_string_tools(self) -> List[str]:
        """Get string tool names from the tools list."""
        return [tool for tool in self.tools if isinstance(tool, str)]

    def has_mcp_tools(self) -> bool:
        """Check if agent has any MCP tools."""
        return len(self.get_mcp_tools()) > 0

    def has_function_tools(self) -> bool:
        """Check if agent has any function tools."""
        return len(self.get_function_tools()) > 0


class AgentInstance(BaseModel):
    """Runtime instance of an agent."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    agent_id: str = Field(..., description="Agent identifier")
    config: AgentConfig = Field(..., description="Agent configuration")
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Current status")
    
    # Runtime data
    current_task_id: Optional[str] = Field(None, description="Currently executing task ID")
    session_data: Dict[str, Any] = Field(default_factory=dict, description="Session data")
    memory: Dict[str, Any] = Field(default_factory=dict, description="Agent memory")
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Instance creation timestamp"
    )
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    
    def is_available(self) -> bool:
        """Check if agent instance is available."""
        return self.status == AgentStatus.IDLE
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)


class MultiAgentConfig(BaseModel):
    """Configuration for multi-agent system execution."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    framework: str = Field(..., description="AI framework to use")
    fallback_frameworks: List[str] = Field(default_factory=list, description="Fallback frameworks")

    # Agent configurations
    agents: List[AgentConfig] = Field(..., description="Agent configurations", min_length=1)

    # Workflow configuration
    workflow: "WorkflowConfig" = Field(..., description="Workflow configuration")

    # Security and isolation
    security: Optional[Dict[str, Any]] = Field(None, description="Security configuration")

    # Monitoring and debugging
    monitoring: Optional[Dict[str, Any]] = Field(None, description="Monitoring configuration")


class WorkflowConfig(BaseModel):
    """Configuration for workflow execution."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    workflow_type: WorkflowType = Field(..., description="Type of workflow")
    execution_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.FAIL_FAST,
        description="Execution strategy"
    )

    # Workflow-specific configuration
    manager_agent_id: Optional[str] = Field(None, description="Manager agent ID for hierarchical workflows")
    expert_agent_ids: List[str] = Field(default_factory=list, description="Expert agent IDs")

    # Execution parameters
    max_concurrent_agents: int = Field(default=3, description="Maximum concurrent agents", ge=1)
    timeout_seconds: Optional[int] = Field(None, description="Workflow timeout", ge=1)

    # Task decomposition
    task_decomposition: Optional[Dict[str, Any]] = Field(None, description="Task decomposition config")

    # Result aggregation
    result_aggregation: Optional[Dict[str, Any]] = Field(None, description="Result aggregation config")

    # Pipeline configuration (for sequential workflows)
    pipeline_stages: List[Dict[str, Any]] = Field(default_factory=list[Dict[str, Any]], description="Pipeline stages")

    # Custom workflow definition
    workflow_definition: Optional[Dict[str, Any]] = Field(None, description="Custom workflow definition")

    # Streaming configuration
    streaming_config: Optional[Dict[str, Any]] = Field(None, description="Streaming configuration")

    # Batch configuration
    batch_config: Optional[Dict[str, Any]] = Field(None, description="Batch processing configuration")

    # Retry configuration
    retry_config: Optional[Dict[str, Any]] = Field(None, description="Retry configuration")


class TaskResult(BaseModel):
    """Result of task execution."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    task_id: str = Field(..., description="Task identifier")
    success: bool = Field(..., description="Whether execution was successful")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result data")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Execution metrics
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds", ge=0)
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")

    # Agent information
    agent_id: Optional[str] = Field(None, description="Executing agent ID")
    agent_type: Optional[AgentType] = Field(None, description="Executing agent type")

    def is_successful(self) -> bool:
        """Check if task execution was successful."""
        return self.success and self.error_message is None


class ToolCallResult(BaseModel):
    """Result of a tool call."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    tool_name: str = Field(..., description="Tool name", min_length=1)
    tool_id: str = Field(..., description="Tool identifier", min_length=1)
    success: bool = Field(..., description="Whether call was successful")
    result: Optional[Dict[str, Any]] = Field(None, description="Tool call result")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time_ms: Optional[int] = Field(None, description="Execution time", ge=0)

    def is_successful(self) -> bool:
        """Check if tool call was successful."""
        return self.success and self.error_message is None


class KnowledgeBaseQueryResult(BaseModel):
    """Result of a knowledge base query."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    kb_name: str = Field(..., description="Knowledge base name", min_length=1)
    kb_id: str = Field(..., description="Knowledge base identifier", min_length=1)
    query: str = Field(..., description="Query text", min_length=1)
    success: bool = Field(..., description="Whether query was successful")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="Query results")
    results_count: Optional[int] = Field(None, description="Number of results", ge=0)
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time_ms: Optional[int] = Field(None, description="Execution time", ge=0)

    def is_successful(self) -> bool:
        """Check if query was successful."""
        return self.success and self.error_message is None

    def has_results(self) -> bool:
        """Check if query returned results."""
        return self.results is not None and len(self.results) > 0


class AgentExecutionResult(BaseModel):
    """Result of agent execution with detailed information."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    success: bool = Field(..., description="Whether execution was successful")
    result: Optional[Dict[str, Any]] = Field(None, description="Execution result")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Execution details
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds", ge=0)
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")

    # Tool and knowledge base interactions
    tool_calls: List[ToolCallResult] = Field(default_factory=list[ToolCallResult], description="Tool call results")
    kb_queries: List[KnowledgeBaseQueryResult] = Field(default_factory=list[KnowledgeBaseQueryResult], description="KB query results")

    # Reasoning and intermediate steps
    reasoning_steps: List[str] = Field(default_factory=list, description="Reasoning steps")
    intermediate_results: List[Dict[str, Any]] = Field(default_factory=list[Dict[str, Any]], description="Intermediate results")

    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.success and self.error_message is None

    def has_tool_calls(self) -> bool:
        """Check if there were tool calls."""
        return len(self.tool_calls) > 0

    def has_kb_queries(self) -> bool:
        """Check if there were knowledge base queries."""
        return len(self.kb_queries) > 0


class MultiAgentResult(BaseModel):
    """Result of multi-agent system execution."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    success: bool = Field(..., description="Whether overall execution was successful")
    result: Optional[Dict[str, Any]] = Field(None, description="Aggregated result")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Individual agent results
    agent_results: Dict[str, AgentExecutionResult] = Field(
        default_factory=dict,
        description="Results from individual agents"
    )

    # Task results
    task_results: Dict[str, TaskResult] = Field(
        default_factory=dict,
        description="Results from individual tasks"
    )

    # Execution metrics
    total_execution_time_ms: Optional[int] = Field(None, description="Total execution time", ge=0)
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")

    # Workflow information
    workflow_type: Optional[WorkflowType] = Field(None, description="Workflow type used")
    agents_used: List[str] = Field(default_factory=list, description="Agent IDs that participated")

    def is_successful(self) -> bool:
        """Check if multi-agent execution was successful."""
        return self.success and self.error_message is None

    def get_successful_agents(self) -> List[str]:
        """Get list of agents that executed successfully."""
        return [
            agent_id for agent_id, result in self.agent_results.items()
            if result.is_successful()
        ]

    def get_failed_agents(self) -> List[str]:
        """Get list of agents that failed."""
        return [
            agent_id for agent_id, result in self.agent_results.items()
            if not result.is_successful()
        ]


class ExecutionContext(BaseModel):
    """Context information for task execution."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    execution_id: str = Field(
        default_factory=lambda: str(uuid_lib.uuid4()),
        description="Unique execution identifier"
    )
    task_id: str = Field(..., description="Task being executed")
    agent_id: Optional[str] = Field(None, description="Executing agent ID")

    # Context data
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")

    # Execution environment
    framework_name: Optional[str] = Field(None, description="Framework being used")
    workflow_type: Optional[WorkflowType] = Field(None, description="Workflow type")

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Context creation time"
    )

    # Additional context data
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ExecutionMetrics(BaseModel):
    """Metrics collected during execution."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    execution_id: str = Field(..., description="Execution identifier")

    # Timing metrics
    total_duration_ms: Optional[int] = Field(None, description="Total execution duration", ge=0)
    agent_execution_time_ms: Optional[int] = Field(None, description="Agent execution time", ge=0)
    tool_execution_time_ms: Optional[int] = Field(None, description="Tool execution time", ge=0)
    kb_query_time_ms: Optional[int] = Field(None, description="KB query time", ge=0)

    # Resource metrics
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB", ge=0)
    cpu_usage_percent: Optional[float] = Field(None, description="CPU usage percentage", ge=0, le=100)

    # Token metrics (for LLM usage)
    input_tokens: Optional[int] = Field(None, description="Input tokens used", ge=0)
    output_tokens: Optional[int] = Field(None, description="Output tokens generated", ge=0)
    total_tokens: Optional[int] = Field(None, description="Total tokens used", ge=0)

    # Interaction metrics
    tool_calls_count: int = Field(default=0, description="Number of tool calls", ge=0)
    kb_queries_count: int = Field(default=0, description="Number of KB queries", ge=0)

    # Quality metrics
    success_rate: Optional[float] = Field(None, description="Success rate", ge=0, le=1)
    error_count: int = Field(default=0, description="Number of errors", ge=0)

    def calculate_total_tokens(self) -> int:
        """Calculate total tokens if not already set."""
        if self.total_tokens is not None:
            return self.total_tokens

        input_tokens = self.input_tokens or 0
        output_tokens = self.output_tokens or 0
        return input_tokens + output_tokens


class Session(BaseModel):
    """User session model for managing conversation context."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    session_id: str = Field(..., description="Session ID, required")
    user_id: str = Field(..., description="User ID, required")
    session_type: SessionType = Field(
        default=SessionType.SINGLE_CHAT,
        description="Session type: 1-single chat, 2-group chat"
    )

    # Session state
    status: str = Field(default="active", description="Session status: active, inactive, expired")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Session creation timestamp"
    )
    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last activity timestamp"
    )
    expires_at: Optional[datetime] = Field(None, description="Session expiration time")

    # Session data
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")

    def is_group_chat(self) -> bool:
        """Check if this is a group chat session."""
        return self.session_type == SessionType.GROUP_CHAT

    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == "active"

    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)


class SessionConfig(BaseModel):
    """Configuration for session management."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    session_timeout_minutes: int = Field(
        default=30,
        description="Session timeout in minutes",
        ge=1
    )
    max_sessions_per_user: int = Field(
        default=10,
        description="Maximum sessions per user",
        ge=1
    )
    enable_persistence: bool = Field(
        default=True,
        description="Whether to persist sessions"
    )
    storage_backend: str = Field(
        default="memory",
        description="Storage backend: memory, redis, database"
    )
    cleanup_interval_minutes: int = Field(
        default=60,
        description="Cleanup interval for expired sessions",
        ge=1
    )


class MemoryEntry(BaseModel):
    """Memory entry for storing conversation and context information."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    memory_id: str = Field(
        default_factory=lambda: str(uuid_lib.uuid4()),
        description="Unique memory identifier"
    )
    session_id: str = Field(..., description="Associated session ID")
    session_type: SessionType = Field(..., description="Session type")
    agent_id: Optional[str] = Field(None, description="Associated agent ID")

    # Memory content
    content: str = Field(..., description="Memory content", min_length=1)
    memory_type: str = Field(
        ...,
        description="Memory type: conversation, fact, preference, context"
    )
    importance: float = Field(
        default=0.5,
        description="Importance score 0-1",
        ge=0.0,
        le=1.0
    )

    # Time information
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Memory creation timestamp"
    )
    last_accessed: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last access timestamp"
    )
    access_count: int = Field(default=0, description="Access count", ge=0)

    # Metadata
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def update_access(self) -> None:
        """Update access information."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

    def is_important(self, threshold: float = 0.7) -> bool:
        """Check if memory is considered important."""
        return self.importance >= threshold


class MemoryConfig(BaseModel):
    """Configuration for memory management."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    max_memories_per_session: int = Field(
        default=1000,
        description="Maximum memories per session",
        ge=1
    )
    memory_retention_days: int = Field(
        default=30,
        description="Memory retention period in days",
        ge=1
    )
    enable_semantic_search: bool = Field(
        default=True,
        description="Enable semantic similarity search"
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Similarity threshold for search",
        ge=0.0,
        le=1.0
    )
    storage_backend: str = Field(
        default="memory",
        description="Storage backend: memory, redis, database, vector_db"
    )
    cleanup_interval_hours: int = Field(
        default=24,
        description="Cleanup interval for old memories",
        ge=1
    )
    importance_decay_rate: float = Field(
        default=0.1,
        description="Rate at which importance decays over time",
        ge=0.0,
        le=1.0
    )



class MCPTool(BaseModel):
    """Represents an MCP tool definition."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    name: str = Field(..., description="Tool name", min_length=1, max_length=100)
    title: Optional[str] = Field(None, description="Human-readable title", max_length=200)
    description: str = Field(..., description="Tool description", min_length=1, max_length=1000)

    # Schema definitions
    input_schema: Dict[str, Any] = Field(..., description="JSON Schema for input parameters")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for output")

    # Tool metadata
    server_id: str = Field(..., description="MCP server providing this tool")
    annotations: Dict[str, Any] = Field(default_factory=dict, description="Tool annotations")

    # Security and permissions
    requires_confirmation: bool = Field(default=True, description="Whether tool requires user confirmation")
    allowed_contexts: List[str] = Field(default_factory=list, description="Allowed execution contexts")

    # Caching and performance
    cacheable: bool = Field(default=False, description="Whether results can be cached")
    cache_ttl_seconds: Optional[int] = Field(None, description="Cache TTL in seconds", ge=0)

    # Metadata
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Tool discovery time"
    )
    last_used: Optional[datetime] = Field(None, description="Last usage time")
    usage_count: int = Field(default=0, description="Usage count", ge=0)


class MCPToolCallRequest(BaseModel):
    """Request to call an MCP tool."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    request_id: str = Field(
        default_factory=lambda: str(uuid_lib.uuid4()),
        description="Unique request identifier"
    )
    tool_name: str = Field(..., description="Tool name to call", min_length=1)
    server_id: str = Field(..., description="MCP server ID", min_length=1)

    # Call parameters
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

    # Execution context
    agent_id: Optional[str] = Field(None, description="Calling agent ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    user_id: Optional[str] = Field(None, description="User ID")

    # Security and permissions
    user_approved: bool = Field(default=False, description="Whether user approved the call")
    approval_timestamp: Optional[datetime] = Field(None, description="Approval timestamp")

    # Request metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request creation time"
    )
    timeout_seconds: int = Field(default=30, description="Request timeout", ge=1, le=300)


class MCPToolCallResult(BaseModel):
    """Result of an MCP tool call."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    request_id: str = Field(..., description="Original request ID")
    tool_name: str = Field(..., description="Tool name that was called")
    server_id: str = Field(..., description="MCP server ID")

    # Execution result
    success: bool = Field(..., description="Whether call was successful")
    content: List[Content] = Field(default_factory=list[Content], description="Tool result content")
    text: Optional[str] = Field(None, description="Extracted text content")
    is_error: bool = Field(default=False, description="Whether result is an error")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Execution metrics
    execution_time_ms: Optional[int] = Field(None, description="Execution time", ge=0)
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")

    # Resource usage
    tokens_used: Optional[int] = Field(None, description="Tokens used", ge=0)
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage", ge=0)

    def is_successful(self) -> bool:
        """Check if tool call was successful."""
        return self.success and not self.is_error and self.error_message is None


# Update forward references
MultiAgentConfig.model_rebuild()
WorkflowConfig.model_rebuild()
AgentExecutionResult.model_rebuild()
