"""
Enumerations for the multi-agent system.

This module defines all the enums used throughout the multi-agent system,
providing type safety and clear definitions for various states and types.
"""

from enum import Enum, IntEnum


class AgentType(str, Enum):
    """Agent type enumeration.
    
    Defines the different types of agents supported by the system:
    - MANAGER: Manages and coordinates other agents
    - EXPERT: Specialized agent for specific domains
    - WORKFLOW: Orchestrates complex workflows
    - CUSTOM: User-defined custom agent type
    """
    MANAGER = "manager"
    EXPERT = "expert"
    WORKFLOW = "workflow"
    CUSTOM = "custom"

    def __str__(self) -> str:
        return self.value


class AgentStatus(str, Enum):
    """Agent status enumeration.
    
    Defines the runtime status of agents:
    - IDLE: Available for new tasks
    - BUSY: Currently executing a task
    - ERROR: In error state, needs intervention
    - OFFLINE: Not available
    """
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

    def __str__(self) -> str:
        return self.value


class TaskType(str, Enum):
    """Task type enumeration.
    
    Defines different types of tasks:
    - SIMPLE: Single-step task for one agent
    - COMPLEX: Multi-step task requiring coordination
    - PIPELINE: Sequential processing task
    - PARALLEL: Parallel processing task
    - BATCH: Batch processing task
    """
    SIMPLE = "simple"
    COMPLEX = "complex"
    PIPELINE = "pipeline"
    PARALLEL = "parallel"
    BATCH = "batch"

    def __str__(self) -> str:
        return self.value


class TaskStatus(str, Enum):
    """Task status enumeration.
    
    Defines the execution status of tasks:
    - PENDING: Waiting to be executed
    - ASSIGNED: Assigned to agent(s)
    - RUNNING: Currently being executed
    - COMPLETED: Successfully completed
    - FAILED: Failed during execution
    - CANCELLED: Cancelled before completion
    - TIMEOUT: Timed out during execution
    """
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

    def __str__(self) -> str:
        return self.value


class TaskPriority(IntEnum):
    """Task priority enumeration.
    
    Defines task execution priorities (higher number = higher priority):
    - LOW: Low priority tasks
    - MEDIUM: Medium priority tasks
    - HIGH: High priority tasks
    - CRITICAL: Critical priority tasks
    """
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class WorkflowType(str, Enum):
    """Workflow type enumeration.
    
    Defines different workflow execution patterns:
    - SINGLE: Single agent execution
    - HIERARCHICAL: Manager-expert hierarchical execution
    - SEQUENTIAL: Sequential agent execution
    - PARALLEL: Parallel agent execution
    - CUSTOM: Custom workflow definition
    - STREAMING: Real-time streaming execution
    - BATCH: Batch processing workflow
    """
    SINGLE = "single"
    HIERARCHICAL = "hierarchical"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CUSTOM = "custom"
    STREAMING = "streaming"
    BATCH = "batch"

    def __str__(self) -> str:
        return self.value


class ExecutionStrategy(str, Enum):
    """Execution strategy enumeration.
    
    Defines how tasks should be executed when failures occur:
    - FAIL_FAST: Stop immediately on first failure
    - CONTINUE_ON_FAILURE: Continue execution despite failures
    - RETRY_ON_FAILURE: Retry failed tasks
    - BEST_EFFORT: Complete as many tasks as possible
    - SECURE: Execute with security constraints
    - MONITORED: Execute with detailed monitoring
    """
    FAIL_FAST = "fail_fast"
    CONTINUE_ON_FAILURE = "continue_on_failure"
    RETRY_ON_FAILURE = "retry_on_failure"
    BEST_EFFORT = "best_effort"
    SECURE = "secure"
    MONITORED = "monitored"

    def __str__(self) -> str:
        return self.value


class SessionType(str, Enum):
    """Session type enumeration.

    Defines different types of user sessions:
    - SINGLE_CHAT: Single user chat session (1)
    - GROUP_CHAT: Group chat session (2)
    """
    SINGLE_CHAT = "1"
    GROUP_CHAT = "2"

    def __str__(self) -> str:
        return self.value


class FrameworkCapability(str, Enum):
    """Framework capability enumeration.

    Defines capabilities that different AI frameworks may support:
    - SINGLE_AGENT: Single agent execution
    - MULTI_AGENT: Multi-agent coordination
    - STREAMING: Streaming execution
    - BATCH_PROCESSING: Batch processing
    - TOOL_CALLING: Tool integration
    - KNOWLEDGE_BASE: Knowledge base integration
    - MEMORY: Agent memory/state management
    - MONITORING: Execution monitoring
    - FAULT_TOLERANCE: Fault tolerance and recovery
    """
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    STREAMING = "streaming"
    BATCH_PROCESSING = "batch_processing"
    TOOL_CALLING = "tool_calling"
    KNOWLEDGE_BASE = "knowledge_base"
    MEMORY = "memory"
    MONITORING = "monitoring"
    FAULT_TOLERANCE = "fault_tolerance"

    def __str__(self) -> str:
        return self.value
