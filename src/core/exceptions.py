"""
Exception classes for the multi-agent system.

This module defines a comprehensive hierarchy of exceptions used throughout
the multi-agent system, providing clear error handling and debugging information.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timezone


class MultiAgentError(Exception):
    """Base exception for all multi-agent system errors.
    
    This is the root exception class that all other exceptions inherit from.
    It provides common functionality for error tracking and debugging.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc)
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


# Framework-related exceptions
class FrameworkError(MultiAgentError):
    """Base exception for framework-related errors."""
    
    def __init__(
        self, 
        message: str, 
        framework_name: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, context)
        self.framework_name = framework_name


class FrameworkNotFoundError(FrameworkError):
    """Exception raised when a requested framework is not found."""
    pass


class FrameworkInitializationError(FrameworkError):
    """Exception raised when framework initialization fails."""
    pass


class FrameworkUnavailableError(FrameworkError):
    """Exception raised when a framework is temporarily unavailable."""
    pass


# Agent-related exceptions
class AgentError(MultiAgentError):
    """Base exception for agent-related errors."""
    
    def __init__(
        self, 
        message: str, 
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, context)
        self.agent_id = agent_id
        self.agent_type = agent_type


class AgentNotFoundError(AgentError):
    """Exception raised when a requested agent is not found."""
    pass


class AgentCreationError(AgentError):
    """Exception raised when agent creation fails."""
    pass


class AgentExecutionError(AgentError):
    """Exception raised when agent execution fails."""
    pass


# Task-related exceptions
class TaskError(MultiAgentError):
    """Base exception for task-related errors."""
    
    def __init__(
        self, 
        message: str, 
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, context)
        self.task_id = task_id
        self.task_type = task_type


class TaskExecutionError(TaskError):
    """Exception raised when task execution fails."""
    pass


class TaskTimeoutError(TaskError):
    """Exception raised when task execution times out."""
    pass


class TaskValidationError(TaskError):
    """Exception raised when task validation fails."""
    pass


# Workflow-related exceptions
class WorkflowError(MultiAgentError):
    """Base exception for workflow-related errors."""
    
    def __init__(
        self, 
        message: str, 
        workflow_type: Optional[str] = None,
        workflow_id: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, context)
        self.workflow_type = workflow_type
        self.workflow_id = workflow_id


class WorkflowExecutionError(WorkflowError):
    """Exception raised when workflow execution fails."""
    pass


class WorkflowConfigurationError(WorkflowError):
    """Exception raised when workflow configuration is invalid."""
    pass
