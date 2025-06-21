"""
Base workflow implementation.

This module provides the base class for all workflow implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from ..core.models import (
    Task, AgentInstance, WorkflowConfig, MultiAgentResult,
    ExecutionContext, TaskResult, AgentExecutionResult
)
from ..core.enums import WorkflowType, ExecutionStrategy
from ..core.exceptions import WorkflowError

logger = logging.getLogger(__name__)


class BaseWorkflow(ABC):
    """Base class for all workflow implementations.
    
    This class provides common functionality and defines the interface
    that all workflow implementations must follow.
    """
    
    def __init__(self, workflow_type: WorkflowType):
        self.workflow_type = workflow_type
        self._logger = logging.getLogger(f"{__name__}.{workflow_type.value}")
    
    @abstractmethod
    async def execute(
        self,
        config: WorkflowConfig,
        task: Task,
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute the workflow.
        
        Args:
            config: Workflow configuration
            task: Task to execute
            agents: List of agent instances
            context: Execution context
            
        Returns:
            Multi-agent execution result
            
        Raises:
            WorkflowExecutionError: If execution fails
        """
        pass
    
    def validate_agents(self, agents: List[AgentInstance], config: WorkflowConfig) -> None:
        """Validate that agents are suitable for this workflow.
        
        Args:
            agents: List of agent instances
            config: Workflow configuration
            
        Raises:
            WorkflowError: If agents are not suitable
        """
        if not agents:
            raise WorkflowError(f"No agents provided for {self.workflow_type} workflow")
    
    def create_execution_result(
        self,
        success: bool,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        agent_results: Optional[Dict[str, AgentExecutionResult]] = None,
        task_results: Optional[Dict[str, TaskResult]] = None,
        agents_used: Optional[List[str]] = None
    ) -> MultiAgentResult:
        """Create a standardized execution result.
        
        Args:
            success: Whether execution was successful
            result: Execution result data
            error_message: Error message if failed
            agent_results: Results from individual agents
            task_results: Results from individual tasks
            agents_used: List of agent IDs that participated
            
        Returns:
            Multi-agent execution result
        """
        now = datetime.now(timezone.utc)
        return MultiAgentResult(
            success=success,
            result=result,
            error_message=error_message,
            agent_results=agent_results or {},
            task_results=task_results or {},
            total_execution_time_ms=0,
            started_at=now,
            completed_at=now,
            workflow_type=self.workflow_type,
            agents_used=agents_used or []
        )
    
    def should_continue_on_failure(self, strategy: ExecutionStrategy) -> bool:
        """Check if execution should continue when an agent fails.
        
        Args:
            strategy: Execution strategy
            
        Returns:
            True if should continue, False if should stop
        """
        return strategy in [
            ExecutionStrategy.CONTINUE_ON_FAILURE,
            ExecutionStrategy.BEST_EFFORT,
            ExecutionStrategy.RETRY_ON_FAILURE
        ]
    
    def log_workflow_start(self, task: Task, agents: List[AgentInstance]) -> None:
        """Log workflow execution start."""
        self._logger.info(
            f"Starting {self.workflow_type} workflow for task {task.task_id} "
            f"with {len(agents)} agents"
        )
    
    def log_workflow_complete(self, result: MultiAgentResult) -> None:
        """Log workflow execution completion."""
        self._logger.info(
            f"Completed {self.workflow_type} workflow: "
            f"success={result.success}, agents_used={len(result.agents_used)}"
        )
    
    def log_agent_execution(self, agent_id: str, success: bool, error: Optional[str] = None) -> None:
        """Log individual agent execution result."""
        if success:
            self._logger.info(f"Agent {agent_id} executed successfully")
        else:
            self._logger.warning(f"Agent {agent_id} failed: {error}")
    
    async def handle_execution_error(
        self,
        error: Exception,
        task: Task,
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Handle execution errors and create appropriate result.
        
        Args:
            error: The exception that occurred
            task: Task that failed
            agents: Agents involved in execution
            context: Execution context
            
        Returns:
            MultiAgentResult with error information
        """
        # Note: task and context parameters available for future use
        error_message = str(error)
        self._logger.error(f"{self.workflow_type} workflow execution failed: {error_message}")
        
        return self.create_execution_result(
            success=False,
            error_message=error_message,
            agents_used=[agent.agent_id for agent in agents]
        )
