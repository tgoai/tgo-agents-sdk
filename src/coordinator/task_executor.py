"""
Task executor implementation.

This module provides the task execution logic for individual agents.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from src.registry.adapter_registry import AdapterRegistry

from ..core.interfaces import TaskExecutor as ITaskExecutor
from ..core.models import Task, AgentInstance, ExecutionContext, TaskResult
from ..core.exceptions import TaskExecutionError
from ..registry import get_registry

logger = logging.getLogger(__name__)


class TaskExecutor(ITaskExecutor):
    """Task executor for individual agent task execution."""
    
    def __init__(self, registry: AdapterRegistry):
        self._registry = registry
        logger.info("TaskExecutor initialized")
    
    async def execute(
        self, 
        task: Task, 
        agent: AgentInstance, 
        context: ExecutionContext
    ) -> TaskResult:
        """Execute a single task with an agent.
        
        Args:
            task: Task to execute
            agent: Agent instance to execute the task
            context: Execution context
            
        Returns:
            Task execution result
            
        Raises:
            TaskExecutionError: If execution fails
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Executing task {task.task_id} with agent {agent.agent_id}")
            
            # Get framework adapter
            framework_name = context.framework_name or "default"
            adapter = self._registry.get_adapter(framework_name)
            if not adapter:
                # Try to get default adapter as fallback
                adapter = self._registry.get_default_adapter()
                if not adapter:
                    raise TaskExecutionError(f"No framework adapter available (requested: {framework_name})")
            
            # Execute task through adapter
            execution_result = await adapter.execute_task(agent.agent_id, task, context)
            
            # Convert to TaskResult
            end_time = datetime.now(timezone.utc)
            
            result = TaskResult(
                task_id=task.task_id,
                success=execution_result.success,
                result=execution_result.result,
                error_message=execution_result.error_message,
                execution_time_ms=execution_result.execution_time_ms,
                started_at=start_time,
                completed_at=end_time,
                agent_id=agent.agent_id,
                agent_type=agent.config.agent_type
            )
            
            logger.info(f"Task {task.task_id} execution completed: {result.success}")
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {task.task_id}: {e}")
            end_time = datetime.now(timezone.utc)
            
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                started_at=start_time,
                completed_at=end_time,
                agent_id=agent.agent_id,
                agent_type=agent.config.agent_type
            )
