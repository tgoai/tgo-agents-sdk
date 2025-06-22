"""
Multi-agent coordinator implementation.

This module provides the main coordination logic for executing tasks
across multiple agents and different AI frameworks.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timezone
import uuid as uuid_lib

from ..core.interfaces import (
    MultiAgentCoordinator as IMultiAgentCoordinator,
    BaseFrameworkAdapter, SessionManager, MemoryManager
)
from ..core.models import (
    Task, AgentConfig, AgentInstance, MultiAgentConfig,
    WorkflowConfig, TaskResult, MultiAgentResult, ExecutionContext,
    ExecutionMetrics, Session
)

from ..core.exceptions import (
    MultiAgentError, FrameworkNotFoundError, AgentCreationError,
    WorkflowExecutionError
)

from ..tools.fastmcp_tool_manager import MCPToolManager

from ..registry import get_registry, AdapterRegistry
from .workflow_engine import WorkflowEngine

from .result_aggregator import ResultAggregator

logger = logging.getLogger(__name__)


class MultiAgentCoordinator(IMultiAgentCoordinator):
    """Multi-agent system coordinator.
    
    This class orchestrates the execution of tasks across multiple agents
    and different AI frameworks, providing:
    - Framework-agnostic task execution
    - Dynamic framework switching
    - Multi-agent workflow coordination
    - Result aggregation and synthesis
    - Error handling and recovery
    - Streaming and batch processing
    """
    
    def __init__(
        self,
        registry: Optional['AdapterRegistry'] = None,
        session_manager: Optional[SessionManager] = None,
        memory_manager: Optional[MemoryManager] = None,
        mcp_tool_manager: Optional[MCPToolManager] = None
    ):
        self._registry = registry or get_registry()
        self._workflow_engine = WorkflowEngine(self._registry)
        self._result_aggregator = ResultAggregator()

        # Session and Memory managers (optional)
        self._session_manager = session_manager
        self._memory_manager = memory_manager

        # MCP tool manager (optional)
        self._mcp_tool_manager = mcp_tool_manager

        # Execution tracking
        self._active_executions: Dict[str, Dict[str, Any]] = {}
        self._execution_metrics: Dict[str, ExecutionMetrics] = {}

        # Configuration
        self._max_concurrent_executions = 10
        self._default_timeout = 300  # 5 minutes

        logger.info("MultiAgentCoordinator initialized")
    
    async def execute_task(
        self,
        config: MultiAgentConfig,
        task: Task,
        session: Optional[Session] = None
    ) -> MultiAgentResult:
        """Execute a task using the multi-agent system.
        
        Args:
            config: Multi-agent configuration
            task: Task to execute
            
        Returns:
            Multi-agent execution result
            
        Raises:
            MultiAgentError: If execution fails
        """
        execution_id = str(uuid_lib.uuid4())
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting multi-agent task execution: {execution_id}")
        
        try:
            # Validate configuration
            await self._validate_config(config)
            
            # Create execution context
            context = ExecutionContext(
                execution_id=execution_id,
                task_id=task.task_id,
                framework_name=config.framework,
                workflow_type=config.workflow.workflow_type,
                created_at=start_time,
                session_id=session.session_id if session else None,
                user_id=session.user_id if session else None,
                agent_id=None  # Will be set by individual agents during execution
            )
            
            # Track execution
            self._active_executions[execution_id] = {
                "config": config,
                "task": task,
                "context": context,
                "start_time": start_time,
                "status": "running"
            }
            
            # Get framework adapter
            adapter = await self._get_framework_adapter(config.framework)
            # Create agents
            agents = await self._create_agents(config.agents, adapter,context)
            
            # Execute workflow
            result = await self._execute_workflow(
                config.workflow, task, agents, context
            )
            
            # Update execution tracking
            self._active_executions[execution_id]["status"] = "completed"
            self._active_executions[execution_id]["result"] = result
            
            logger.info(f"Completed multi-agent task execution: {execution_id}")
            return result
            
        except Exception as e:
            logger.error(f"Multi-agent execution failed: {execution_id}: {e}")
            
            # Update execution tracking
            if execution_id in self._active_executions:
                self._active_executions[execution_id]["status"] = "failed"
                self._active_executions[execution_id]["error"] = str(e)
            
            # Handle error with fallback if configured
            if config.fallback_frameworks:
                return await self._handle_execution_error_with_fallback(
                    config, task, e, execution_id
                )
            
            raise MultiAgentError(f"Multi-agent execution failed: {e}")
        
        finally:
            # Clean up execution tracking after some time
            asyncio.create_task(self._cleanup_execution_tracking(execution_id, delay=3600))
    
    async def execute_task_stream( # type: ignore
        self, 
        config: MultiAgentConfig, 
        task: Task
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a task with streaming results.
        
        Args:
            config: Multi-agent configuration
            task: Task to execute
            
        Yields:
            Streaming execution updates
        """
        execution_id = str(uuid_lib.uuid4())
        
        try:
            yield {"type": "execution_started", "execution_id": execution_id}
            
            # Validate configuration
            await self._validate_config(config)
            yield {"type": "config_validated"}
            
            # Get framework adapter
            adapter = await self._get_framework_adapter(config.framework)
            yield {"type": "framework_ready", "framework": config.framework}
            
            # Create agents
            agents = await self._create_agents(config.agents, adapter)
            yield {"type": "agents_created", "agent_count": len(agents)}
            
            # Execute workflow with streaming
            async for update in self._execute_workflow_stream(
                config.workflow, task, agents, execution_id
            ):
                yield update
            
            yield {"type": "execution_completed", "execution_id": execution_id}
            
        except Exception as e:
            logger.error(f"Streaming execution failed: {execution_id}: {e}")
            yield {
                "type": "execution_error", 
                "execution_id": execution_id,
                "error": str(e)
            }
    
    async def execute_batch_tasks(
        self, 
        config: MultiAgentConfig, 
        tasks: List[Task]
    ) -> List[MultiAgentResult]:
        """Execute multiple tasks in batch.
        
        Args:
            config: Multi-agent configuration
            tasks: List of tasks to execute
            
        Returns:
            List of execution results
        """
        logger.info(f"Starting batch execution of {len(tasks)} tasks")
        
        # Execute tasks concurrently with limit
        semaphore = asyncio.Semaphore(self._max_concurrent_executions)
        
        async def execute_single_task(task: Task) -> MultiAgentResult:
            async with semaphore:
                return await self.execute_task(config, task)
        
        # Create tasks for concurrent execution
        execution_tasks = [execute_single_task(task) for task in tasks]
        
        # Execute all tasks
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results: List[MultiAgentResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch task {i} failed: {result}")
                now = datetime.now(timezone.utc)
                processed_results.append(MultiAgentResult(
                    success=False,
                    result=None,
                    error_message=str(result),
                    agent_results={},
                    task_results={tasks[i].task_id: TaskResult(
                        task_id=tasks[i].task_id,
                        success=False,
                        result=None,
                        error_message=str(result),
                        execution_time_ms=0,
                        started_at=now,
                        completed_at=now,
                        agent_id=None,
                        agent_type=None
                    )},
                    total_execution_time_ms=0,
                    started_at=now,
                    completed_at=now,
                    workflow_type=None,
                    agents_used=[]
                ))
            else:
                # result is MultiAgentResult in this case
                if isinstance(result, MultiAgentResult):
                    processed_results.append(result)
        
        logger.info(f"Completed batch execution of {len(tasks)} tasks")
        return processed_results
    
    async def _validate_config(self, config: MultiAgentConfig) -> None:
        """Validate multi-agent configuration."""
        if not config.framework:
            raise MultiAgentError("Framework not specified in configuration")
        
        if not config.agents:
            raise MultiAgentError("No agents specified in configuration")
        
        if not config.workflow:
            raise MultiAgentError("Workflow configuration not specified")
        
        # Validate framework is registered
        if not self._registry.is_registered(config.framework):
            raise FrameworkNotFoundError(f"Framework not registered: {config.framework}")
    
    async def _get_framework_adapter(self, framework_name: str):
        """Get and initialize framework adapter."""
        adapter = self._registry.get_adapter(framework_name)
        if not adapter:
            raise FrameworkNotFoundError(f"Framework adapter not found: {framework_name}")

        if not adapter.is_initialized:
            await self._registry.initialize_adapter(framework_name)

        # Inject memory manager
        adapter.set_memory_manager(self._memory_manager)
        logger.debug(f"Injected memory manager into {framework_name} adapter")
            
        # Inject mcp tool manager
        adapter.set_mcp_tool_manager(self._mcp_tool_manager)
        logger.debug(f"Injected MCP tool manager into {framework_name} adapter")
            

        return adapter
    
    async def _create_agents(
        self,
        agent_configs: List[AgentConfig],
        adapter: 'BaseFrameworkAdapter',
        context: ExecutionContext
    ) -> List[AgentInstance]:
        """Create agent instances using the framework adapter."""
        agents: List[AgentInstance] = []
        
        for config in agent_configs:
            try:
                agent = await adapter.create_agent(config,context)
                agents.append(agent)
                logger.info(f"Created agent: {config.agent_id}")
            except Exception as e:
                logger.error(f"Failed to create agent {config.agent_id}: {e}")
                raise AgentCreationError(f"Failed to create agent {config.agent_id}: {e}")
        
        return agents
    
    async def _execute_workflow(
        self, 
        workflow_config: WorkflowConfig,
        task: Task, 
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute workflow using the workflow engine."""
        try:
            return await self._workflow_engine.execute_workflow(
                workflow_config.workflow_type,
                workflow_config,
                task,
                agents,
                context
            )
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise WorkflowExecutionError(f"Workflow execution failed: {e}")
    
    async def _execute_workflow_stream(
        self, 
        workflow_config: WorkflowConfig,
        task: Task, 
        agents: List[AgentInstance],
        execution_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute workflow with streaming updates."""
        # Create execution context
        context = ExecutionContext(
            execution_id=execution_id,
            task_id=task.task_id,
            framework_name="unknown",  # Framework name not available in this context
            workflow_type=workflow_config.workflow_type,
            created_at=datetime.now(timezone.utc),
            session_id=None,
            user_id=None,
            agent_id=None
        )
        
        # Stream workflow execution
        async for update in self._workflow_engine.execute_workflow_stream(
            workflow_config.workflow_type,
            workflow_config,
            task,
            agents,
            context
        ):
            yield update
    
    async def _handle_execution_error_with_fallback(
        self, 
        config: MultiAgentConfig,
        task: Task, 
        original_error: Exception,
        execution_id: str
    ) -> MultiAgentResult:
        """Handle execution error with fallback frameworks."""
        logger.info(f"Attempting fallback execution for: {execution_id}")
        
        for fallback_framework in config.fallback_frameworks:
            try:
                logger.info(f"Trying fallback framework: {fallback_framework}")
                
                # Create new config with fallback framework
                fallback_config = config.model_copy()
                fallback_config.framework = fallback_framework
                
                # Attempt execution with fallback
                result = await self.execute_task(fallback_config, task)
                
                logger.info(f"Fallback execution successful: {fallback_framework}")
                return result
                
            except Exception as e:
                logger.warning(f"Fallback framework {fallback_framework} also failed: {e}")
                continue
        
        # All fallbacks failed
        raise MultiAgentError(
            f"All frameworks failed. Original error: {original_error}. "
            f"Fallback frameworks: {config.fallback_frameworks}"
        )
    
    async def _cleanup_execution_tracking(self, execution_id: str, delay: int = 3600):
        """Clean up execution tracking after delay."""
        await asyncio.sleep(delay)
        self._active_executions.pop(execution_id, None)
        self._execution_metrics.pop(execution_id, None)
    
    def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active executions."""
        return self._active_executions.copy()
    
    def get_execution_metrics(self, execution_id: str) -> Optional[ExecutionMetrics]:
        """Get metrics for a specific execution."""
        return self._execution_metrics.get(execution_id)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        if execution_id not in self._active_executions:
            return False
        
        try:
            # Mark as cancelled
            self._active_executions[execution_id]["status"] = "cancelled"
            
            # TODO: Implement actual cancellation logic
            # This would involve stopping the workflow engine and cleaning up agents
            
            logger.info(f"Cancelled execution: {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel execution {execution_id}: {e}")
            return False
