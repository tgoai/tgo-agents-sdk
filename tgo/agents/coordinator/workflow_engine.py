"""
Workflow engine implementation.

This module provides the workflow execution engine that supports
different workflow types and execution strategies.
"""

import logging
import asyncio
from typing import Dict, Any, List, AsyncGenerator
from datetime import datetime, timezone

from ..registry.adapter_registry import AdapterRegistry

from ..core.interfaces import WorkflowEngine as IWorkflowEngine
from ..core.models import (
    Task, AgentInstance, WorkflowConfig, MultiAgentResult,
    ExecutionContext, AgentExecutionResult, TaskResult
)
from ..core.enums import WorkflowType, ExecutionStrategy, AgentType
from ..core.exceptions import WorkflowError, WorkflowExecutionError
from .task_executor import TaskExecutor
from .result_aggregator import ResultAggregator

logger = logging.getLogger(__name__)


class WorkflowEngine(IWorkflowEngine):
    """Workflow execution engine.
    
    This class handles the execution of different workflow types:
    - Single agent execution
    - Hierarchical (manager-expert) workflows
    - Sequential (pipeline) workflows
    - Parallel workflows
    - Custom workflows
    - Streaming workflows
    - Batch workflows
    """
    
    def __init__(self, registry: AdapterRegistry):
        self._task_executor = TaskExecutor(registry)
        self._result_aggregator = ResultAggregator()
        
        # Workflow type handlers
        self._workflow_handlers = {
            WorkflowType.SINGLE: self._execute_single_workflow,
            WorkflowType.HIERARCHICAL: self._execute_hierarchical_workflow,
            WorkflowType.SEQUENTIAL: self._execute_sequential_workflow,
            WorkflowType.PARALLEL: self._execute_parallel_workflow,
            WorkflowType.CUSTOM: self._execute_custom_workflow,
            WorkflowType.STREAMING: self._execute_streaming_workflow,
            WorkflowType.BATCH: self._execute_batch_workflow,
        }
        
        logger.info("WorkflowEngine initialized")
    
    def supports_workflow_type(self, workflow_type: WorkflowType) -> bool:
        """Check if workflow type is supported."""
        return workflow_type in self._workflow_handlers
    
    async def execute_workflow(
        self, 
        workflow_type: WorkflowType,
        config: WorkflowConfig, 
        task: Task, 
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute a workflow.
        
        Args:
            workflow_type: Type of workflow to execute
            config: Workflow configuration
            task: Task to execute
            agents: List of agent instances
            context: Execution context
            
        Returns:
            Multi-agent execution result
            
        Raises:
            WorkflowError: If workflow type is not supported
            WorkflowExecutionError: If execution fails
        """
        if not self.supports_workflow_type(workflow_type):
            raise WorkflowError(f"Unsupported workflow type: {workflow_type}")
        
        logger.info(f"Executing {workflow_type} workflow with {len(agents)} agents")
        
        try:
            handler = self._workflow_handlers[workflow_type]
            result = await handler(config, task, agents, context)
            
            logger.info(f"Workflow execution completed: {workflow_type}")
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {workflow_type}: {e}")
            raise WorkflowExecutionError(f"Workflow execution failed: {e}")
    
    async def execute_workflow_stream(
        self, 
        workflow_type: WorkflowType,
        config: WorkflowConfig, 
        task: Task, 
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute workflow with streaming updates."""
        yield {"type": "workflow_started", "workflow_type": workflow_type.value}
        
        try:
            if workflow_type == WorkflowType.STREAMING:
                async for update in self._execute_streaming_workflow_stream(
                    config, task, agents, context
                ):
                    yield update
            else:
                # For non-streaming workflows, execute normally and yield final result
                result = await self.execute_workflow(workflow_type, config, task, agents, context)
                yield {"type": "workflow_result", "result": result.model_dump()}
            
            yield {"type": "workflow_completed", "workflow_type": workflow_type.value}
            
        except Exception as e:
            yield {"type": "workflow_error", "error": str(e)}
    
    async def _execute_single_workflow(
        self, 
        config: WorkflowConfig,
        task: Task, 
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute single agent workflow."""
        if not agents:
            raise WorkflowExecutionError("No agents provided for single workflow")
        
        # Use the first agent
        agent = agents[0]
        
        # Execute task
        result = await self._task_executor.execute(task, agent, context)
        
        # Create multi-agent result
        now = datetime.now(timezone.utc)
        return MultiAgentResult(
            success=result.success,
            result=result.result,
            error_message=result.error_message,
            agent_results={agent.agent_id: AgentExecutionResult(
                success=result.success,
                result=result.result,
                error_message=result.error_message,
                execution_time_ms=result.execution_time_ms or 0,
                started_at=result.started_at or now,
                completed_at=result.completed_at or now
            )},
            task_results={task.task_id: result},
            total_execution_time_ms=result.execution_time_ms or 0,
            started_at=result.started_at or now,
            completed_at=result.completed_at or now,
            workflow_type=WorkflowType.SINGLE,
            agents_used=[agent.agent_id]
        )
    
    async def _execute_hierarchical_workflow(
        self, 
        config: WorkflowConfig,
        task: Task, 
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute hierarchical (manager-expert) workflow."""
        # Find manager and expert agents
        manager_agent = None
        expert_agents: List[AgentInstance] = []
        
        for agent in agents:
            if agent.config.agent_type == AgentType.MANAGER:
                if config.manager_agent_id and agent.agent_id == config.manager_agent_id:
                    manager_agent = agent
                elif not config.manager_agent_id and not manager_agent:
                    manager_agent = agent
            elif agent.config.agent_type == AgentType.EXPERT:
                if not config.expert_agent_ids or agent.agent_id in config.expert_agent_ids:
                    expert_agents.append(agent)
        
        if not manager_agent:
            raise WorkflowExecutionError("No manager agent found for hierarchical workflow")
        
        if not expert_agents:
            raise WorkflowExecutionError("No expert agents found for hierarchical workflow")
        
        logger.info(f"Hierarchical workflow: 1 manager, {len(expert_agents)} experts")
        
        # Phase 1: Manager analyzes and decomposes task
        now = datetime.now(timezone.utc)
        decomposition_task = Task(
            title=f"Analyze and decompose: {task.title}",
            description=f"Analyze the following task and break it down into subtasks for expert agents:\n{task.description}",
            input_data={
                "original_task": task.model_dump(),
                "available_experts": [
                    {"agent_id": agent.agent_id, "capabilities": agent.config.capabilities}
                    for agent in expert_agents
                ]
            },
            output_data=None,
            timeout_seconds=None,
            started_at=now,
            completed_at=None,
            parent_task_id=task.task_id
        )
        
        manager_result = await self._task_executor.execute(decomposition_task, manager_agent, context)
        
        if not manager_result.success:
            now = datetime.now(timezone.utc)
            return MultiAgentResult(
                success=False,
                result=None,
                error_message=f"Manager task decomposition failed: {manager_result.error_message}",
                agent_results={manager_agent.agent_id: AgentExecutionResult(
                    success=False,
                    result=None,
                    error_message=manager_result.error_message,
                    execution_time_ms=0,
                    started_at=now,
                    completed_at=now
                )},
                task_results={},
                total_execution_time_ms=0,
                started_at=now,
                completed_at=now,
                workflow_type=WorkflowType.HIERARCHICAL,
                agents_used=[manager_agent.agent_id]
            )
        
        # Phase 2: Execute subtasks with expert agents
        expert_results: Dict[str, AgentExecutionResult] = {}

        if config.execution_strategy == ExecutionStrategy.FAIL_FAST:
            # Execute experts sequentially, stop on first failure
            for expert in expert_agents:
                now = datetime.now(timezone.utc)
                expert_task = Task(
                    title=f"Expert task for {expert.agent_id}",
                    description=f"Execute your specialized part of the task: {task.description}",
                    input_data=task.input_data,
                    output_data=None,
                    timeout_seconds=None,
                    started_at=now,
                    completed_at=None,
                    parent_task_id=task.task_id
                )

                result = await self._task_executor.execute(expert_task, expert, context)
                expert_results[expert.agent_id] = AgentExecutionResult(
                    success=result.success,
                    result=result.result,
                    error_message=result.error_message,
                    execution_time_ms=result.execution_time_ms or 0,
                    started_at=result.started_at or now,
                    completed_at=result.completed_at or now
                )

                if not result.success:
                    break
        else:
            # Execute experts in parallel
            expert_tasks: List[Any] = []
            for expert in expert_agents:
                now = datetime.now(timezone.utc)
                expert_task = Task(
                    title=f"Expert task for {expert.agent_id}",
                    description=f"Execute your specialized part of the task: {task.description}",
                    input_data=task.input_data,
                    output_data=None,
                    timeout_seconds=None,
                    started_at=now,
                    completed_at=None,
                    parent_task_id=task.task_id
                )
                expert_tasks.append(self._task_executor.execute(expert_task, expert, context))

            results = await asyncio.gather(*expert_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                expert = expert_agents[i]
                now = datetime.now(timezone.utc)
                if isinstance(result, Exception):
                    expert_results[expert.agent_id] = AgentExecutionResult(
                        success=False,
                        result=None,
                        error_message=str(result),
                        execution_time_ms=0,
                        started_at=now,
                        completed_at=now
                    )
                else:
                    # result is TaskResult in this case
                    if hasattr(result, 'success'):
                        expert_results[expert.agent_id] = AgentExecutionResult(
                            success=getattr(result, 'success', False),
                            result=getattr(result, 'result', None),
                            error_message=getattr(result, 'error_message', None),
                            execution_time_ms=getattr(result, 'execution_time_ms', 0) or 0,
                            started_at=getattr(result, 'started_at', now) or now,
                            completed_at=getattr(result, 'completed_at', now) or now
                        )
                    else:
                        expert_results[expert.agent_id] = AgentExecutionResult(
                            success=False,
                            result=None,
                            error_message=f"Unknown result type: {type(result)}",
                            execution_time_ms=0,
                            started_at=now,
                            completed_at=now
                        )
        
        # Phase 3: Manager synthesizes results
        now = datetime.now(timezone.utc)
        synthesis_task = Task(
            title=f"Synthesize results: {task.title}",
            description="Synthesize the results from expert agents into a final response",
            input_data={
                "original_task": task.model_dump(),
                "expert_results": {
                    agent_id: result.model_dump()
                    for agent_id, result in expert_results.items()
                }
            },
            output_data=None,
            timeout_seconds=None,
            started_at=now,
            completed_at=None,
            parent_task_id=task.task_id
        )

        final_result = await self._task_executor.execute(synthesis_task, manager_agent, context)

        # Aggregate all results
        all_agent_results = {manager_agent.agent_id: AgentExecutionResult(
            success=final_result.success,
            result=final_result.result,
            error_message=final_result.error_message,
            execution_time_ms=final_result.execution_time_ms or 0,
            started_at=final_result.started_at or now,
            completed_at=final_result.completed_at or now
        )}
        all_agent_results.update(expert_results)

        return MultiAgentResult(
            success=final_result.success,
            result=final_result.result,
            error_message=final_result.error_message,
            agent_results=all_agent_results,
            task_results={task.task_id: final_result},
            total_execution_time_ms=final_result.execution_time_ms or 0,
            started_at=final_result.started_at or now,
            completed_at=final_result.completed_at or now,
            workflow_type=WorkflowType.HIERARCHICAL,
            agents_used=[agent.agent_id for agent in agents]
        )
    
    async def _execute_sequential_workflow(
        self, 
        config: WorkflowConfig,
        task: Task, 
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute sequential (pipeline) workflow."""
        if not agents:
            raise WorkflowExecutionError("No agents provided for sequential workflow")
        
        logger.info(f"Sequential workflow with {len(agents)} agents")

        agent_results: Dict[str, AgentExecutionResult] = {}
        task_results: Dict[str, TaskResult] = {}
        current_input = task.input_data

        for i, agent in enumerate(agents):
            # Create task for current agent
            now = datetime.now(timezone.utc)
            agent_task = Task(
                title=f"Stage {i+1}: {task.title}",
                description=f"Process stage {i+1} of the pipeline: {task.description}",
                input_data=current_input,
                output_data=None,
                timeout_seconds=None,
                started_at=now,
                completed_at=None,
                parent_task_id=task.task_id
            )

            # Execute task
            result = await self._task_executor.execute(agent_task, agent, context)

            # Store results
            agent_results[agent.agent_id] = AgentExecutionResult(
                success=result.success,
                result=result.result,
                error_message=result.error_message,
                execution_time_ms=result.execution_time_ms or 0,
                started_at=result.started_at or now,
                completed_at=result.completed_at or now
            )
            task_results[agent_task.task_id] = result
            
            # Check for failure
            if not result.success:
                if config.execution_strategy == ExecutionStrategy.FAIL_FAST:
                    now = datetime.now(timezone.utc)
                    return MultiAgentResult(
                        success=False,
                        result=None,
                        error_message=f"Sequential workflow failed at stage {i+1}: {result.error_message}",
                        agent_results=agent_results,
                        task_results=task_results,
                        total_execution_time_ms=0,
                        started_at=now,
                        completed_at=now,
                        workflow_type=WorkflowType.SEQUENTIAL,
                        agents_used=[a.agent_id for a in agents[:i+1]]
                    )
                else:
                    # Continue with empty input for next stage
                    current_input = {}
            else:
                # Use output as input for next stage
                current_input = result.result or {}

        # Get final result from last successful agent
        final_result = None
        for agent in reversed(agents):
            if agent.agent_id in agent_results and agent_results[agent.agent_id].success:
                final_result = agent_results[agent.agent_id].result
                break

        now = datetime.now(timezone.utc)
        return MultiAgentResult(
            success=True,
            result=final_result,
            error_message=None,
            agent_results=agent_results,
            task_results=task_results,
            total_execution_time_ms=0,
            started_at=now,
            completed_at=now,
            workflow_type=WorkflowType.SEQUENTIAL,
            agents_used=[agent.agent_id for agent in agents]
        )
    
    async def _execute_parallel_workflow(
        self, 
        config: WorkflowConfig,
        task: Task, 
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute parallel workflow."""
        if not agents:
            raise WorkflowExecutionError("No agents provided for parallel workflow")
        
        logger.info(f"Parallel workflow with {len(agents)} agents")
        
        # Create tasks for all agents
        agent_tasks: List[Any] = []
        for i, agent in enumerate(agents):
            now = datetime.now(timezone.utc)
            agent_task = Task(
                title=f"Parallel task {i+1}: {task.title}",
                description=f"Execute parallel task: {task.description}",
                input_data=task.input_data,
                output_data=None,
                timeout_seconds=None,
                started_at=now,
                completed_at=None,
                parent_task_id=task.task_id
            )
            agent_tasks.append(self._task_executor.execute(agent_task, agent, context))

        # Execute all tasks in parallel
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Process results
        agent_results: Dict[str, AgentExecutionResult] = {}
        task_results: Dict[str, TaskResult] = {}
        successful_results: List[Any] = []

        for i, result in enumerate(results):
            agent = agents[i]
            now = datetime.now(timezone.utc)
            if isinstance(result, Exception):
                agent_results[agent.agent_id] = AgentExecutionResult(
                    success=False,
                    result=None,
                    error_message=str(result),
                    execution_time_ms=0,
                    started_at=now,
                    completed_at=now
                )
            else:
                # result is TaskResult in this case
                if hasattr(result, 'success'):
                    agent_results[agent.agent_id] = AgentExecutionResult(
                        success=getattr(result, 'success', False),
                        result=getattr(result, 'result', None),
                        error_message=getattr(result, 'error_message', None),
                        execution_time_ms=getattr(result, 'execution_time_ms', 0) or 0,
                        started_at=getattr(result, 'started_at', now) or now,
                        completed_at=getattr(result, 'completed_at', now) or now
                    )
                    if isinstance(result, TaskResult):
                        task_results[f"parallel_task_{i+1}"] = result

                    if getattr(result, 'success', False):
                        successful_results.append(getattr(result, 'result', None))
                else:
                    agent_results[agent.agent_id] = AgentExecutionResult(
                        success=False,
                        result=None,
                        error_message=f"Unknown result type: {type(result)}",
                        execution_time_ms=0,
                        started_at=now,
                        completed_at=now
                    )
        
        # Aggregate results
        aggregated_result = await self._result_aggregator.aggregate_results(
            task, agent_results, strategy="parallel"
        )
        
        return aggregated_result
    
    async def _execute_custom_workflow(
        self, 
        config: WorkflowConfig,
        task: Task, 
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute custom workflow based on workflow definition."""
        if not config.workflow_definition:
            raise WorkflowExecutionError("No workflow definition provided for custom workflow")
        
        # TODO: Implement custom workflow execution based on workflow definition
        # This would parse the workflow definition and execute steps accordingly
        
        # For now, fall back to single agent execution
        return await self._execute_single_workflow(config, task, agents, context)
    
    async def _execute_streaming_workflow(
        self, 
        config: WorkflowConfig,
        task: Task, 
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute streaming workflow."""
        # For now, execute as single workflow
        # TODO: Implement actual streaming execution
        return await self._execute_single_workflow(config, task, agents, context)
    
    async def _execute_streaming_workflow_stream(
        self, 
        config: WorkflowConfig,
        task: Task, 
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute streaming workflow with real-time updates."""
        yield {"type": "streaming_started", "agent_count": len(agents)}
        
        # TODO: Implement actual streaming execution with real-time updates
        # For now, just execute and yield final result
        result = await self._execute_single_workflow(config, task, agents, context)
        yield {"type": "streaming_result", "result": result.model_dump()}
    
    async def _execute_batch_workflow(
        self, 
        config: WorkflowConfig,
        task: Task, 
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute batch workflow."""
        # For now, execute as parallel workflow
        # TODO: Implement actual batch processing logic
        return await self._execute_parallel_workflow(config, task, agents, context)
