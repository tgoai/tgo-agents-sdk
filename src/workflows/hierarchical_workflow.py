"""
Hierarchical workflow implementation.

This module implements the manager-expert hierarchical workflow pattern.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .base_workflow import BaseWorkflow
from ..core.models import (
    Task, AgentInstance, WorkflowConfig, MultiAgentResult,
    ExecutionContext, TaskResult, AgentExecutionResult
)
from ..core.enums import WorkflowType, ExecutionStrategy, AgentType
from ..core.exceptions import WorkflowError, WorkflowExecutionError
from ..coordinator.task_executor import TaskExecutor

logger = logging.getLogger(__name__)


class HierarchicalWorkflow(BaseWorkflow):
    """Hierarchical workflow implementation.
    
    This workflow follows the manager-expert pattern where:
    1. Manager agent analyzes and decomposes the task
    2. Expert agents execute specialized subtasks
    3. Manager agent synthesizes the final result
    """
    
    def __init__(self):
        super().__init__(WorkflowType.HIERARCHICAL)
        self._task_executor = TaskExecutor()
    
    def validate_agents(self, agents: List[AgentInstance], config: WorkflowConfig) -> None:
        """Validate agents for hierarchical workflow."""
        super().validate_agents(agents, config)
        
        # Find manager and expert agents
        managers = [a for a in agents if a.config.agent_type == AgentType.MANAGER]
        experts = [a for a in agents if a.config.agent_type == AgentType.EXPERT]
        
        if not managers:
            raise WorkflowError("Hierarchical workflow requires at least one manager agent")
        
        if not experts:
            raise WorkflowError("Hierarchical workflow requires at least one expert agent")
        
        if len(managers) > 1:
            # If multiple managers, check if specific manager is configured
            if config.manager_agent_id:
                manager_found = any(a.agent_id == config.manager_agent_id for a in managers)
                if not manager_found:
                    raise WorkflowError(f"Configured manager agent {config.manager_agent_id} not found")
            else:
                self._logger.warning(f"Multiple managers found, will use first one: {managers[0].agent_id}")
    
    async def execute(
        self,
        config: WorkflowConfig,
        task: Task,
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute hierarchical workflow."""
        self.log_workflow_start(task, agents)
        
        try:
            # Validate agents
            self.validate_agents(agents, config)
            
            # Identify manager and expert agents
            manager_agent, expert_agents = self._identify_agents(agents, config)
            
            # Phase 1: Manager analyzes and decomposes task
            decomposition_result = await self._manager_decompose_task(
                manager_agent, task, expert_agents, context
            )
            
            if not decomposition_result.success:
                return self.create_execution_result(
                    success=False,
                    error_message=f"Task decomposition failed: {decomposition_result.error_message}",
                    agent_results={manager_agent.agent_id: AgentExecutionResult(
                        success=False,
                        error_message=decomposition_result.error_message
                    )},
                    agents_used=[manager_agent.agent_id]
                )
            
            # Phase 2: Execute subtasks with expert agents
            expert_results = await self._execute_expert_tasks(
                expert_agents, task, context, config.execution_strategy
            )
            
            # Phase 3: Manager synthesizes results
            synthesis_result = await self._manager_synthesize_results(
                manager_agent, task, expert_results, context
            )
            
            # Combine all results
            all_agent_results = {manager_agent.agent_id: AgentExecutionResult(
                success=synthesis_result.success,
                result=synthesis_result.result,
                error_message=synthesis_result.error_message
            )}
            all_agent_results.update(expert_results)
            
            result = self.create_execution_result(
                success=synthesis_result.success,
                result=synthesis_result.result,
                error_message=synthesis_result.error_message,
                agent_results=all_agent_results,
                task_results={task.task_id: synthesis_result},
                agents_used=[agent.agent_id for agent in agents]
            )
            
            self.log_workflow_complete(result)
            return result
            
        except Exception as e:
            return await self.handle_execution_error(e, task, agents, context)
    
    def _identify_agents(
        self, 
        agents: List[AgentInstance], 
        config: WorkflowConfig
    ) -> tuple[AgentInstance, List[AgentInstance]]:
        """Identify manager and expert agents."""
        managers = [a for a in agents if a.config.agent_type == AgentType.MANAGER]
        experts = [a for a in agents if a.config.agent_type == AgentType.EXPERT]
        
        # Select manager
        if config.manager_agent_id:
            manager_agent = next(
                (a for a in managers if a.agent_id == config.manager_agent_id), 
                managers[0]
            )
        else:
            manager_agent = managers[0]
        
        # Filter experts if specific ones are configured
        if config.expert_agent_ids:
            expert_agents = [
                a for a in experts 
                if a.agent_id in config.expert_agent_ids
            ]
        else:
            expert_agents = experts
        
        return manager_agent, expert_agents
    
    async def _manager_decompose_task(
        self,
        manager_agent: AgentInstance,
        task: Task,
        expert_agents: List[AgentInstance],
        context: ExecutionContext
    ) -> TaskResult:
        """Manager decomposes the task into subtasks."""
        decomposition_task = Task(
            title=f"Analyze and decompose: {task.title}",
            description=f"""Analyze the following task and create an execution plan:

Original Task: {task.title}
Description: {task.description or 'No description provided'}
Input Data: {task.input_data}

Available Expert Agents:
{self._format_expert_capabilities(expert_agents)}

Please provide:
1. Analysis of the task requirements
2. Breakdown into subtasks for expert agents
3. Execution strategy and coordination plan
4. Expected deliverables from each expert
""",
            input_data={
                "original_task": task.model_dump(),
                "available_experts": [
                    {
                        "agent_id": agent.agent_id,
                        "name": agent.config.name,
                        "capabilities": agent.config.capabilities,
                        "tools": agent.config.tools
                    }
                    for agent in expert_agents
                ]
            }
        )
        
        self._logger.info(f"Manager {manager_agent.agent_id} decomposing task")
        return await self._task_executor.execute(decomposition_task, manager_agent, context)
    
    async def _execute_expert_tasks(
        self,
        expert_agents: List[AgentInstance],
        task: Task,
        context: ExecutionContext,
        strategy: ExecutionStrategy
    ) -> Dict[str, AgentExecutionResult]:
        """Execute tasks with expert agents."""
        expert_results = {}
        
        if strategy == ExecutionStrategy.FAIL_FAST:
            # Execute experts sequentially, stop on first failure
            for expert in expert_agents:
                result = await self._execute_single_expert_task(expert, task, context)
                expert_results[expert.agent_id] = AgentExecutionResult(
                    success=result.success,
                    result=result.result,
                    error_message=result.error_message
                )
                
                self.log_agent_execution(expert.agent_id, result.success, result.error_message)
                
                if not result.success:
                    self._logger.warning(f"Expert {expert.agent_id} failed, stopping execution")
                    break
        else:
            # Execute experts in parallel
            import asyncio
            expert_tasks = [
                self._execute_single_expert_task(expert, task, context)
                for expert in expert_agents
            ]
            
            results = await asyncio.gather(*expert_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                expert = expert_agents[i]
                if isinstance(result, Exception):
                    expert_results[expert.agent_id] = AgentExecutionResult(
                        success=False,
                        error_message=str(result)
                    )
                    self.log_agent_execution(expert.agent_id, False, str(result))
                else:
                    expert_results[expert.agent_id] = AgentExecutionResult(
                        success=result.success,
                        result=result.result,
                        error_message=result.error_message
                    )
                    self.log_agent_execution(expert.agent_id, result.success, result.error_message)
        
        return expert_results
    
    async def _execute_single_expert_task(
        self,
        expert_agent: AgentInstance,
        original_task: Task,
        context: ExecutionContext
    ) -> TaskResult:
        """Execute task with a single expert agent."""
        expert_task = Task(
            title=f"Expert task for {expert_agent.config.name}",
            description=f"""Execute your specialized part of this task using your expertise in {', '.join(expert_agent.config.capabilities)}:

Original Task: {original_task.title}
Description: {original_task.description or 'No description provided'}

Focus on your area of expertise and provide detailed, high-quality results.
Use your available tools and knowledge bases as needed.
""",
            input_data=original_task.input_data
        )
        
        return await self._task_executor.execute(expert_task, expert_agent, context)
    
    async def _manager_synthesize_results(
        self,
        manager_agent: AgentInstance,
        task: Task,
        expert_results: Dict[str, AgentExecutionResult],
        context: ExecutionContext
    ) -> TaskResult:
        """Manager synthesizes results from expert agents."""
        synthesis_task = Task(
            title=f"Synthesize results: {task.title}",
            description=f"""Synthesize the results from expert agents into a comprehensive final response:

Original Task: {task.title}
Description: {task.description or 'No description provided'}

Expert Results:
{self._format_expert_results(expert_results)}

Please provide:
1. Comprehensive synthesis of all expert contributions
2. Final answer addressing the original task
3. Quality assessment and confidence level
4. Any recommendations or next steps
""",
            input_data={
                "original_task": task.model_dump(),
                "expert_results": {
                    agent_id: {
                        "success": result.success,
                        "result": result.result,
                        "error": result.error_message
                    }
                    for agent_id, result in expert_results.items()
                }
            }
        )
        
        self._logger.info(f"Manager {manager_agent.agent_id} synthesizing results")
        return await self._task_executor.execute(synthesis_task, manager_agent, context)
    
    def _format_expert_capabilities(self, expert_agents: List[AgentInstance]) -> str:
        """Format expert agent capabilities for display."""
        lines = []
        for agent in expert_agents:
            capabilities = ', '.join(agent.config.capabilities) if agent.config.capabilities else 'General'
            tools = ', '.join(agent.config.tools) if agent.config.tools else 'None'
            lines.append(f"- {agent.config.name} ({agent.agent_id}): {capabilities} | Tools: {tools}")
        return '\n'.join(lines)
    
    def _format_expert_results(self, expert_results: Dict[str, AgentExecutionResult]) -> str:
        """Format expert results for display."""
        lines = []
        for agent_id, result in expert_results.items():
            status = "✓ Success" if result.success else "✗ Failed"
            content = str(result.result) if result.result else result.error_message or "No output"
            lines.append(f"- {agent_id}: {status}\n  {content[:200]}{'...' if len(str(content)) > 200 else ''}")
        return '\n'.join(lines)
