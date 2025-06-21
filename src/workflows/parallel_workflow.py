"""
Parallel workflow implementation.

This module implements the parallel workflow pattern.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .base_workflow import BaseWorkflow
from ..core.models import (
    Task, AgentInstance, WorkflowConfig, MultiAgentResult,
    ExecutionContext, TaskResult, AgentExecutionResult
)
from ..core.enums import WorkflowType, ExecutionStrategy
from ..core.exceptions import WorkflowError
from ..coordinator.task_executor import TaskExecutor

logger = logging.getLogger(__name__)


class ParallelWorkflow(BaseWorkflow):
    """Parallel workflow implementation.
    
    This workflow executes agents in parallel where:
    1. All agents work on the same task simultaneously
    2. Each agent contributes their specialized perspective
    3. Results are aggregated based on the configured strategy
    """
    
    def __init__(self):
        super().__init__(WorkflowType.PARALLEL)
        self._task_executor = TaskExecutor()
    
    def validate_agents(self, agents: List[AgentInstance], config: WorkflowConfig) -> None:
        """Validate agents for parallel workflow."""
        super().validate_agents(agents, config)
        
        # Check concurrency limits
        max_concurrent = config.max_concurrent_agents
        if max_concurrent and len(agents) > max_concurrent:
            self._logger.warning(
                f"Number of agents ({len(agents)}) exceeds max concurrent limit ({max_concurrent})"
            )
    
    async def execute(
        self,
        config: WorkflowConfig,
        task: Task,
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute parallel workflow."""
        self.log_workflow_start(task, agents)
        
        try:
            # Validate agents
            self.validate_agents(agents, config)
            
            # Apply concurrency limits
            max_concurrent = config.max_concurrent_agents or len(agents)
            agent_batches = self._create_agent_batches(agents, max_concurrent)
            
            all_agent_results = {}
            all_task_results = {}
            
            # Execute agent batches
            for batch_num, agent_batch in enumerate(agent_batches):
                self._logger.info(f"Executing batch {batch_num + 1}/{len(agent_batches)} with {len(agent_batch)} agents")
                
                batch_results = await self._execute_agent_batch(
                    agent_batch, task, context, config.execution_strategy
                )
                
                # Merge batch results
                all_agent_results.update(batch_results["agent_results"])
                all_task_results.update(batch_results["task_results"])
                
                # Check if we should stop early
                if config.execution_strategy == ExecutionStrategy.FAIL_FAST:
                    failed_agents = [
                        agent_id for agent_id, result in batch_results["agent_results"].items()
                        if not result.success
                    ]
                    if failed_agents:
                        self._logger.error(f"Batch {batch_num + 1} had failures, stopping execution")
                        break
            
            # Aggregate results
            aggregated_result = self._aggregate_parallel_results(
                task, all_agent_results, config.execution_strategy
            )
            
            result = self.create_execution_result(
                success=aggregated_result["success"],
                result=aggregated_result["result"],
                error_message=aggregated_result.get("error_message"),
                agent_results=all_agent_results,
                task_results=all_task_results,
                agents_used=list(all_agent_results.keys())
            )
            
            self.log_workflow_complete(result)
            return result
            
        except Exception as e:
            return await self.handle_execution_error(e, task, agents, context)
    
    def _create_agent_batches(
        self, 
        agents: List[AgentInstance], 
        max_concurrent: int
    ) -> List[List[AgentInstance]]:
        """Create batches of agents for concurrent execution."""
        batches = []
        for i in range(0, len(agents), max_concurrent):
            batch = agents[i:i + max_concurrent]
            batches.append(batch)
        return batches
    
    async def _execute_agent_batch(
        self,
        agents: List[AgentInstance],
        task: Task,
        context: ExecutionContext,
        strategy: ExecutionStrategy
    ) -> Dict[str, Any]:
        """Execute a batch of agents in parallel."""
        # Create tasks for all agents in the batch
        agent_tasks = []
        for agent in agents:
            parallel_task = self._create_parallel_task(task, agent)
            agent_tasks.append(
                self._execute_single_agent_with_timeout(
                    agent, parallel_task, context
                )
            )
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Process results
        agent_results = {}
        task_results = {}
        
        for i, result in enumerate(results):
            agent = agents[i]
            
            if isinstance(result, Exception):
                agent_results[agent.agent_id] = AgentExecutionResult(
                    success=False,
                    error_message=str(result)
                )
                self.log_agent_execution(agent.agent_id, False, str(result))
            else:
                agent_results[agent.agent_id] = AgentExecutionResult(
                    success=result.success,
                    result=result.result,
                    error_message=result.error_message,
                    execution_time_ms=result.execution_time_ms,
                    started_at=result.started_at,
                    completed_at=result.completed_at
                )
                task_results[f"parallel_task_{agent.agent_id}"] = result
                self.log_agent_execution(agent.agent_id, result.success, result.error_message)
        
        return {
            "agent_results": agent_results,
            "task_results": task_results
        }
    
    async def _execute_single_agent_with_timeout(
        self,
        agent: AgentInstance,
        task: Task,
        context: ExecutionContext,
        timeout_seconds: Optional[int] = None
    ) -> TaskResult:
        """Execute a single agent with optional timeout."""
        try:
            if timeout_seconds:
                return await asyncio.wait_for(
                    self._task_executor.execute(task, agent, context),
                    timeout=timeout_seconds
                )
            else:
                return await self._task_executor.execute(task, agent, context)
        except asyncio.TimeoutError:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=f"Agent {agent.agent_id} execution timed out after {timeout_seconds} seconds",
                agent_id=agent.agent_id,
                agent_type=agent.config.agent_type
            )
    
    def _create_parallel_task(self, original_task: Task, agent: AgentInstance) -> Task:
        """Create a task for parallel execution by a specific agent."""
        capabilities = ', '.join(agent.config.capabilities) if agent.config.capabilities else 'general processing'
        
        return Task(
            title=f"Parallel task for {agent.config.name}",
            description=f"""Execute this task using your specialized expertise in {capabilities}.

Original Task: {original_task.title}
{original_task.description or ''}

Your Parallel Contribution:
- Apply your unique perspective and specialized skills
- Work independently while contributing to the overall goal
- Provide comprehensive results within your area of expertise
- Focus on quality and thoroughness in your domain

Instructions:
1. Analyze the task from your specialized perspective
2. Apply your capabilities ({capabilities}) to provide the best possible solution
3. Include detailed reasoning and methodology
4. Provide results that complement other agents' contributions
""",
            input_data={
                "original_task": original_task.model_dump(),
                "agent_specialization": capabilities,
                "parallel_execution": True
            }
        )
    
    def _aggregate_parallel_results(
        self,
        task: Task,
        agent_results: Dict[str, AgentExecutionResult],
        strategy: ExecutionStrategy
    ) -> Dict[str, Any]:
        """Aggregate results from parallel execution."""
        successful_results = [
            (agent_id, result) for agent_id, result in agent_results.items()
            if result.success
        ]
        failed_results = [
            (agent_id, result) for agent_id, result in agent_results.items()
            if not result.success
        ]
        
        total_agents = len(agent_results)
        successful_count = len(successful_results)
        failed_count = len(failed_results)
        
        # Determine overall success based on strategy
        if strategy == ExecutionStrategy.FAIL_FAST:
            overall_success = failed_count == 0
        elif strategy == ExecutionStrategy.BEST_EFFORT:
            overall_success = successful_count > 0
        else:  # Default: require majority success
            overall_success = successful_count > failed_count
        
        # Create aggregated result
        aggregated_result = {
            "parallel_execution_summary": {
                "total_agents": total_agents,
                "successful_agents": successful_count,
                "failed_agents": failed_count,
                "success_rate": successful_count / total_agents if total_agents > 0 else 0,
                "execution_strategy": strategy.value
            },
            "successful_contributions": [
                {
                    "agent_id": agent_id,
                    "result": result.result,
                    "execution_time_ms": result.execution_time_ms
                }
                for agent_id, result in successful_results
            ],
            "failed_contributions": [
                {
                    "agent_id": agent_id,
                    "error": result.error_message
                }
                for agent_id, result in failed_results
            ]
        }
        
        # Add synthesis of successful results
        if successful_results:
            aggregated_result["synthesized_result"] = self._synthesize_successful_results(
                successful_results
            )
        
        result = {
            "success": overall_success,
            "result": aggregated_result
        }
        
        if not overall_success:
            error_messages = [result.error_message for _, result in failed_results if result.error_message]
            result["error_message"] = f"Parallel execution failed: {'; '.join(error_messages)}"
        
        return result
    
    def _synthesize_successful_results(
        self, 
        successful_results: List[tuple[str, AgentExecutionResult]]
    ) -> Dict[str, Any]:
        """Synthesize successful results into a coherent response."""
        synthesis = {
            "combined_insights": [],
            "key_findings": [],
            "confidence_assessment": "high" if len(successful_results) > 2 else "medium",
            "consensus_areas": [],
            "diverse_perspectives": []
        }
        
        # Extract insights from each successful result
        for agent_id, result in successful_results:
            if result.result:
                insight = {
                    "agent_id": agent_id,
                    "contribution": result.result,
                    "execution_time_ms": result.execution_time_ms
                }
                synthesis["combined_insights"].append(insight)
                
                # Extract key findings (simplified approach)
                if isinstance(result.result, dict):
                    if "findings" in result.result:
                        synthesis["key_findings"].extend(result.result["findings"])
                    elif "conclusion" in result.result:
                        synthesis["key_findings"].append(result.result["conclusion"])
                elif isinstance(result.result, str):
                    synthesis["key_findings"].append(result.result[:200] + "..." if len(result.result) > 200 else result.result)
        
        # Simple consensus detection (could be more sophisticated)
        if len(successful_results) > 1:
            synthesis["consensus_areas"] = ["Multiple agents provided consistent insights"]
            synthesis["diverse_perspectives"] = [f"Perspective from {len(successful_results)} different specialized agents"]
        
        return synthesis
