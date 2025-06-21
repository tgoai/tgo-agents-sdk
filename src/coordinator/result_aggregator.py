"""
Result aggregator implementation.

This module provides result aggregation logic for multi-agent execution.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timezone

from ..core.interfaces import ResultAggregator as IResultAggregator
from ..core.models import Task, AgentExecutionResult, MultiAgentResult, TaskResult
from ..core.enums import WorkflowType

logger = logging.getLogger(__name__)


class ResultAggregator(IResultAggregator):
    """Result aggregator for multi-agent execution results."""
    
    def __init__(self):
        self._aggregation_strategies = {
            "default": self._aggregate_default,
            "parallel": self._aggregate_parallel,
            "hierarchical": self._aggregate_hierarchical,
            "sequential": self._aggregate_sequential,
            "majority_vote": self._aggregate_majority_vote,
            "best_result": self._aggregate_best_result,
        }
        logger.info("ResultAggregator initialized")
    
    async def aggregate_results(
        self, 
        task: Task,
        agent_results: Dict[str, AgentExecutionResult],
        strategy: str = "default"
    ) -> MultiAgentResult:
        """Aggregate results from multiple agents.
        
        Args:
            task: Original task
            agent_results: Results from individual agents
            strategy: Aggregation strategy to use
            
        Returns:
            Aggregated multi-agent result
        """
        logger.info(f"Aggregating results from {len(agent_results)} agents using {strategy} strategy")
        
        if strategy not in self._aggregation_strategies:
            logger.warning(f"Unknown aggregation strategy: {strategy}, using default")
            strategy = "default"
        
        aggregator = self._aggregation_strategies[strategy]
        return await aggregator(task, agent_results)
    
    async def _aggregate_default(
        self, 
        task: Task,
        agent_results: Dict[str, AgentExecutionResult]
    ) -> MultiAgentResult:
        """Default aggregation strategy."""
        successful_results = [
            result for result in agent_results.values() 
            if result.is_successful()
        ]
        
        overall_success = len(successful_results) > 0
        
        if overall_success:
            # Combine all successful results
            combined_result = {
                "results": [result.result for result in successful_results],
                "summary": f"Successfully executed by {len(successful_results)} out of {len(agent_results)} agents"
            }
        else:
            combined_result = None
        
        # Get error messages from failed results
        error_messages = [
            result.error_message for result in agent_results.values()
            if not result.is_successful() and result.error_message
        ]
        
        return MultiAgentResult(
            success=overall_success,
            result=combined_result,
            error_message="; ".join(error_messages) if error_messages else None,
            agent_results=agent_results,
            task_results={task.task_id: TaskResult(
                task_id=task.task_id,
                success=overall_success,
                result=combined_result,
                error_message="; ".join(error_messages) if error_messages else None
            )},
            agents_used=list(agent_results.keys())
        )
    
    async def _aggregate_parallel(
        self, 
        task: Task,
        agent_results: Dict[str, AgentExecutionResult]
    ) -> MultiAgentResult:
        """Parallel aggregation strategy - combine all results."""
        successful_results = []
        failed_results = []
        
        for agent_id, result in agent_results.items():
            if result.is_successful():
                successful_results.append({
                    "agent_id": agent_id,
                    "result": result.result
                })
            else:
                failed_results.append({
                    "agent_id": agent_id,
                    "error": result.error_message
                })
        
        overall_success = len(successful_results) > 0
        
        combined_result = {
            "successful_agents": len(successful_results),
            "failed_agents": len(failed_results),
            "total_agents": len(agent_results),
            "results": successful_results,
            "failures": failed_results
        }
        
        return MultiAgentResult(
            success=overall_success,
            result=combined_result,
            agent_results=agent_results,
            task_results={task.task_id: TaskResult(
                task_id=task.task_id,
                success=overall_success,
                result=combined_result
            )},
            workflow_type=WorkflowType.PARALLEL,
            agents_used=list(agent_results.keys())
        )
    
    async def _aggregate_hierarchical(
        self, 
        task: Task,
        agent_results: Dict[str, AgentExecutionResult]
    ) -> MultiAgentResult:
        """Hierarchical aggregation strategy - manager result takes precedence."""
        # Find manager result (usually the last one in hierarchical workflows)
        manager_result = None
        expert_results = {}
        
        for agent_id, result in agent_results.items():
            # Simple heuristic: if result contains synthesis or final result, it's from manager
            if (result.result and isinstance(result.result, dict) and 
                any(key in str(result.result).lower() for key in ['synthesis', 'final', 'summary'])):
                manager_result = result
            else:
                expert_results[agent_id] = result
        
        if not manager_result:
            # If no clear manager result, use the last successful result
            for result in agent_results.values():
                if result.is_successful():
                    manager_result = result
                    break
        
        overall_success = manager_result is not None and manager_result.is_successful()
        
        combined_result = {
            "manager_result": manager_result.result if manager_result else None,
            "expert_results": {
                agent_id: result.result for agent_id, result in expert_results.items()
                if result.is_successful()
            },
            "coordination_summary": f"Manager coordinated {len(expert_results)} expert agents"
        }
        
        return MultiAgentResult(
            success=overall_success,
            result=combined_result,
            error_message=manager_result.error_message if manager_result and not manager_result.is_successful() else None,
            agent_results=agent_results,
            task_results={task.task_id: TaskResult(
                task_id=task.task_id,
                success=overall_success,
                result=combined_result
            )},
            workflow_type=WorkflowType.HIERARCHICAL,
            agents_used=list(agent_results.keys())
        )
    
    async def _aggregate_sequential(
        self, 
        task: Task,
        agent_results: Dict[str, AgentExecutionResult]
    ) -> MultiAgentResult:
        """Sequential aggregation strategy - use final stage result."""
        # In sequential workflows, the last successful result is the final output
        final_result = None
        pipeline_results = []
        
        # Collect results in order (assuming agent_results maintains order)
        for agent_id, result in agent_results.items():
            pipeline_results.append({
                "agent_id": agent_id,
                "success": result.is_successful(),
                "result": result.result,
                "error": result.error_message
            })
            
            if result.is_successful():
                final_result = result.result
        
        overall_success = final_result is not None
        
        combined_result = {
            "final_result": final_result,
            "pipeline_stages": pipeline_results,
            "stages_completed": len([r for r in pipeline_results if r["success"]]),
            "total_stages": len(pipeline_results)
        }
        
        return MultiAgentResult(
            success=overall_success,
            result=combined_result,
            agent_results=agent_results,
            task_results={task.task_id: TaskResult(
                task_id=task.task_id,
                success=overall_success,
                result=combined_result
            )},
            workflow_type=WorkflowType.SEQUENTIAL,
            agents_used=list(agent_results.keys())
        )
    
    async def _aggregate_majority_vote(
        self, 
        task: Task,
        agent_results: Dict[str, AgentExecutionResult]
    ) -> MultiAgentResult:
        """Majority vote aggregation strategy."""
        successful_results = [
            result.result for result in agent_results.values() 
            if result.is_successful()
        ]
        
        if not successful_results:
            return await self._aggregate_default(task, agent_results)
        
        # Simple majority vote based on string similarity
        # In a real implementation, this would be more sophisticated
        result_counts = {}
        for result in successful_results:
            result_str = str(result)
            result_counts[result_str] = result_counts.get(result_str, 0) + 1
        
        # Get the most common result
        majority_result = max(result_counts.items(), key=lambda x: x[1])
        
        combined_result = {
            "majority_result": majority_result[0],
            "vote_count": majority_result[1],
            "total_votes": len(successful_results),
            "confidence": majority_result[1] / len(successful_results)
        }
        
        return MultiAgentResult(
            success=True,
            result=combined_result,
            agent_results=agent_results,
            task_results={task.task_id: TaskResult(
                task_id=task.task_id,
                success=True,
                result=combined_result
            )},
            agents_used=list(agent_results.keys())
        )
    
    async def _aggregate_best_result(
        self, 
        task: Task,
        agent_results: Dict[str, AgentExecutionResult]
    ) -> MultiAgentResult:
        """Best result aggregation strategy - select highest quality result."""
        successful_results = [
            (agent_id, result) for agent_id, result in agent_results.items() 
            if result.is_successful()
        ]
        
        if not successful_results:
            return await self._aggregate_default(task, agent_results)
        
        # Simple quality scoring based on result length and execution time
        # In a real implementation, this would use more sophisticated metrics
        best_agent_id, best_result = max(
            successful_results,
            key=lambda x: self._calculate_result_quality(x[1])
        )
        
        combined_result = {
            "best_result": best_result.result,
            "best_agent": best_agent_id,
            "quality_score": self._calculate_result_quality(best_result),
            "alternatives": [
                {
                    "agent_id": agent_id,
                    "result": result.result,
                    "quality_score": self._calculate_result_quality(result)
                }
                for agent_id, result in successful_results
                if agent_id != best_agent_id
            ]
        }
        
        return MultiAgentResult(
            success=True,
            result=combined_result,
            agent_results=agent_results,
            task_results={task.task_id: TaskResult(
                task_id=task.task_id,
                success=True,
                result=combined_result
            )},
            agents_used=list(agent_results.keys())
        )
    
    def _calculate_result_quality(self, result: AgentExecutionResult) -> float:
        """Calculate quality score for a result."""
        score = 0.0
        
        # Base score for successful execution
        if result.is_successful():
            score += 1.0
        
        # Bonus for having detailed result
        if result.result:
            result_length = len(str(result.result))
            score += min(result_length / 1000, 1.0)  # Up to 1 point for length
        
        # Penalty for long execution time
        if result.execution_time_ms:
            time_penalty = min(result.execution_time_ms / 10000, 0.5)  # Up to 0.5 penalty
            score -= time_penalty
        
        # Bonus for reasoning steps
        if result.reasoning_steps:
            score += min(len(result.reasoning_steps) / 10, 0.5)  # Up to 0.5 for reasoning
        
        return max(score, 0.0)
