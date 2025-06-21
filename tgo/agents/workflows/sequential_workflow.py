"""
Sequential workflow implementation.

This module implements the sequential (pipeline) workflow pattern.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .base_workflow import BaseWorkflow
from ..core.models import (
    Task, AgentInstance, WorkflowConfig, MultiAgentResult,
    ExecutionContext, TaskResult, AgentExecutionResult
)
from ..core.enums import WorkflowType, ExecutionStrategy

from ..coordinator.task_executor import TaskExecutor
from ..registry import AdapterRegistry

logger = logging.getLogger(__name__)


class SequentialWorkflow(BaseWorkflow):
    """Sequential workflow implementation.
    
    This workflow executes agents in sequence where:
    1. Each agent processes the task in order
    2. Output from one agent becomes input for the next
    3. Final result comes from the last successful agent
    """
    
    def __init__(self, registry: AdapterRegistry):
        super().__init__(WorkflowType.SEQUENTIAL)
        self._task_executor = TaskExecutor(registry)
    
    def validate_agents(self, agents: List[AgentInstance], config: WorkflowConfig) -> None:
        """Validate agents for sequential workflow."""
        super().validate_agents(agents, config)
        
        if len(agents) < 2:
            self._logger.warning("Sequential workflow works best with multiple agents")
    
    async def execute(
        self,
        config: WorkflowConfig,
        task: Task,
        agents: List[AgentInstance],
        context: ExecutionContext
    ) -> MultiAgentResult:
        """Execute sequential workflow."""
        self.log_workflow_start(task, agents)
        
        try:
            # Validate agents
            self.validate_agents(agents, config)
            
            # Execute pipeline stages
            agent_results: Dict[str, AgentExecutionResult] = {}
            task_results: Dict[str, TaskResult] = {}
            current_input = task.input_data
            pipeline_stages: List[Dict[str, Any]] = []
            
            for i, agent in enumerate(agents):
                self._logger.info(f"Executing pipeline stage {i+1}/{len(agents)} with agent {agent.agent_id}")
                
                # Create task for current stage
                stage_task = self._create_stage_task(task, agent, current_input, i+1)
                
                # Execute stage
                result = await self._task_executor.execute(stage_task, agent, context)
                
                # Store results
                agent_results[agent.agent_id] = AgentExecutionResult(
                    success=result.success,
                    result=result.result,
                    error_message=result.error_message,
                    execution_time_ms=result.execution_time_ms,
                    started_at=result.started_at,
                    completed_at=result.completed_at
                )
                task_results[stage_task.task_id] = result
                
                # Track pipeline stage
                pipeline_stages.append({
                    "stage": i+1,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.config.name,
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms,
                    "error": result.error_message
                })
                
                self.log_agent_execution(agent.agent_id, result.success, result.error_message)
                
                # Handle failure
                if not result.success:
                    if config.execution_strategy == ExecutionStrategy.FAIL_FAST:
                        self._logger.error(f"Pipeline failed at stage {i+1}, stopping execution")
                        return self._create_pipeline_result(
                            success=False,
                            error_message=f"Pipeline failed at stage {i+1}: {result.error_message}",
                            agent_results=agent_results,
                            task_results=task_results,
                            pipeline_stages=pipeline_stages,
                            agents_used=[a.agent_id for a in agents[:i+1]],
                            task=task
                        )
                    else:
                        # Continue with empty input for next stage
                        self._logger.warning(f"Stage {i+1} failed, continuing with empty input")
                        current_input = {}
                else:
                    # Use output as input for next stage
                    if result.result and hasattr(result.result, 'get'):
                        # result.result is dict-like
                        current_input = result.result
                    else:
                        # Wrap non-dict results
                        current_input = {"previous_result": result.result}
            
            # Determine overall success and final result
            successful_stages: List[Dict[str, Any]] = [s for s in pipeline_stages if s["success"]]
            overall_success = len(successful_stages) > 0

            # Get final result from last successful stage
            final_result: Any = None
            if successful_stages:
                last_successful_agent: str = successful_stages[-1]["agent_id"]
                final_result = agent_results[last_successful_agent].result
            
            # Create comprehensive result
            pipeline_result: Dict[str, Any] = {
                "final_result": final_result,
                "pipeline_summary": {
                    "total_stages": len(agents),
                    "successful_stages": len(successful_stages),
                    "failed_stages": len(pipeline_stages) - len(successful_stages),
                    "success_rate": len(successful_stages) / len(agents) if agents else 0
                },
                "pipeline_stages": pipeline_stages,
                "execution_flow": self._create_execution_flow(pipeline_stages)
            }
            
            result = self._create_pipeline_result(
                success=overall_success,
                result=pipeline_result,
                agent_results=agent_results,
                task_results=task_results,
                pipeline_stages=pipeline_stages,
                agents_used=[agent.agent_id for agent in agents],
                task=task
            )
            
            self.log_workflow_complete(result)
            return result
            
        except Exception as e:
            return await self.handle_execution_error(e, task, agents, context)
    
    def _create_stage_task(
        self,
        original_task: Task,
        agent: AgentInstance,
        current_input: Dict[str, Any],
        stage_number: int
    ) -> Task:
        """Create a task for a specific pipeline stage."""
        # Determine stage description based on agent capabilities
        stage_description = self._generate_stage_description(agent, original_task, stage_number)
        
        now = datetime.now(timezone.utc)
        return Task(
            title=f"Pipeline Stage {stage_number}: {original_task.title}",
            description=stage_description,
            input_data={
                "original_task": {
                    "title": original_task.title,
                    "description": original_task.description,
                    "input_data": original_task.input_data
                },
                "stage_input": current_input,
                "stage_number": stage_number,
                "agent_capabilities": agent.config.capabilities
            },
            output_data=None,
            timeout_seconds=None,
            started_at=now,
            completed_at=None,
            parent_task_id=original_task.task_id
        )
    
    def _generate_stage_description(
        self,
        agent: AgentInstance,
        original_task: Task,
        stage_number: int
    ) -> str:
        """Generate description for a pipeline stage."""
        capabilities = ', '.join(agent.config.capabilities) if agent.config.capabilities else 'general processing'
        
        return f"""Process stage {stage_number} of the pipeline using your expertise in {capabilities}.

Original Task: {original_task.title}
{original_task.description or ''}

Your Role in Pipeline:
- Apply your specialized skills ({capabilities}) to process the input
- Build upon or transform the previous stage's output
- Prepare output that will be useful for subsequent stages
- Focus on your area of expertise while maintaining pipeline flow

Instructions:
1. Process the stage input using your capabilities
2. Add value through your specialized knowledge
3. Provide clear, structured output for the next stage
4. Include reasoning about your processing decisions
"""
    
    def _create_execution_flow(self, pipeline_stages: List[Dict[str, Any]]) -> List[str]:
        """Create a visual representation of the execution flow."""
        flow: List[str] = []
        for i, stage in enumerate(pipeline_stages):
            status_icon = "✓" if stage["success"] else "✗"
            arrow = " → " if i < len(pipeline_stages) - 1 else ""
            flow.append(f"{status_icon} {stage['agent_name']}{arrow}")
        return flow
    
    def _create_pipeline_result(
        self,
        success: bool,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        agent_results: Optional[Dict[str, AgentExecutionResult]] = None,
        task_results: Optional[Dict[str, TaskResult]] = None,
        pipeline_stages: Optional[List[Dict[str, Any]]] = None,
        agents_used: Optional[List[str]] = None,
        task: Optional[Task] = None
    ) -> MultiAgentResult:
        """Create a pipeline-specific execution result."""
        # Note: task parameter available for future use
        # Add pipeline metadata to result
        if result and pipeline_stages:
            result["pipeline_metadata"] = {
                "workflow_type": "sequential",
                "total_execution_time_ms": sum(
                    stage.get("execution_time_ms", 0) or 0 
                    for stage in pipeline_stages
                ),
                "stage_count": len(pipeline_stages),
                "success_rate": len([s for s in pipeline_stages if s["success"]]) / len(pipeline_stages)
            }
        
        return self.create_execution_result(
            success=success,
            result=result,
            error_message=error_message,
            agent_results=agent_results,
            task_results=task_results,
            agents_used=agents_used
        )
