"""
Unit tests for workflow engine.

This module tests the WorkflowEngine class and its different workflow
execution patterns.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from src.coordinator.workflow_engine import WorkflowEngine
from src.core.models import (
    Task, AgentInstance, WorkflowConfig, MultiAgentResult,
    ExecutionContext, TaskResult, AgentExecutionResult
)
from src.core.enums import WorkflowType, ExecutionStrategy, AgentType
from src.core.exceptions import WorkflowError, WorkflowExecutionError


class TestWorkflowEngine:
    """Test cases for WorkflowEngine."""

    def test_workflow_engine_initialization(self):
        """Test workflow engine initialization."""
        engine = WorkflowEngine()
        
        assert engine.supports_workflow_type(WorkflowType.SINGLE)
        assert engine.supports_workflow_type(WorkflowType.HIERARCHICAL)
        assert engine.supports_workflow_type(WorkflowType.SEQUENTIAL)
        assert engine.supports_workflow_type(WorkflowType.PARALLEL)
        assert engine.supports_workflow_type(WorkflowType.CUSTOM)

    def test_unsupported_workflow_type(self):
        """Test handling of unsupported workflow types."""
        engine = WorkflowEngine()
        
        # Create a mock workflow type that's not supported
        unsupported_type = "unsupported_workflow"
        assert not engine.supports_workflow_type(unsupported_type)

    @pytest.mark.asyncio
    async def test_unsupported_workflow_execution(self, sample_workflow_config, sample_task, sample_execution_context):
        """Test execution of unsupported workflow type fails."""
        engine = WorkflowEngine()

        # Create a mock unsupported workflow type
        class UnsupportedType:
            def __str__(self):
                return "unsupported"

        unsupported_type = UnsupportedType()

        with pytest.raises(WorkflowError):
            await engine.execute_workflow(
                unsupported_type,
                sample_workflow_config,
                sample_task,
                [],
                sample_execution_context
            )

    @pytest.mark.asyncio
    async def test_single_workflow_execution(self, sample_workflow_config, sample_task, sample_agent_instance, sample_execution_context):
        """Test single agent workflow execution."""
        engine = WorkflowEngine()
        
        # Mock task executor
        with patch.object(engine._task_executor, 'execute') as mock_execute:
            mock_execute.return_value = TaskResult(
                task_id=sample_task.task_id,
                success=True,
                result={"response": "Single agent response"},
                execution_time_ms=1000
            )
            
            result = await engine.execute_workflow(
                WorkflowType.SINGLE,
                sample_workflow_config,
                sample_task,
                [sample_agent_instance],
                sample_execution_context
            )
            
            assert isinstance(result, MultiAgentResult)
            assert result.success is True
            assert result.workflow_type == WorkflowType.SINGLE
            assert len(result.agents_used) == 1
            assert sample_agent_instance.agent_id in result.agents_used
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_single_workflow_no_agents(self, sample_workflow_config, sample_task, sample_execution_context):
        """Test single workflow with no agents fails."""
        engine = WorkflowEngine()
        
        with pytest.raises(WorkflowExecutionError):
            await engine.execute_workflow(
                WorkflowType.SINGLE,
                sample_workflow_config,
                sample_task,
                [],  # No agents
                sample_execution_context
            )

    @pytest.mark.asyncio
    async def test_hierarchical_workflow_execution(self, sample_workflow_config, sample_task, sample_execution_context):
        """Test hierarchical workflow execution."""
        engine = WorkflowEngine()
        
        # Create manager and expert agents
        manager_config = Mock()
        manager_config.agent_type = AgentType.MANAGER
        manager_agent = Mock()
        manager_agent.agent_id = "manager_001"
        manager_agent.config = manager_config
        
        expert_config = Mock()
        expert_config.agent_type = AgentType.EXPERT
        expert_agent = Mock()
        expert_agent.agent_id = "expert_001"
        expert_agent.config = expert_config
        
        agents = [manager_agent, expert_agent]
        
        # Mock task executor responses
        with patch.object(engine._task_executor, 'execute') as mock_execute:
            # Manager decomposition, expert execution, manager synthesis
            mock_execute.side_effect = [
                TaskResult(task_id="decomp", success=True, result={"plan": "decomposed"}),
                TaskResult(task_id="expert", success=True, result={"expert_result": "done"}),
                TaskResult(task_id="synth", success=True, result={"final": "synthesized"})
            ]
            
            result = await engine.execute_workflow(
                WorkflowType.HIERARCHICAL,
                sample_workflow_config,
                sample_task,
                agents,
                sample_execution_context
            )
            
            assert isinstance(result, MultiAgentResult)
            assert result.success is True
            assert result.workflow_type == WorkflowType.HIERARCHICAL
            assert len(result.agents_used) == 2
            assert mock_execute.call_count == 3  # decomposition + expert + synthesis

    @pytest.mark.asyncio
    async def test_hierarchical_workflow_no_manager(self, sample_workflow_config, sample_task, sample_execution_context):
        """Test hierarchical workflow without manager fails."""
        engine = WorkflowEngine()
        
        # Only expert agents, no manager
        expert_config = Mock()
        expert_config.agent_type = AgentType.EXPERT
        expert_agent = Mock()
        expert_agent.config = expert_config
        
        with pytest.raises(WorkflowExecutionError):
            await engine.execute_workflow(
                WorkflowType.HIERARCHICAL,
                sample_workflow_config,
                sample_task,
                [expert_agent],
                sample_execution_context
            )

    @pytest.mark.asyncio
    async def test_hierarchical_workflow_no_experts(self, sample_workflow_config, sample_task, sample_execution_context):
        """Test hierarchical workflow without experts fails."""
        engine = WorkflowEngine()
        
        # Only manager, no experts
        manager_config = Mock()
        manager_config.agent_type = AgentType.MANAGER
        manager_agent = Mock()
        manager_agent.config = manager_config
        
        with pytest.raises(WorkflowExecutionError):
            await engine.execute_workflow(
                WorkflowType.HIERARCHICAL,
                sample_workflow_config,
                sample_task,
                [manager_agent],
                sample_execution_context
            )

    @pytest.mark.asyncio
    async def test_hierarchical_workflow_manager_failure(self, sample_workflow_config, sample_task, sample_execution_context):
        """Test hierarchical workflow with manager failure."""
        engine = WorkflowEngine()
        
        # Create manager and expert agents
        manager_config = Mock()
        manager_config.agent_type = AgentType.MANAGER
        manager_agent = Mock()
        manager_agent.agent_id = "manager_001"
        manager_agent.config = manager_config
        
        expert_config = Mock()
        expert_config.agent_type = AgentType.EXPERT
        expert_agent = Mock()
        expert_agent.agent_id = "expert_001"
        expert_agent.config = expert_config
        
        agents = [manager_agent, expert_agent]
        
        # Mock manager decomposition failure
        with patch.object(engine._task_executor, 'execute') as mock_execute:
            mock_execute.return_value = TaskResult(
                task_id="decomp",
                success=False,
                error_message="Manager failed"
            )
            
            result = await engine.execute_workflow(
                WorkflowType.HIERARCHICAL,
                sample_workflow_config,
                sample_task,
                agents,
                sample_execution_context
            )
            
            assert result.success is False
            assert "Manager task decomposition failed" in result.error_message

    @pytest.mark.asyncio
    async def test_sequential_workflow_execution(self, sample_workflow_config, sample_task, sample_execution_context):
        """Test sequential workflow execution."""
        engine = WorkflowEngine()
        
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = Mock()
            agent.agent_id = f"agent_{i}"
            agents.append(agent)
        
        # Mock task executor
        with patch.object(engine._task_executor, 'execute') as mock_execute:
            mock_execute.side_effect = [
                TaskResult(task_id=f"task_{i}", success=True, result={"stage": i})
                for i in range(3)
            ]
            
            result = await engine.execute_workflow(
                WorkflowType.SEQUENTIAL,
                sample_workflow_config,
                sample_task,
                agents,
                sample_execution_context
            )
            
            assert result.success is True
            assert result.workflow_type == WorkflowType.SEQUENTIAL
            assert len(result.agents_used) == 3
            assert mock_execute.call_count == 3

    @pytest.mark.asyncio
    async def test_sequential_workflow_fail_fast(self, sample_workflow_config, sample_task, sample_execution_context):
        """Test sequential workflow with fail fast strategy."""
        engine = WorkflowEngine()
        sample_workflow_config.execution_strategy = ExecutionStrategy.FAIL_FAST
        
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = Mock()
            agent.agent_id = f"agent_{i}"
            agents.append(agent)
        
        # Mock task executor with failure in second stage
        with patch.object(engine._task_executor, 'execute') as mock_execute:
            mock_execute.side_effect = [
                TaskResult(task_id="task_0", success=True, result={"stage": 0}),
                TaskResult(task_id="task_1", success=False, error_message="Stage 2 failed"),
            ]
            
            result = await engine.execute_workflow(
                WorkflowType.SEQUENTIAL,
                sample_workflow_config,
                sample_task,
                agents,
                sample_execution_context
            )
            
            assert result.success is False
            assert "Sequential workflow failed at stage 2" in result.error_message
            assert mock_execute.call_count == 2  # Should stop after failure

    @pytest.mark.asyncio
    async def test_parallel_workflow_execution(self, sample_workflow_config, sample_task, sample_execution_context):
        """Test parallel workflow execution."""
        engine = WorkflowEngine()
        
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = Mock()
            agent.agent_id = f"agent_{i}"
            agents.append(agent)
        
        # Mock task executor
        with patch.object(engine._task_executor, 'execute') as mock_execute:
            mock_execute.side_effect = [
                TaskResult(task_id=f"task_{i}", success=True, result={"agent": i})
                for i in range(3)
            ]
            
            # Mock result aggregator
            with patch.object(engine._result_aggregator, 'aggregate_results') as mock_aggregate:
                mock_aggregate.return_value = MultiAgentResult(
                    success=True,
                    result={"aggregated": "results"},
                    workflow_type=WorkflowType.PARALLEL
                )
                
                result = await engine.execute_workflow(
                    WorkflowType.PARALLEL,
                    sample_workflow_config,
                    sample_task,
                    agents,
                    sample_execution_context
                )
                
                assert result.success is True
                assert result.workflow_type == WorkflowType.PARALLEL
                mock_aggregate.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_workflow_with_exceptions(self, sample_workflow_config, sample_task, sample_execution_context):
        """Test parallel workflow handling exceptions."""
        engine = WorkflowEngine()
        
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = Mock()
            agent.agent_id = f"agent_{i}"
            agents.append(agent)
        
        # Mock task executor with one exception
        with patch.object(engine._task_executor, 'execute') as mock_execute:
            mock_execute.side_effect = [
                TaskResult(task_id="task_0", success=True, result={"agent": 0}),
                Exception("Agent 1 failed"),
                TaskResult(task_id="task_2", success=True, result={"agent": 2})
            ]
            
            # Mock result aggregator
            with patch.object(engine._result_aggregator, 'aggregate_results') as mock_aggregate:
                mock_aggregate.return_value = MultiAgentResult(
                    success=True,
                    result={"aggregated": "results"},
                    workflow_type=WorkflowType.PARALLEL
                )
                
                result = await engine.execute_workflow(
                    WorkflowType.PARALLEL,
                    sample_workflow_config,
                    sample_task,
                    agents,
                    sample_execution_context
                )
                
                # Should handle exception and continue
                assert result.success is True
                mock_aggregate.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_workflow_fallback(self, sample_workflow_config, sample_task, sample_agent_instance, sample_execution_context):
        """Test custom workflow with workflow definition."""
        engine = WorkflowEngine()

        # Add workflow definition to config
        sample_workflow_config.workflow_definition = {
            "steps": [
                {"type": "agent", "agent_id": sample_agent_instance.agent_id}
            ]
        }

        # Mock task executor
        with patch.object(engine._task_executor, 'execute') as mock_execute:
            mock_execute.return_value = TaskResult(
                task_id=sample_task.task_id,
                success=True,
                result={"response": "Custom workflow response"}
            )

            result = await engine.execute_workflow(
                WorkflowType.CUSTOM,
                sample_workflow_config,
                sample_task,
                [sample_agent_instance],
                sample_execution_context
            )

            assert result.success is True
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_workflow_fallback(self, sample_workflow_config, sample_task, sample_agent_instance, sample_execution_context):
        """Test streaming workflow falls back to single workflow."""
        engine = WorkflowEngine()
        
        # Mock task executor
        with patch.object(engine._task_executor, 'execute') as mock_execute:
            mock_execute.return_value = TaskResult(
                task_id=sample_task.task_id,
                success=True,
                result={"response": "Streaming workflow response"}
            )
            
            result = await engine.execute_workflow(
                WorkflowType.STREAMING,
                sample_workflow_config,
                sample_task,
                [sample_agent_instance],
                sample_execution_context
            )
            
            assert result.success is True
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_workflow_fallback(self, sample_workflow_config, sample_task, sample_execution_context):
        """Test batch workflow falls back to parallel workflow."""
        engine = WorkflowEngine()
        
        # Create multiple agents
        agents = []
        for i in range(2):
            agent = Mock()
            agent.agent_id = f"agent_{i}"
            agents.append(agent)
        
        # Mock task executor
        with patch.object(engine._task_executor, 'execute') as mock_execute:
            mock_execute.side_effect = [
                TaskResult(task_id=f"task_{i}", success=True, result={"agent": i})
                for i in range(2)
            ]
            
            # Mock result aggregator
            with patch.object(engine._result_aggregator, 'aggregate_results') as mock_aggregate:
                mock_aggregate.return_value = MultiAgentResult(
                    success=True,
                    result={"aggregated": "batch results"},
                    workflow_type=WorkflowType.BATCH
                )
                
                result = await engine.execute_workflow(
                    WorkflowType.BATCH,
                    sample_workflow_config,
                    sample_task,
                    agents,
                    sample_execution_context
                )
                
                assert result.success is True
                mock_aggregate.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_execution_exception_handling(self, sample_workflow_config, sample_task, sample_agent_instance, sample_execution_context):
        """Test workflow execution exception handling."""
        engine = WorkflowEngine()
        
        # Mock task executor to raise exception
        with patch.object(engine._task_executor, 'execute') as mock_execute:
            mock_execute.side_effect = Exception("Task execution failed")
            
            with pytest.raises(WorkflowExecutionError):
                await engine.execute_workflow(
                    WorkflowType.SINGLE,
                    sample_workflow_config,
                    sample_task,
                    [sample_agent_instance],
                    sample_execution_context
                )

    @pytest.mark.asyncio
    async def test_workflow_stream_execution(self, sample_workflow_config, sample_task, sample_agent_instance, sample_execution_context):
        """Test workflow streaming execution."""
        engine = WorkflowEngine()
        
        # Mock task executor
        with patch.object(engine._task_executor, 'execute') as mock_execute:
            mock_execute.return_value = TaskResult(
                task_id=sample_task.task_id,
                success=True,
                result={"response": "Stream response"}
            )
            
            updates = []
            async for update in engine.execute_workflow_stream(
                WorkflowType.SINGLE,
                sample_workflow_config,
                sample_task,
                [sample_agent_instance],
                sample_execution_context
            ):
                updates.append(update)
            
            assert len(updates) >= 2  # At least start and end
            assert updates[0]["type"] == "workflow_started"
            assert updates[-1]["type"] == "workflow_completed"

    @pytest.mark.asyncio
    async def test_workflow_stream_execution_error(self, sample_workflow_config, sample_task, sample_agent_instance, sample_execution_context):
        """Test workflow streaming execution with error."""
        engine = WorkflowEngine()
        
        # Mock task executor to raise exception
        with patch.object(engine._task_executor, 'execute') as mock_execute:
            mock_execute.side_effect = Exception("Stream execution failed")
            
            updates = []
            async for update in engine.execute_workflow_stream(
                WorkflowType.SINGLE,
                sample_workflow_config,
                sample_task,
                [sample_agent_instance],
                sample_execution_context
            ):
                updates.append(update)
            
            # Should get error update
            error_updates = [u for u in updates if u.get("type") == "workflow_error"]
            assert len(error_updates) > 0
            assert "Stream execution failed" in error_updates[0]["error"]
