"""
Unit tests for multi-agent coordinator.

This module tests the MultiAgentCoordinator class and its orchestration
of multi-agent task execution.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from src.coordinator.multi_agent_coordinator import MultiAgentCoordinator
from src.core.models import (
    Task, AgentConfig, AgentInstance, MultiAgentConfig, WorkflowConfig,
    MultiAgentResult, AgentExecutionResult, TaskResult
)
from src.core.enums import WorkflowType, ExecutionStrategy, AgentType
from src.core.exceptions import (
    MultiAgentError, FrameworkNotFoundError, AgentCreationError,
    WorkflowExecutionError
)


class TestMultiAgentCoordinator:
    """Test cases for MultiAgentCoordinator."""

    def test_coordinator_initialization(self, registry_with_mock_adapter):
        """Test coordinator initialization."""
        coordinator = MultiAgentCoordinator(registry_with_mock_adapter)
        
        assert coordinator._registry == registry_with_mock_adapter
        assert coordinator._workflow_engine is not None
        assert coordinator._task_executor is not None
        assert coordinator._result_aggregator is not None
        assert coordinator._max_concurrent_executions == 10
        assert coordinator._default_timeout == 300

    def test_coordinator_with_default_registry(self):
        """Test coordinator with default registry."""
        coordinator = MultiAgentCoordinator()
        
        assert coordinator._registry is not None

    @pytest.mark.asyncio
    async def test_config_validation_success(self, coordinator_with_mock_registry, sample_multi_agent_config):
        """Test successful configuration validation."""
        coordinator = coordinator_with_mock_registry
        
        # Should not raise exception
        await coordinator._validate_config(sample_multi_agent_config)

    @pytest.mark.asyncio
    async def test_config_validation_no_framework(self, coordinator_with_mock_registry, sample_multi_agent_config):
        """Test configuration validation with no framework."""
        coordinator = coordinator_with_mock_registry
        sample_multi_agent_config.framework = ""
        
        with pytest.raises(MultiAgentError, match="Framework not specified"):
            await coordinator._validate_config(sample_multi_agent_config)

    @pytest.mark.asyncio
    async def test_config_validation_no_agents(self, coordinator_with_mock_registry, sample_multi_agent_config):
        """Test configuration validation with no agents."""
        coordinator = coordinator_with_mock_registry
        sample_multi_agent_config.agents = []
        
        with pytest.raises(MultiAgentError, match="No agents specified"):
            await coordinator._validate_config(sample_multi_agent_config)

    @pytest.mark.asyncio
    async def test_config_validation_no_workflow(self, coordinator_with_mock_registry, sample_multi_agent_config):
        """Test configuration validation with no workflow."""
        coordinator = coordinator_with_mock_registry
        sample_multi_agent_config.workflow = None
        
        with pytest.raises(MultiAgentError, match="Workflow configuration not specified"):
            await coordinator._validate_config(sample_multi_agent_config)

    @pytest.mark.asyncio
    async def test_config_validation_unregistered_framework(self, coordinator_with_mock_registry, sample_multi_agent_config):
        """Test configuration validation with unregistered framework."""
        coordinator = coordinator_with_mock_registry
        sample_multi_agent_config.framework = "unregistered-framework"
        
        with pytest.raises(FrameworkNotFoundError):
            await coordinator._validate_config(sample_multi_agent_config)

    @pytest.mark.asyncio
    async def test_get_framework_adapter_success(self, coordinator_with_mock_registry, mock_adapter):
        """Test successful framework adapter retrieval."""
        coordinator = coordinator_with_mock_registry
        
        adapter = await coordinator._get_framework_adapter("test-framework")
        assert adapter == mock_adapter

    @pytest.mark.asyncio
    async def test_get_framework_adapter_not_found(self, coordinator_with_mock_registry):
        """Test framework adapter retrieval with nonexistent framework."""
        coordinator = coordinator_with_mock_registry
        
        with pytest.raises(FrameworkNotFoundError):
            await coordinator._get_framework_adapter("nonexistent-framework")

    @pytest.mark.asyncio
    async def test_get_framework_adapter_initialization(self, coordinator_with_mock_registry, mock_adapter):
        """Test framework adapter initialization during retrieval."""
        coordinator = coordinator_with_mock_registry
        mock_adapter.is_initialized = False
        
        adapter = await coordinator._get_framework_adapter("test-framework")
        
        assert adapter == mock_adapter
        coordinator._registry.initialize_adapter.assert_called_once_with("test-framework")

    @pytest.mark.asyncio
    async def test_create_agents_success(self, coordinator_with_mock_registry, mock_adapter, sample_agent_config):
        """Test successful agent creation."""
        coordinator = coordinator_with_mock_registry
        
        mock_instance = Mock(spec=AgentInstance)
        mock_adapter.create_agent.return_value = mock_instance
        
        agents = await coordinator._create_agents([sample_agent_config], mock_adapter)
        
        assert len(agents) == 1
        assert agents[0] == mock_instance
        mock_adapter.create_agent.assert_called_once_with(sample_agent_config)

    @pytest.mark.asyncio
    async def test_create_agents_failure(self, coordinator_with_mock_registry, mock_adapter, sample_agent_config):
        """Test agent creation failure."""
        coordinator = coordinator_with_mock_registry
        mock_adapter.create_agent.side_effect = Exception("Creation failed")
        
        with pytest.raises(AgentCreationError):
            await coordinator._create_agents([sample_agent_config], mock_adapter)

    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, coordinator_with_mock_registry, sample_workflow_config, sample_task, sample_execution_context):
        """Test successful workflow execution."""
        coordinator = coordinator_with_mock_registry
        
        mock_result = MultiAgentResult(
            success=True,
            result={"workflow": "completed"},
            workflow_type=WorkflowType.SINGLE
        )
        
        with patch.object(coordinator._workflow_engine, 'execute_workflow') as mock_execute:
            mock_execute.return_value = mock_result
            
            result = await coordinator._execute_workflow(
                sample_workflow_config,
                sample_task,
                [],
                sample_execution_context
            )
            
            assert result == mock_result
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_workflow_failure(self, coordinator_with_mock_registry, sample_workflow_config, sample_task, sample_execution_context):
        """Test workflow execution failure."""
        coordinator = coordinator_with_mock_registry
        
        with patch.object(coordinator._workflow_engine, 'execute_workflow') as mock_execute:
            mock_execute.side_effect = Exception("Workflow failed")
            
            with pytest.raises(WorkflowExecutionError):
                await coordinator._execute_workflow(
                    sample_workflow_config,
                    sample_task,
                    [],
                    sample_execution_context
                )

    @pytest.mark.asyncio
    async def test_execute_task_success(self, coordinator_with_mock_registry, sample_multi_agent_config, sample_task, mock_adapter):
        """Test successful task execution."""
        coordinator = coordinator_with_mock_registry
        
        # Mock agent creation
        mock_instance = Mock(spec=AgentInstance)
        mock_adapter.create_agent.return_value = mock_instance
        
        # Mock workflow execution
        mock_result = MultiAgentResult(
            success=True,
            result={"task": "completed"},
            workflow_type=WorkflowType.HIERARCHICAL
        )
        
        with patch.object(coordinator._workflow_engine, 'execute_workflow') as mock_execute:
            mock_execute.return_value = mock_result
            
            result = await coordinator.execute_task(sample_multi_agent_config, sample_task)
            
            assert result == mock_result
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_task_with_fallback(self, coordinator_with_mock_registry, sample_multi_agent_config, sample_task, mock_adapter):
        """Test task execution with fallback framework."""
        coordinator = coordinator_with_mock_registry
        
        # Setup fallback framework
        fallback_adapter = Mock()
        fallback_adapter.framework_name = "fallback-framework"
        fallback_adapter.is_initialized = True
        fallback_adapter.create_agent = AsyncMock()
        coordinator._registry.register("fallback-framework", fallback_adapter)
        
        # Mock primary framework failure
        mock_adapter.create_agent.side_effect = Exception("Primary failed")
        
        # Mock fallback success
        fallback_instance = Mock(spec=AgentInstance)
        fallback_adapter.create_agent.return_value = fallback_instance
        
        mock_result = MultiAgentResult(
            success=True,
            result={"fallback": "success"},
            workflow_type=WorkflowType.HIERARCHICAL
        )
        
        with patch.object(coordinator._workflow_engine, 'execute_workflow') as mock_execute:
            mock_execute.return_value = mock_result
            
            # First call should fail, second (fallback) should succeed
            with patch.object(coordinator, 'execute_task') as mock_execute_task:
                mock_execute_task.side_effect = [
                    Exception("Primary failed"),  # First call fails
                    mock_result  # Fallback succeeds
                ]
                
                result = await coordinator._handle_execution_error_with_fallback(
                    sample_multi_agent_config,
                    sample_task,
                    Exception("Primary failed"),
                    "test_execution_id"
                )
                
                assert result == mock_result

    @pytest.mark.asyncio
    async def test_execute_task_all_fallbacks_fail(self, coordinator_with_mock_registry, sample_multi_agent_config, sample_task):
        """Test task execution when all fallbacks fail."""
        coordinator = coordinator_with_mock_registry
        
        with patch.object(coordinator, 'execute_task') as mock_execute_task:
            mock_execute_task.side_effect = Exception("All failed")
            
            with pytest.raises(MultiAgentError, match="All frameworks failed"):
                await coordinator._handle_execution_error_with_fallback(
                    sample_multi_agent_config,
                    sample_task,
                    Exception("Original error"),
                    "test_execution_id"
                )

    @pytest.mark.asyncio
    async def test_execute_task_stream(self, coordinator_with_mock_registry, sample_multi_agent_config, sample_task, mock_adapter):
        """Test streaming task execution."""
        coordinator = coordinator_with_mock_registry
        
        # Mock agent creation
        mock_instance = Mock(spec=AgentInstance)
        mock_adapter.create_agent.return_value = mock_instance
        
        # Mock workflow streaming
        async def mock_stream():
            yield {"type": "workflow_started"}
            yield {"type": "workflow_completed"}
        
        with patch.object(coordinator._workflow_engine, 'execute_workflow_stream') as mock_stream_execute:
            mock_stream_execute.return_value = mock_stream()
            
            updates = []
            async for update in coordinator.execute_task_stream(sample_multi_agent_config, sample_task):
                updates.append(update)
            
            assert len(updates) >= 4  # execution_started, config_validated, framework_ready, agents_created, workflow updates, execution_completed
            assert updates[0]["type"] == "execution_started"
            assert updates[-1]["type"] == "execution_completed"

    @pytest.mark.asyncio
    async def test_execute_task_stream_error(self, coordinator_with_mock_registry, sample_multi_agent_config, sample_task, mock_adapter):
        """Test streaming task execution with error."""
        coordinator = coordinator_with_mock_registry
        
        # Mock agent creation failure
        mock_adapter.create_agent.side_effect = Exception("Agent creation failed")
        
        updates = []
        async for update in coordinator.execute_task_stream(sample_multi_agent_config, sample_task):
            updates.append(update)
        
        # Should get error update
        error_updates = [u for u in updates if u.get("type") == "execution_error"]
        assert len(error_updates) > 0

    @pytest.mark.asyncio
    async def test_execute_batch_tasks(self, coordinator_with_mock_registry, sample_multi_agent_config, task_factory):
        """Test batch task execution."""
        coordinator = coordinator_with_mock_registry
        
        # Create multiple tasks
        tasks = [task_factory(f"task_{i}") for i in range(3)]
        
        # Mock individual task execution
        mock_results = [
            MultiAgentResult(success=True, result={"task": i})
            for i in range(3)
        ]
        
        with patch.object(coordinator, 'execute_task') as mock_execute:
            mock_execute.side_effect = mock_results
            
            results = await coordinator.execute_batch_tasks(sample_multi_agent_config, tasks)
            
            assert len(results) == 3
            assert all(result.success for result in results)
            assert mock_execute.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_batch_tasks_with_failures(self, coordinator_with_mock_registry, sample_multi_agent_config, task_factory):
        """Test batch task execution with some failures."""
        coordinator = coordinator_with_mock_registry
        
        # Create multiple tasks
        tasks = [task_factory(f"task_{i}") for i in range(3)]
        
        # Mock mixed results (success, failure, success)
        with patch.object(coordinator, 'execute_task') as mock_execute:
            mock_execute.side_effect = [
                MultiAgentResult(success=True, result={"task": 0}),
                Exception("Task 1 failed"),
                MultiAgentResult(success=True, result={"task": 2})
            ]
            
            results = await coordinator.execute_batch_tasks(sample_multi_agent_config, tasks)
            
            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is False
            assert results[2].success is True
            assert "Task 1 failed" in results[1].error_message

    def test_get_active_executions(self, coordinator_with_mock_registry):
        """Test getting active executions."""
        coordinator = coordinator_with_mock_registry
        
        # Initially empty
        executions = coordinator.get_active_executions()
        assert len(executions) == 0
        
        # Add mock execution
        coordinator._active_executions["test_id"] = {"status": "running"}
        executions = coordinator.get_active_executions()
        assert len(executions) == 1
        assert executions["test_id"]["status"] == "running"

    def test_get_execution_metrics(self, coordinator_with_mock_registry):
        """Test getting execution metrics."""
        coordinator = coordinator_with_mock_registry
        
        # No metrics initially
        metrics = coordinator.get_execution_metrics("test_id")
        assert metrics is None
        
        # Add mock metrics
        mock_metrics = Mock()
        coordinator._execution_metrics["test_id"] = mock_metrics
        metrics = coordinator.get_execution_metrics("test_id")
        assert metrics == mock_metrics

    @pytest.mark.asyncio
    async def test_cancel_execution(self, coordinator_with_mock_registry):
        """Test cancelling execution."""
        coordinator = coordinator_with_mock_registry
        
        # Cancel nonexistent execution
        result = await coordinator.cancel_execution("nonexistent")
        assert result is False
        
        # Add active execution and cancel
        coordinator._active_executions["test_id"] = {"status": "running"}
        result = await coordinator.cancel_execution("test_id")
        assert result is True
        assert coordinator._active_executions["test_id"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cleanup_execution_tracking(self, coordinator_with_mock_registry):
        """Test cleanup of execution tracking."""
        coordinator = coordinator_with_mock_registry
        
        # Add execution data
        coordinator._active_executions["test_id"] = {"status": "completed"}
        coordinator._execution_metrics["test_id"] = Mock()
        
        # Cleanup with no delay for testing
        await coordinator._cleanup_execution_tracking("test_id", delay=0)
        
        assert "test_id" not in coordinator._active_executions
        assert "test_id" not in coordinator._execution_metrics
