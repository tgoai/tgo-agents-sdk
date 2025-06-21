"""
Integration tests for the multi-agent system.

This module tests the integration between different components
of the multi-agent system.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.registry.adapter_registry import AdapterRegistry
from src.coordinator.multi_agent_coordinator import MultiAgentCoordinator
from src.adapters.google_adk_adapter import GoogleADKAdapter
from src.core.models import (
    Task, AgentConfig, MultiAgentConfig, WorkflowConfig,
    MultiAgentResult, AgentExecutionResult, TaskResult
)
from src.core.enums import WorkflowType, ExecutionStrategy, AgentType


class TestIntegration:
    """Integration test cases."""

    @pytest.mark.asyncio
    async def test_end_to_end_single_agent_execution(self):
        """Test end-to-end single agent execution."""
        # Setup registry and adapter
        registry = AdapterRegistry()
        adapter = GoogleADKAdapter()
        registry.register("google-adk", adapter, is_default=True)
        
        # Initialize adapter
        await registry.initialize_adapter("google-adk")
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(registry)
        
        # Create configuration
        agent_config = AgentConfig(
            agent_id="test_agent_001",
            name="Test Agent",
            agent_type=AgentType.EXPERT,
            model="gemini-2.0-flash",
            capabilities=["reasoning"]
        )
        
        workflow_config = WorkflowConfig(
            workflow_type=WorkflowType.SINGLE,
            execution_strategy=ExecutionStrategy.FAIL_FAST
        )
        
        multi_agent_config = MultiAgentConfig(
            framework="google-adk",
            agents=[agent_config],
            workflow=workflow_config
        )
        
        # Create task
        task = Task(
            title="Test Task",
            description="A simple test task",
            input_data={"test": "data"}
        )
        
        # Execute task
        result = await coordinator.execute_task(multi_agent_config, task)
        
        # Verify result
        assert isinstance(result, MultiAgentResult)
        assert result.success is True
        assert result.workflow_type == WorkflowType.SINGLE
        assert len(result.agents_used) == 1
        assert "test_agent_001" in result.agents_used
        
        # Cleanup
        await registry.cleanup_all()

    @pytest.mark.asyncio
    async def test_end_to_end_hierarchical_execution(self):
        """Test end-to-end hierarchical workflow execution."""
        # Setup registry and adapter
        registry = AdapterRegistry()
        adapter = GoogleADKAdapter()
        registry.register("google-adk", adapter, is_default=True)
        
        # Initialize adapter
        await registry.initialize_adapter("google-adk")
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(registry)
        
        # Create configuration with manager and expert
        manager_config = AgentConfig(
            agent_id="manager_001",
            name="Task Manager",
            agent_type=AgentType.MANAGER,
            model="gemini-2.0-flash",
            capabilities=["reasoning", "delegation"]
        )
        
        expert_config = AgentConfig(
            agent_id="expert_001",
            name="Research Expert",
            agent_type=AgentType.EXPERT,
            model="gemini-2.0-flash",
            capabilities=["research", "analysis"]
        )
        
        workflow_config = WorkflowConfig(
            workflow_type=WorkflowType.HIERARCHICAL,
            execution_strategy=ExecutionStrategy.FAIL_FAST,
            manager_agent_id="manager_001",
            expert_agent_ids=["expert_001"]
        )
        
        multi_agent_config = MultiAgentConfig(
            framework="google-adk",
            agents=[manager_config, expert_config],
            workflow=workflow_config
        )
        
        # Create task
        task = Task(
            title="Research AI Trends",
            description="Research the latest trends in AI development",
            input_data={"topic": "artificial intelligence"}
        )
        
        # Execute task
        result = await coordinator.execute_task(multi_agent_config, task)
        
        # Verify result
        assert isinstance(result, MultiAgentResult)
        assert result.success is True
        assert result.workflow_type == WorkflowType.HIERARCHICAL
        assert len(result.agents_used) == 2
        assert "manager_001" in result.agents_used
        assert "expert_001" in result.agents_used
        
        # Cleanup
        await registry.cleanup_all()

    @pytest.mark.asyncio
    async def test_framework_fallback_mechanism(self):
        """Test framework fallback mechanism."""
        # Setup registry with primary and fallback adapters
        registry = AdapterRegistry()

        # Create a proper primary adapter that will fail
        from src.adapters.base_adapter import BaseFrameworkAdapter
        from src.core.models import ToolCallResult, KnowledgeBaseQueryResult
        from datetime import datetime, timezone

        class FailingAdapter(BaseFrameworkAdapter):
            def __init__(self):
                super().__init__("primary-framework", "1.0.0")

            async def _create_framework_agent(self, config):
                raise Exception("Primary failed")

            async def _execute_framework_task(self, framework_agent, task, context):
                raise Exception("Primary failed")

            async def call_tool(self, agent_id, tool_id, tool_name, parameters, context):
                return ToolCallResult(
                    tool_name=tool_name,
                    tool_id=tool_id,
                    success=False,
                    error_message="Primary failed",
                    execution_time_ms=0
                )

            async def query_knowledge_base(self, agent_id, kb_id, kb_name, query, parameters=None, context=None):
                return KnowledgeBaseQueryResult(
                    kb_name=kb_name,
                    kb_id=kb_id,
                    query=query,
                    success=False,
                    results=[],
                    results_count=0,
                    error_message="Primary failed",
                    execution_time_ms=0
                )

        primary_adapter = FailingAdapter()

        # Fallback adapter that will succeed
        fallback_adapter = GoogleADKAdapter()

        registry.register("primary-framework", primary_adapter, is_default=True)
        registry.register("google-adk", fallback_adapter)
        
        # Initialize adapters
        await registry.initialize_all()
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(registry)
        
        # Create configuration with fallback
        agent_config = AgentConfig(
            agent_id="test_agent_001",
            name="Test Agent",
            agent_type=AgentType.EXPERT,
            model="gemini-2.0-flash"
        )
        
        workflow_config = WorkflowConfig(
            workflow_type=WorkflowType.SINGLE,
            execution_strategy=ExecutionStrategy.FAIL_FAST
        )
        
        multi_agent_config = MultiAgentConfig(
            framework="primary-framework",
            fallback_frameworks=["google-adk"],
            agents=[agent_config],
            workflow=workflow_config
        )
        
        # Create task
        task = Task(
            title="Test Task",
            description="A test task for fallback testing"
        )
        
        # Mock the fallback execution to succeed
        with patch.object(coordinator, 'execute_task') as mock_execute:
            # First call (primary) fails, second call (fallback) succeeds
            mock_execute.side_effect = [
                Exception("Primary framework failed"),
                MultiAgentResult(
                    success=True,
                    result={"fallback": "success"},
                    workflow_type=WorkflowType.SINGLE,
                    agents_used=["test_agent_001"]
                )
            ]
            
            result = await coordinator._handle_execution_error_with_fallback(
                multi_agent_config,
                task,
                Exception("Primary framework failed"),
                "test_execution_id"
            )
            
            assert result.success is True
            assert result.result["fallback"] == "success"
        
        # Cleanup
        await registry.cleanup_all()

    @pytest.mark.asyncio
    async def test_streaming_execution_integration(self):
        """Test streaming execution integration."""
        # Setup registry and adapter
        registry = AdapterRegistry()
        adapter = GoogleADKAdapter()
        registry.register("google-adk", adapter, is_default=True)
        
        # Initialize adapter
        await registry.initialize_adapter("google-adk")
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(registry)
        
        # Create configuration
        agent_config = AgentConfig(
            agent_id="stream_agent_001",
            name="Streaming Agent",
            agent_type=AgentType.EXPERT,
            model="gemini-2.0-flash"
        )
        
        workflow_config = WorkflowConfig(
            workflow_type=WorkflowType.SINGLE,
            execution_strategy=ExecutionStrategy.FAIL_FAST
        )
        
        multi_agent_config = MultiAgentConfig(
            framework="google-adk",
            agents=[agent_config],
            workflow=workflow_config
        )
        
        # Create task
        task = Task(
            title="Streaming Test Task",
            description="A task for testing streaming execution"
        )
        
        # Execute task with streaming
        updates = []
        async for update in coordinator.execute_task_stream(multi_agent_config, task):
            updates.append(update)
        
        # Verify streaming updates
        assert len(updates) > 0
        
        # Check for expected update types
        update_types = [update.get("type") for update in updates]
        assert "execution_started" in update_types
        assert "config_validated" in update_types
        assert "framework_ready" in update_types
        assert "agents_created" in update_types
        assert "execution_completed" in update_types
        
        # Cleanup
        await registry.cleanup_all()

    @pytest.mark.asyncio
    async def test_batch_execution_integration(self):
        """Test batch execution integration."""
        # Setup registry and adapter
        registry = AdapterRegistry()
        adapter = GoogleADKAdapter()
        registry.register("google-adk", adapter, is_default=True)
        
        # Initialize adapter
        await registry.initialize_adapter("google-adk")
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(registry)
        
        # Create configuration
        agent_config = AgentConfig(
            agent_id="batch_agent_001",
            name="Batch Agent",
            agent_type=AgentType.EXPERT,
            model="gemini-2.0-flash"
        )
        
        workflow_config = WorkflowConfig(
            workflow_type=WorkflowType.SINGLE,
            execution_strategy=ExecutionStrategy.FAIL_FAST
        )
        
        multi_agent_config = MultiAgentConfig(
            framework="google-adk",
            agents=[agent_config],
            workflow=workflow_config
        )
        
        # Create multiple tasks
        tasks = [
            Task(title=f"Batch Task {i}", description=f"Task {i} for batch testing")
            for i in range(3)
        ]
        
        # Execute tasks in batch
        results = await coordinator.execute_batch_tasks(multi_agent_config, tasks)
        
        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, MultiAgentResult)
            assert result.success is True
            assert len(result.agents_used) == 1
            assert "batch_agent_001" in result.agents_used
        
        # Cleanup
        await registry.cleanup_all()

    @pytest.mark.asyncio
    async def test_adapter_lifecycle_management(self):
        """Test adapter lifecycle management."""
        # Create registry
        registry = AdapterRegistry()
        
        # Create multiple adapters
        adapter1 = GoogleADKAdapter()
        adapter2 = GoogleADKAdapter()
        
        # Register adapters
        registry.register("adapter1", adapter1)
        registry.register("adapter2", adapter2)
        
        # Verify initial state
        assert not adapter1.is_initialized
        assert not adapter2.is_initialized
        
        # Initialize all adapters
        results = await registry.initialize_all()
        assert all(results.values())
        assert adapter1.is_initialized
        assert adapter2.is_initialized
        
        # Get health status
        status = registry.get_health_status()
        assert status["total_adapters"] == 2
        assert status["initialized_adapters"] == 2
        
        # Cleanup all adapters
        cleanup_results = await registry.cleanup_all()
        assert all(cleanup_results.values())
        assert not adapter1.is_initialized
        assert not adapter2.is_initialized

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling integration across components."""
        # Setup registry with failing adapter
        registry = AdapterRegistry()

        # Create a proper failing adapter
        from src.adapters.base_adapter import BaseFrameworkAdapter
        from src.core.models import ToolCallResult, KnowledgeBaseQueryResult

        class FailingAdapter(BaseFrameworkAdapter):
            def __init__(self):
                super().__init__("failing-framework", "1.0.0")

            async def _create_framework_agent(self, config):
                raise Exception("Agent creation failed")

            async def _execute_framework_task(self, framework_agent, task, context):
                raise Exception("Agent creation failed")

            async def call_tool(self, agent_id, tool_id, tool_name, parameters, context):
                return ToolCallResult(
                    tool_name=tool_name,
                    tool_id=tool_id,
                    success=False,
                    error_message="Agent creation failed",
                    execution_time_ms=0
                )

            async def query_knowledge_base(self, agent_id, kb_id, kb_name, query, parameters=None, context=None):
                return KnowledgeBaseQueryResult(
                    kb_name=kb_name,
                    kb_id=kb_id,
                    query=query,
                    success=False,
                    results=[],
                    results_count=0,
                    error_message="Agent creation failed",
                    execution_time_ms=0
                )

        failing_adapter = FailingAdapter()

        registry.register("failing-framework", failing_adapter, is_default=True)
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(registry)
        
        # Create configuration
        agent_config = AgentConfig(
            agent_id="failing_agent_001",
            name="Failing Agent",
            agent_type=AgentType.EXPERT,
            model="test-model"
        )
        
        workflow_config = WorkflowConfig(
            workflow_type=WorkflowType.SINGLE,
            execution_strategy=ExecutionStrategy.FAIL_FAST
        )
        
        multi_agent_config = MultiAgentConfig(
            framework="failing-framework",
            agents=[agent_config],
            workflow=workflow_config
        )
        
        # Create task
        task = Task(
            title="Failing Task",
            description="A task that will fail"
        )
        
        # Execute task and expect failure
        with pytest.raises(Exception):
            await coordinator.execute_task(multi_agent_config, task)

    @pytest.mark.asyncio
    async def test_concurrent_execution_safety(self):
        """Test concurrent execution safety."""
        # Setup registry and adapter
        registry = AdapterRegistry()
        adapter = GoogleADKAdapter()
        registry.register("google-adk", adapter, is_default=True)
        
        # Initialize adapter
        await registry.initialize_adapter("google-adk")
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(registry)
        
        # Create configuration
        agent_config = AgentConfig(
            agent_id="concurrent_agent_001",
            name="Concurrent Agent",
            agent_type=AgentType.EXPERT,
            model="gemini-2.0-flash"
        )
        
        workflow_config = WorkflowConfig(
            workflow_type=WorkflowType.SINGLE,
            execution_strategy=ExecutionStrategy.FAIL_FAST
        )
        
        multi_agent_config = MultiAgentConfig(
            framework="google-adk",
            agents=[agent_config],
            workflow=workflow_config
        )
        
        # Create multiple tasks for concurrent execution
        tasks = [
            Task(title=f"Concurrent Task {i}", description=f"Task {i} for concurrent testing")
            for i in range(5)
        ]
        
        # Execute tasks concurrently
        import asyncio
        execution_tasks = [
            coordinator.execute_task(multi_agent_config, task)
            for task in tasks
        ]
        
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Verify all executions completed
        assert len(results) == 5
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent execution failed: {result}")
            assert isinstance(result, MultiAgentResult)
            assert result.success is True
        
        # Cleanup
        await registry.cleanup_all()
