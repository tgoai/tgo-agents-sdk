"""
Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timezone

from src.core.models import (
    Task, Agent, AgentConfig, AgentInstance, MultiAgentConfig,
    WorkflowConfig, TaskResult, AgentExecutionResult, MultiAgentResult,
    ExecutionContext
)
from src.core.enums import (
    AgentType, AgentStatus, TaskType, TaskStatus, TaskPriority,
    WorkflowType, ExecutionStrategy, FrameworkCapability
)
from src.registry.adapter_registry import AdapterRegistry
from src.coordinator.multi_agent_coordinator import MultiAgentCoordinator


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for testing."""
    return Task(
        task_id="test_task_001",
        title="Test Task",
        description="A test task for unit testing",
        task_type=TaskType.SIMPLE,
        priority=TaskPriority.MEDIUM,
        input_data={"test_input": "test_value"}
    )


@pytest.fixture
def sample_agent_config() -> AgentConfig:
    """Create a sample agent configuration."""
    return AgentConfig(
        agent_id="test_agent_001",
        name="Test Agent",
        agent_type=AgentType.EXPERT,
        description="A test agent for unit testing",
        model="gemini-2.0-flash",
        capabilities=["reasoning", "tool_calling"],
        tools=["test_tool"],
        knowledge_bases=["test_kb"],
        max_iterations=5,
        timeout_seconds=60,
        temperature=0.7
    )


@pytest.fixture
def sample_manager_config() -> AgentConfig:
    """Create a sample manager agent configuration."""
    return AgentConfig(
        agent_id="manager_001",
        name="Test Manager",
        agent_type=AgentType.MANAGER,
        description="A test manager agent",
        model="gemini-2.0-flash",
        capabilities=["reasoning", "delegation"],
        max_iterations=10,
        timeout_seconds=120,
        temperature=0.5
    )


@pytest.fixture
def sample_agent_instance(sample_agent_config: AgentConfig) -> AgentInstance:
    """Create a sample agent instance."""
    return AgentInstance(
        agent_id=sample_agent_config.agent_id,
        config=sample_agent_config,
        status=AgentStatus.IDLE,
        created_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_workflow_config() -> WorkflowConfig:
    """Create a sample workflow configuration."""
    return WorkflowConfig(
        workflow_type=WorkflowType.HIERARCHICAL,
        execution_strategy=ExecutionStrategy.FAIL_FAST,
        manager_agent_id="manager_001",
        expert_agent_ids=["expert_001", "expert_002"],
        max_concurrent_agents=2,
        timeout_seconds=300
    )


@pytest.fixture
def sample_multi_agent_config(
    sample_agent_config: AgentConfig,
    sample_manager_config: AgentConfig,
    sample_workflow_config: WorkflowConfig
) -> MultiAgentConfig:
    """Create a sample multi-agent configuration."""
    return MultiAgentConfig(
        framework="test-framework",
        fallback_frameworks=["fallback-framework"],
        agents=[sample_manager_config, sample_agent_config],
        workflow=sample_workflow_config
    )


@pytest.fixture
def sample_execution_context() -> ExecutionContext:
    """Create a sample execution context."""
    return ExecutionContext(
        task_id="test_task_001",
        agent_id="test_agent_001",
        framework_name="test-framework",
        workflow_type=WorkflowType.SINGLE,
        tenant_id="test_tenant",
        user_id="test_user",
        session_id="test_session"
    )


@pytest.fixture
def sample_task_result() -> TaskResult:
    """Create a sample task result."""
    return TaskResult(
        task_id="test_task_001",
        success=True,
        result={"output": "test result"},
        execution_time_ms=1000,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        agent_id="test_agent_001",
        agent_type=AgentType.EXPERT
    )


@pytest.fixture
def sample_agent_execution_result() -> AgentExecutionResult:
    """Create a sample agent execution result."""
    return AgentExecutionResult(
        success=True,
        result={"response": "test response"},
        execution_time_ms=1500,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        tool_calls=[],
        kb_queries=[],
        reasoning_steps=["Step 1: Analyze", "Step 2: Execute"],
        intermediate_results=[]
    )


@pytest.fixture
def mock_adapter():
    """Create a mock framework adapter."""
    from src.adapters.base_adapter import BaseFrameworkAdapter
    from unittest.mock import AsyncMock

    class MockFrameworkAdapter(BaseFrameworkAdapter):
        def __init__(self):
            super().__init__("test-framework", "1.0.0")
            self._capabilities = [
                FrameworkCapability.SINGLE_AGENT,
                FrameworkCapability.MULTI_AGENT,
                FrameworkCapability.TOOL_CALLING
            ]

        async def _create_framework_agent(self, config):
            return Mock()

        async def _execute_framework_task(self, framework_agent, task, context):
            from src.core.models import AgentExecutionResult
            from datetime import datetime, timezone
            return AgentExecutionResult(
                success=True,
                result={"response": "Mock execution result"},
                execution_time_ms=100,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc)
            )

        async def call_tool(self, agent_id, tool_id, tool_name, parameters, context):
            from src.core.models import ToolCallResult
            return ToolCallResult(
                tool_name=tool_name,
                tool_id=tool_id,
                success=True,
                result={"mock": "tool result"},
                execution_time_ms=50
            )

        async def query_knowledge_base(self, agent_id, kb_id, kb_name, query, parameters=None, context=None):
            from src.core.models import KnowledgeBaseQueryResult
            return KnowledgeBaseQueryResult(
                kb_name=kb_name,
                kb_id=kb_id,
                query=query,
                success=True,
                results=[{"content": "Mock KB result", "score": 0.9}],
                results_count=1,
                execution_time_ms=30
            )

    return MockFrameworkAdapter()


@pytest.fixture
def registry_with_mock_adapter(mock_adapter):
    """Create a registry with a mock adapter."""
    registry = AdapterRegistry()
    registry.register("test-framework", mock_adapter, is_default=True)
    return registry


@pytest.fixture
def coordinator_with_mock_registry(registry_with_mock_adapter):
    """Create a coordinator with mock registry."""
    return MultiAgentCoordinator(registry_with_mock_adapter)


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Test data generators
@pytest.fixture
def task_factory():
    """Factory for creating test tasks."""
    def _create_task(
        task_id: str = None,
        title: str = "Test Task",
        task_type: TaskType = TaskType.SIMPLE,
        priority: TaskPriority = TaskPriority.MEDIUM,
        input_data: Dict[str, Any] = None
    ) -> Task:
        return Task(
            task_id=task_id or f"task_{datetime.now().timestamp()}",
            title=title,
            description=f"Description for {title}",
            task_type=task_type,
            priority=priority,
            input_data=input_data or {}
        )
    return _create_task


@pytest.fixture
def agent_config_factory():
    """Factory for creating test agent configurations."""
    def _create_agent_config(
        agent_id: str = None,
        name: str = "Test Agent",
        agent_type: AgentType = AgentType.EXPERT,
        capabilities: List[str] = None
    ) -> AgentConfig:
        return AgentConfig(
            agent_id=agent_id or f"agent_{datetime.now().timestamp()}",
            name=name,
            agent_type=agent_type,
            model="gemini-2.0-flash",
            capabilities=capabilities or ["reasoning"],
            tools=[],
            knowledge_bases=[]
        )
    return _create_agent_config
