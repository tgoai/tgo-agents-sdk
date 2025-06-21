"""
Unit tests for core models.

This module tests the core data models including Task, Agent, AgentConfig,
and other fundamental data structures.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from src.core.models import (
    Task, Agent, AgentConfig, AgentInstance, MultiAgentConfig,
    WorkflowConfig, TaskResult, AgentExecutionResult, MultiAgentResult,
    ExecutionContext, ExecutionMetrics, ToolCallResult, KnowledgeBaseQueryResult
)
from src.core.enums import (
    AgentType, AgentStatus, TaskType, TaskStatus, TaskPriority,
    WorkflowType, ExecutionStrategy, FrameworkCapability
)


class TestTask:
    """Test cases for Task model."""

    def test_task_creation_with_defaults(self):
        """Test creating a task with default values."""
        task = Task(title="Test Task")
        
        assert task.title == "Test Task"
        assert task.task_type == TaskType.SIMPLE
        assert task.priority == TaskPriority.MEDIUM
        assert task.status == TaskStatus.PENDING
        assert task.input_data == {}
        assert task.output_data is None
        assert task.max_retries == 0
        assert task.retry_count == 0
        assert isinstance(task.created_at, datetime)
        assert task.started_at is None
        assert task.completed_at is None

    def test_task_creation_with_custom_values(self):
        """Test creating a task with custom values."""
        input_data = {"key": "value"}
        task = Task(
            title="Custom Task",
            description="Custom description",
            task_type=TaskType.COMPLEX,
            priority=TaskPriority.HIGH,
            input_data=input_data,
            timeout_seconds=120,
            max_retries=3
        )
        
        assert task.title == "Custom Task"
        assert task.description == "Custom description"
        assert task.task_type == TaskType.COMPLEX
        assert task.priority == TaskPriority.HIGH
        assert task.input_data == input_data
        assert task.timeout_seconds == 120
        assert task.max_retries == 3

    def test_task_validation_errors(self):
        """Test task validation errors."""
        # Empty title should fail
        with pytest.raises(ValidationError):
            Task(title="")
        
        # Title too long should fail
        with pytest.raises(ValidationError):
            Task(title="x" * 201)
        
        # Negative timeout should fail
        with pytest.raises(ValidationError):
            Task(title="Test", timeout_seconds=-1)
        
        # Negative max_retries should fail
        with pytest.raises(ValidationError):
            Task(title="Test", max_retries=-1)

    def test_task_status_methods(self):
        """Test task status checking methods."""
        task = Task(title="Test Task")
        
        # Initially pending
        assert not task.is_completed()
        assert not task.is_running()
        assert not task.can_retry()
        
        # Running
        task.status = TaskStatus.RUNNING
        assert not task.is_completed()
        assert task.is_running()
        
        # Completed
        task.status = TaskStatus.COMPLETED
        assert task.is_completed()
        assert not task.is_running()
        
        # Failed with retries available
        task.status = TaskStatus.FAILED
        task.max_retries = 2
        task.retry_count = 1
        assert task.is_completed()
        assert task.can_retry()
        
        # Failed with no retries left
        task.retry_count = 2
        assert not task.can_retry()

    def test_task_execution_duration(self):
        """Test task execution duration calculation."""
        task = Task(title="Test Task")

        # No duration when not started
        assert task.get_execution_duration() is None

        # No duration when started but not completed
        task.started_at = datetime.now(timezone.utc)
        assert task.get_execution_duration() is None

        # Duration when both started and completed
        from datetime import timedelta
        start_time = datetime.now(timezone.utc)
        task.started_at = start_time
        task.completed_at = start_time + timedelta(seconds=5)
        duration = task.get_execution_duration()
        assert duration == 5


class TestAgentConfig:
    """Test cases for AgentConfig model."""

    def test_agent_config_creation(self):
        """Test creating an agent configuration."""
        config = AgentConfig(
            agent_id="test_agent",
            name="Test Agent",
            agent_type=AgentType.EXPERT,
            description="Test description",
            capabilities=["reasoning", "tool_calling"],
            tools=["search", "calculator"],
            knowledge_bases=["kb1", "kb2"]
        )
        
        assert config.agent_id == "test_agent"
        assert config.name == "Test Agent"
        assert config.agent_type == AgentType.EXPERT
        assert config.description == "Test description"
        assert config.model == "gemini-2.0-flash"  # default
        assert config.capabilities == ["reasoning", "tool_calling"]
        assert config.tools == ["search", "calculator"]
        assert config.knowledge_bases == ["kb1", "kb2"]
        assert config.max_iterations == 10  # default
        assert config.temperature == 0.7  # default

    def test_agent_config_validation(self):
        """Test agent configuration validation."""
        # Valid config
        config = AgentConfig(
            agent_id="test",
            name="Test",
            agent_type=AgentType.EXPERT
        )
        assert config.agent_id == "test"
        
        # Invalid max_iterations
        with pytest.raises(ValidationError):
            AgentConfig(
                agent_id="test",
                name="Test",
                agent_type=AgentType.EXPERT,
                max_iterations=0
            )
        
        # Invalid temperature
        with pytest.raises(ValidationError):
            AgentConfig(
                agent_id="test",
                name="Test",
                agent_type=AgentType.EXPERT,
                temperature=3.0
            )


class TestAgentInstance:
    """Test cases for AgentInstance model."""

    def test_agent_instance_creation(self, sample_agent_config):
        """Test creating an agent instance."""
        instance = AgentInstance(
            agent_id=sample_agent_config.agent_id,
            config=sample_agent_config,
            status=AgentStatus.IDLE
        )
        
        assert instance.agent_id == sample_agent_config.agent_id
        assert instance.config == sample_agent_config
        assert instance.status == AgentStatus.IDLE
        assert instance.current_task_id is None
        assert instance.session_data == {}
        assert instance.memory == {}
        assert isinstance(instance.created_at, datetime)

    def test_agent_instance_availability(self, sample_agent_config):
        """Test agent instance availability checking."""
        instance = AgentInstance(
            agent_id=sample_agent_config.agent_id,
            config=sample_agent_config,
            status=AgentStatus.IDLE
        )
        
        # Initially available
        assert instance.is_available()
        
        # Not available when busy
        instance.status = AgentStatus.BUSY
        assert not instance.is_available()
        
        # Not available when in error
        instance.status = AgentStatus.ERROR
        assert not instance.is_available()

    def test_agent_instance_activity_update(self, sample_agent_config):
        """Test updating agent activity timestamp."""
        instance = AgentInstance(
            agent_id=sample_agent_config.agent_id,
            config=sample_agent_config
        )

        # Initially last_activity should be None
        assert instance.last_activity is None

        # Update activity
        instance.update_activity()

        # Now it should have a timestamp
        assert instance.last_activity is not None
        assert isinstance(instance.last_activity, datetime)

        # Update again and verify it changes
        first_update = instance.last_activity
        import time
        time.sleep(0.001)  # Small delay to ensure different timestamp
        instance.update_activity()

        assert instance.last_activity != first_update
        assert instance.last_activity > first_update


class TestWorkflowConfig:
    """Test cases for WorkflowConfig model."""

    def test_workflow_config_creation(self):
        """Test creating a workflow configuration."""
        config = WorkflowConfig(
            workflow_type=WorkflowType.HIERARCHICAL,
            execution_strategy=ExecutionStrategy.FAIL_FAST,
            manager_agent_id="manager_001",
            expert_agent_ids=["expert_001", "expert_002"],
            max_concurrent_agents=3,
            timeout_seconds=300
        )
        
        assert config.workflow_type == WorkflowType.HIERARCHICAL
        assert config.execution_strategy == ExecutionStrategy.FAIL_FAST
        assert config.manager_agent_id == "manager_001"
        assert config.expert_agent_ids == ["expert_001", "expert_002"]
        assert config.max_concurrent_agents == 3
        assert config.timeout_seconds == 300

    def test_workflow_config_defaults(self):
        """Test workflow configuration defaults."""
        config = WorkflowConfig(workflow_type=WorkflowType.SINGLE)
        
        assert config.workflow_type == WorkflowType.SINGLE
        assert config.execution_strategy == ExecutionStrategy.FAIL_FAST
        assert config.manager_agent_id is None
        assert config.expert_agent_ids == []
        assert config.max_concurrent_agents == 3


class TestMultiAgentConfig:
    """Test cases for MultiAgentConfig model."""

    def test_multi_agent_config_creation(self, sample_agent_config, sample_workflow_config):
        """Test creating a multi-agent configuration."""
        config = MultiAgentConfig(
            framework="test-framework",
            fallback_frameworks=["fallback1", "fallback2"],
            agents=[sample_agent_config],
            workflow=sample_workflow_config,
            tenant_id="test_tenant"
        )
        
        assert config.framework == "test-framework"
        assert config.fallback_frameworks == ["fallback1", "fallback2"]
        assert len(config.agents) == 1
        assert config.agents[0] == sample_agent_config
        assert config.workflow == sample_workflow_config
        assert config.tenant_id == "test_tenant"

    def test_multi_agent_config_validation(self, sample_workflow_config):
        """Test multi-agent configuration validation."""
        # Must have at least one agent
        with pytest.raises(ValidationError):
            MultiAgentConfig(
                framework="test",
                agents=[],
                workflow=sample_workflow_config
            )


class TestExecutionResults:
    """Test cases for execution result models."""

    def test_task_result_creation(self):
        """Test creating a task result."""
        result = TaskResult(
            task_id="test_task",
            success=True,
            result={"output": "test"},
            execution_time_ms=1000,
            agent_id="test_agent",
            agent_type=AgentType.EXPERT
        )
        
        assert result.task_id == "test_task"
        assert result.success is True
        assert result.result == {"output": "test"}
        assert result.execution_time_ms == 1000
        assert result.agent_id == "test_agent"
        assert result.agent_type == AgentType.EXPERT
        assert result.is_successful()

    def test_agent_execution_result_creation(self):
        """Test creating an agent execution result."""
        result = AgentExecutionResult(
            success=True,
            result={"response": "test"},
            execution_time_ms=1500,
            tool_calls=[],
            kb_queries=[],
            reasoning_steps=["step1", "step2"]
        )
        
        assert result.success is True
        assert result.result == {"response": "test"}
        assert result.execution_time_ms == 1500
        assert result.reasoning_steps == ["step1", "step2"]
        assert result.is_successful()
        assert not result.has_tool_calls()
        assert not result.has_kb_queries()

    def test_multi_agent_result_methods(self):
        """Test multi-agent result methods."""
        agent_results = {
            "agent1": AgentExecutionResult(success=True, result={"data": "test1"}),
            "agent2": AgentExecutionResult(success=False, error_message="Failed"),
            "agent3": AgentExecutionResult(success=True, result={"data": "test3"})
        }
        
        result = MultiAgentResult(
            success=True,
            result={"final": "result"},
            agent_results=agent_results,
            workflow_type=WorkflowType.PARALLEL,
            agents_used=["agent1", "agent2", "agent3"]
        )
        
        assert result.is_successful()
        assert result.get_successful_agents() == ["agent1", "agent3"]
        assert result.get_failed_agents() == ["agent2"]
