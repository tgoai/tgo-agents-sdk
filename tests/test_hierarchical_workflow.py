"""
Unit tests for hierarchical workflow.

This module tests the HierarchicalWorkflow class and its manager-expert
coordination pattern.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from src.workflows.hierarchical_workflow import HierarchicalWorkflow
from src.core.models import (
    Task, AgentInstance, WorkflowConfig, MultiAgentResult,
    ExecutionContext, TaskResult, AgentExecutionResult
)
from src.core.enums import WorkflowType, ExecutionStrategy, AgentType
from src.core.exceptions import WorkflowError, WorkflowExecutionError


class TestHierarchicalWorkflow:
    """Test cases for HierarchicalWorkflow."""

    def test_workflow_initialization(self):
        """Test workflow initialization."""
        workflow = HierarchicalWorkflow()
        
        assert workflow.workflow_type == WorkflowType.HIERARCHICAL
        assert workflow._task_executor is not None

    def test_validate_agents_success(self, sample_workflow_config):
        """Test successful agent validation."""
        workflow = HierarchicalWorkflow()
        
        # Create manager and expert agents
        manager_config = Mock()
        manager_config.agent_type = AgentType.MANAGER
        manager_agent = Mock()
        manager_agent.config = manager_config
        
        expert_config = Mock()
        expert_config.agent_type = AgentType.EXPERT
        expert_agent = Mock()
        expert_agent.config = expert_config
        
        agents = [manager_agent, expert_agent]
        
        # Should not raise exception
        workflow.validate_agents(agents, sample_workflow_config)

    def test_validate_agents_no_manager(self, sample_workflow_config):
        """Test agent validation with no manager."""
        workflow = HierarchicalWorkflow()
        
        # Only expert agents
        expert_config = Mock()
        expert_config.agent_type = AgentType.EXPERT
        expert_agent = Mock()
        expert_agent.config = expert_config
        
        agents = [expert_agent]
        
        with pytest.raises(WorkflowError, match="requires at least one manager agent"):
            workflow.validate_agents(agents, sample_workflow_config)

    def test_validate_agents_no_experts(self, sample_workflow_config):
        """Test agent validation with no experts."""
        workflow = HierarchicalWorkflow()
        
        # Only manager agent
        manager_config = Mock()
        manager_config.agent_type = AgentType.MANAGER
        manager_agent = Mock()
        manager_agent.config = manager_config
        
        agents = [manager_agent]
        
        with pytest.raises(WorkflowError, match="requires at least one expert agent"):
            workflow.validate_agents(agents, sample_workflow_config)

    def test_validate_agents_multiple_managers_with_config(self, sample_workflow_config):
        """Test agent validation with multiple managers and specific config."""
        workflow = HierarchicalWorkflow()
        
        # Create multiple managers
        manager1_config = Mock()
        manager1_config.agent_type = AgentType.MANAGER
        manager1 = Mock()
        manager1.agent_id = "manager_001"
        manager1.config = manager1_config
        
        manager2_config = Mock()
        manager2_config.agent_type = AgentType.MANAGER
        manager2 = Mock()
        manager2.agent_id = "manager_002"
        manager2.config = manager2_config
        
        expert_config = Mock()
        expert_config.agent_type = AgentType.EXPERT
        expert_agent = Mock()
        expert_agent.config = expert_config
        
        agents = [manager1, manager2, expert_agent]
        sample_workflow_config.manager_agent_id = "manager_001"
        
        # Should not raise exception
        workflow.validate_agents(agents, sample_workflow_config)

    def test_validate_agents_multiple_managers_wrong_config(self, sample_workflow_config):
        """Test agent validation with multiple managers and wrong config."""
        workflow = HierarchicalWorkflow()
        
        # Create multiple managers
        manager1_config = Mock()
        manager1_config.agent_type = AgentType.MANAGER
        manager1 = Mock()
        manager1.agent_id = "manager_001"
        manager1.config = manager1_config
        
        manager2_config = Mock()
        manager2_config.agent_type = AgentType.MANAGER
        manager2 = Mock()
        manager2.agent_id = "manager_002"
        manager2.config = manager2_config
        
        expert_config = Mock()
        expert_config.agent_type = AgentType.EXPERT
        expert_agent = Mock()
        expert_agent.config = expert_config
        
        agents = [manager1, manager2, expert_agent]
        sample_workflow_config.manager_agent_id = "nonexistent_manager"
        
        with pytest.raises(WorkflowError, match="Configured manager agent .* not found"):
            workflow.validate_agents(agents, sample_workflow_config)

    def test_identify_agents_with_config(self, sample_workflow_config):
        """Test agent identification with specific configuration."""
        workflow = HierarchicalWorkflow()
        
        # Create agents
        manager_config = Mock()
        manager_config.agent_type = AgentType.MANAGER
        manager_agent = Mock()
        manager_agent.agent_id = "manager_001"
        manager_agent.config = manager_config
        
        expert1_config = Mock()
        expert1_config.agent_type = AgentType.EXPERT
        expert1 = Mock()
        expert1.agent_id = "expert_001"
        expert1.config = expert1_config
        
        expert2_config = Mock()
        expert2_config.agent_type = AgentType.EXPERT
        expert2 = Mock()
        expert2.agent_id = "expert_002"
        expert2.config = expert2_config
        
        agents = [manager_agent, expert1, expert2]
        sample_workflow_config.manager_agent_id = "manager_001"
        sample_workflow_config.expert_agent_ids = ["expert_001"]
        
        manager, experts = workflow._identify_agents(agents, sample_workflow_config)
        
        assert manager == manager_agent
        assert len(experts) == 1
        assert experts[0] == expert1

    def test_identify_agents_without_config(self, sample_workflow_config):
        """Test agent identification without specific configuration."""
        workflow = HierarchicalWorkflow()
        
        # Create agents
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
        sample_workflow_config.manager_agent_id = None
        sample_workflow_config.expert_agent_ids = []
        
        manager, experts = workflow._identify_agents(agents, sample_workflow_config)
        
        assert manager == manager_agent
        assert len(experts) == 1
        assert experts[0] == expert_agent

    @pytest.mark.asyncio
    async def test_manager_decompose_task(self, sample_task, sample_execution_context):
        """Test manager task decomposition."""
        workflow = HierarchicalWorkflow()
        
        # Create manager agent
        manager_config = Mock()
        manager_config.agent_type = AgentType.MANAGER
        manager_agent = Mock()
        manager_agent.agent_id = "manager_001"
        manager_agent.config = manager_config
        
        # Create expert agents
        expert_config = Mock()
        expert_config.agent_type = AgentType.EXPERT
        expert_config.name = "Expert Agent"
        expert_config.capabilities = ["reasoning"]
        expert_config.tools = ["search"]
        expert_agent = Mock()
        expert_agent.agent_id = "expert_001"
        expert_agent.config = expert_config
        
        expert_agents = [expert_agent]
        
        # Mock task executor
        mock_result = TaskResult(
            task_id="decomp_task",
            success=True,
            result={"plan": "Task decomposed successfully"}
        )
        
        with patch.object(workflow._task_executor, 'execute') as mock_execute:
            mock_execute.return_value = mock_result
            
            result = await workflow._manager_decompose_task(
                manager_agent,
                sample_task,
                expert_agents,
                sample_execution_context
            )
            
            assert result == mock_result
            mock_execute.assert_called_once()
            
            # Check that the decomposition task was created properly
            call_args = mock_execute.call_args
            decomp_task = call_args[0][0]
            assert "Analyze and decompose" in decomp_task.title
            assert sample_task.title in decomp_task.description

    @pytest.mark.asyncio
    async def test_execute_expert_tasks_fail_fast(self, sample_task, sample_execution_context):
        """Test executing expert tasks with fail fast strategy."""
        workflow = HierarchicalWorkflow()
        
        # Create expert agents
        expert1 = Mock()
        expert1.agent_id = "expert_001"
        expert1.config = Mock()
        expert1.config.name = "Expert 1"
        expert1.config.capabilities = ["reasoning"]
        
        expert2 = Mock()
        expert2.agent_id = "expert_002"
        expert2.config = Mock()
        expert2.config.name = "Expert 2"
        expert2.config.capabilities = ["analysis"]
        
        expert_agents = [expert1, expert2]
        
        # Mock task executor - first succeeds, second fails
        with patch.object(workflow._task_executor, 'execute') as mock_execute:
            mock_execute.side_effect = [
                TaskResult(task_id="expert1", success=True, result={"expert1": "done"}),
                TaskResult(task_id="expert2", success=False, error_message="Expert 2 failed")
            ]
            
            results = await workflow._execute_expert_tasks(
                expert_agents,
                sample_task,
                sample_execution_context,
                ExecutionStrategy.FAIL_FAST
            )
            
            assert len(results) == 2
            assert results["expert_001"].success is True
            assert results["expert_002"].success is False
            assert mock_execute.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_expert_tasks_parallel(self, sample_task, sample_execution_context):
        """Test executing expert tasks in parallel."""
        workflow = HierarchicalWorkflow()
        
        # Create expert agents
        expert1 = Mock()
        expert1.agent_id = "expert_001"
        expert1.config = Mock()
        expert1.config.name = "Expert 1"
        expert1.config.capabilities = ["reasoning"]
        
        expert2 = Mock()
        expert2.agent_id = "expert_002"
        expert2.config = Mock()
        expert2.config.name = "Expert 2"
        expert2.config.capabilities = ["analysis"]
        
        expert_agents = [expert1, expert2]
        
        # Mock task executor
        with patch.object(workflow._task_executor, 'execute') as mock_execute:
            mock_execute.side_effect = [
                TaskResult(task_id="expert1", success=True, result={"expert1": "done"}),
                TaskResult(task_id="expert2", success=True, result={"expert2": "done"})
            ]
            
            results = await workflow._execute_expert_tasks(
                expert_agents,
                sample_task,
                sample_execution_context,
                ExecutionStrategy.CONTINUE_ON_FAILURE
            )
            
            assert len(results) == 2
            assert results["expert_001"].success is True
            assert results["expert_002"].success is True
            assert mock_execute.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_expert_tasks_with_exception(self, sample_task, sample_execution_context):
        """Test executing expert tasks with exception."""
        workflow = HierarchicalWorkflow()
        
        # Create expert agents
        expert1 = Mock()
        expert1.agent_id = "expert_001"
        expert1.config = Mock()
        expert1.config.name = "Expert 1"
        expert1.config.capabilities = ["reasoning"]
        
        expert2 = Mock()
        expert2.agent_id = "expert_002"
        expert2.config = Mock()
        expert2.config.name = "Expert 2"
        expert2.config.capabilities = ["analysis"]
        
        expert_agents = [expert1, expert2]
        
        # Mock task executor - one raises exception
        with patch.object(workflow._task_executor, 'execute') as mock_execute:
            mock_execute.side_effect = [
                TaskResult(task_id="expert1", success=True, result={"expert1": "done"}),
                Exception("Expert 2 crashed")
            ]
            
            results = await workflow._execute_expert_tasks(
                expert_agents,
                sample_task,
                sample_execution_context,
                ExecutionStrategy.CONTINUE_ON_FAILURE
            )
            
            assert len(results) == 2
            assert results["expert_001"].success is True
            assert results["expert_002"].success is False
            assert "Expert 2 crashed" in results["expert_002"].error_message

    @pytest.mark.asyncio
    async def test_manager_synthesize_results(self, sample_task, sample_execution_context):
        """Test manager result synthesis."""
        workflow = HierarchicalWorkflow()
        
        # Create manager agent
        manager_agent = Mock()
        manager_agent.agent_id = "manager_001"
        
        # Create expert results
        expert_results = {
            "expert_001": AgentExecutionResult(
                success=True,
                result={"analysis": "Expert 1 analysis"}
            ),
            "expert_002": AgentExecutionResult(
                success=True,
                result={"research": "Expert 2 research"}
            )
        }
        
        # Mock task executor
        mock_result = TaskResult(
            task_id="synth_task",
            success=True,
            result={"synthesis": "Combined expert results"}
        )
        
        with patch.object(workflow._task_executor, 'execute') as mock_execute:
            mock_execute.return_value = mock_result
            
            result = await workflow._manager_synthesize_results(
                manager_agent,
                sample_task,
                expert_results,
                sample_execution_context
            )
            
            assert result == mock_result
            mock_execute.assert_called_once()
            
            # Check that the synthesis task was created properly
            call_args = mock_execute.call_args
            synth_task = call_args[0][0]
            assert "Synthesize results" in synth_task.title
            assert sample_task.title in synth_task.description

    def test_format_expert_capabilities(self):
        """Test formatting expert capabilities."""
        workflow = HierarchicalWorkflow()
        
        # Create expert agents
        expert1_config = Mock()
        expert1_config.name = "Expert 1"
        expert1_config.capabilities = ["reasoning", "analysis"]
        expert1_config.tools = ["search", "calculator"]
        expert1 = Mock()
        expert1.agent_id = "expert_001"
        expert1.config = expert1_config
        
        expert2_config = Mock()
        expert2_config.name = "Expert 2"
        expert2_config.capabilities = []
        expert2_config.tools = []
        expert2 = Mock()
        expert2.agent_id = "expert_002"
        expert2.config = expert2_config
        
        expert_agents = [expert1, expert2]
        
        formatted = workflow._format_expert_capabilities(expert_agents)
        
        assert isinstance(formatted, str)
        assert "Expert 1" in formatted
        assert "Expert 2" in formatted
        assert "reasoning, analysis" in formatted
        assert "search, calculator" in formatted
        assert "General" in formatted  # For empty capabilities
        assert "None" in formatted  # For empty tools

    def test_format_expert_results(self):
        """Test formatting expert results."""
        workflow = HierarchicalWorkflow()
        
        expert_results = {
            "expert_001": AgentExecutionResult(
                success=True,
                result={"analysis": "Detailed analysis result"}
            ),
            "expert_002": AgentExecutionResult(
                success=False,
                error_message="Expert failed to complete task"
            )
        }
        
        formatted = workflow._format_expert_results(expert_results)
        
        assert isinstance(formatted, str)
        assert "expert_001" in formatted
        assert "expert_002" in formatted
        assert "✓ Success" in formatted
        assert "✗ Failed" in formatted
        assert "Detailed analysis result" in formatted
        assert "Expert failed to complete task" in formatted

    @pytest.mark.asyncio
    async def test_full_hierarchical_execution(self, sample_workflow_config, sample_task, sample_execution_context):
        """Test full hierarchical workflow execution."""
        workflow = HierarchicalWorkflow()
        
        # Create manager and expert agents
        manager_config = Mock()
        manager_config.agent_type = AgentType.MANAGER
        manager_agent = Mock()
        manager_agent.agent_id = "manager_001"
        manager_agent.config = manager_config
        
        expert_config = Mock()
        expert_config.agent_type = AgentType.EXPERT
        expert_config.name = "Expert Agent"
        expert_config.capabilities = ["reasoning"]
        expert_config.tools = []
        expert_agent = Mock()
        expert_agent.agent_id = "expert_001"
        expert_agent.config = expert_config
        
        agents = [manager_agent, expert_agent]
        
        # Mock task executor for all three phases
        with patch.object(workflow._task_executor, 'execute') as mock_execute:
            mock_execute.side_effect = [
                # Manager decomposition
                TaskResult(task_id="decomp", success=True, result={"plan": "decomposed"}),
                # Expert execution
                TaskResult(task_id="expert", success=True, result={"expert_result": "done"}),
                # Manager synthesis
                TaskResult(task_id="synth", success=True, result={"final": "synthesized"})
            ]
            
            result = await workflow.execute(
                sample_workflow_config,
                sample_task,
                agents,
                sample_execution_context
            )
            
            assert isinstance(result, MultiAgentResult)
            assert result.success is True
            assert result.workflow_type == WorkflowType.HIERARCHICAL
            assert len(result.agents_used) == 2
            assert mock_execute.call_count == 3
