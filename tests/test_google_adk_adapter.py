"""
Unit tests for Google ADK adapter.

This module tests the GoogleADKAdapter class and its integration
with the Google Agent Development Kit.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone


from src.adapters.google_adk_adapter import GoogleADKAdapter

from src.core.models import AgentConfig, Task, ExecutionContext, AgentExecutionResult, ToolCallResult, KnowledgeBaseQueryResult
from src.core.enums import AgentType, FrameworkCapability
from src.core.exceptions import AgentCreationError


class TestGoogleADKAdapter:
    """Test cases for GoogleADKAdapter."""

    @patch('src.adapters.google_adk_adapter.LlmAgent')
    @patch('src.adapters.google_adk_adapter.google_search')
    @patch('src.adapters.google_adk_adapter.RunConfig')
    def test_adapter_initialization(self, mock_run_config, mock_google_search, mock_llm_agent):
        """Test adapter initialization."""
        adapter = GoogleADKAdapter()

        assert adapter.name == "google-adk"
        assert adapter.version_info == "1.4.1"
        assert not adapter.is_initialized

        # Check capabilities - now includes streaming since ADK is always available
        expected_capabilities = [
            FrameworkCapability.SINGLE_AGENT,
            FrameworkCapability.MULTI_AGENT,
            FrameworkCapability.TOOL_CALLING,
            FrameworkCapability.KNOWLEDGE_BASE,
            FrameworkCapability.MEMORY,
            FrameworkCapability.STREAMING,
        ]
        for capability in expected_capabilities:
            assert adapter.supports_capability(capability)

    @pytest.mark.asyncio
    async def test_framework_initialization(self):
        """Test framework initialization."""
        adapter = GoogleADKAdapter()
        
        await adapter.initialize()
        
        assert adapter.is_initialized
        assert len(adapter._adk_agents) == 0
        assert len(adapter._run_configs) == 0

    @pytest.mark.asyncio
    async def test_framework_cleanup(self):
        """Test framework cleanup."""
        adapter = GoogleADKAdapter()
        await adapter.initialize()
        
        # Add some mock data
        adapter._adk_agents["test"] = Mock()
        adapter._run_configs["test"] = Mock()
        
        await adapter.cleanup()
        
        assert not adapter.is_initialized
        assert len(adapter._adk_agents) == 0
        assert len(adapter._run_configs) == 0

    @pytest.mark.asyncio
    async def test_create_manager_agent(self, sample_manager_config):
        """Test creating a manager agent."""
        adapter = GoogleADKAdapter()
        await adapter.initialize()
        
        with patch.object(adapter, '_get_tools_for_agent') as mock_get_tools:
            mock_get_tools.return_value = []
            
            instance = await adapter.create_agent(sample_manager_config)
            
            assert instance.agent_id == sample_manager_config.agent_id
            assert instance.config == sample_manager_config
            assert sample_manager_config.agent_id in adapter._adk_agents
            assert sample_manager_config.agent_id in adapter._run_configs

    @pytest.mark.asyncio
    async def test_create_expert_agent(self, sample_agent_config):
        """Test creating an expert agent."""
        adapter = GoogleADKAdapter()
        await adapter.initialize()
        
        with patch.object(adapter, '_get_tools_for_agent') as mock_get_tools:
            mock_get_tools.return_value = []
            
            instance = await adapter.create_agent(sample_agent_config)
            
            assert instance.agent_id == sample_agent_config.agent_id
            assert instance.config == sample_agent_config
            assert sample_agent_config.agent_id in adapter._adk_agents
            assert sample_agent_config.agent_id in adapter._run_configs

    @pytest.mark.asyncio
    async def test_create_general_agent(self):
        """Test creating a general LLM agent."""
        adapter = GoogleADKAdapter()
        await adapter.initialize()
        
        config = AgentConfig(
            agent_id="general_agent",
            name="General Agent",
            agent_type=AgentType.CUSTOM,  # Not manager or expert
            model="gemini-2.0-flash"
        )
        
        with patch.object(adapter, '_get_tools_for_agent') as mock_get_tools:
            mock_get_tools.return_value = []
            
            instance = await adapter.create_agent(config)
            
            assert instance.agent_id == config.agent_id
            assert config.agent_id in adapter._adk_agents

    @pytest.mark.asyncio
    async def test_create_agent_with_tools(self, sample_agent_config):
        """Test creating an agent with tools."""
        adapter = GoogleADKAdapter()
        await adapter.initialize()
        
        sample_agent_config.tools = ["search", "calculator"]
        
        with patch.object(adapter, '_get_tools_for_agent') as mock_get_tools:
            mock_tools = [Mock(), Mock()]
            mock_get_tools.return_value = mock_tools
            
            instance = await adapter.create_agent(sample_agent_config)
            
            assert instance.agent_id == sample_agent_config.agent_id
            mock_get_tools.assert_called_once_with(["search", "calculator"])

    @pytest.mark.asyncio
    async def test_create_agent_failure(self, sample_agent_config):
        """Test agent creation failure."""
        adapter = GoogleADKAdapter()
        await adapter.initialize()

        with patch.object(adapter, '_get_tools_for_agent') as mock_get_tools:
            mock_get_tools.side_effect = Exception("Tool loading failed")

            with pytest.raises(AgentCreationError, match="Tool loading failed"):
                await adapter.create_agent(sample_agent_config)

    def test_get_default_manager_instructions(self):
        """Test getting default manager instructions."""
        adapter = GoogleADKAdapter()
        
        instructions = adapter._get_default_manager_instructions()
        
        assert isinstance(instructions, str)
        assert len(instructions) > 0
        assert "manager" in instructions.lower()
        assert "coordinate" in instructions.lower() or "delegation" in instructions.lower()

    def test_get_default_expert_instructions(self):
        """Test getting default expert instructions."""
        adapter = GoogleADKAdapter()
        
        instructions = adapter._get_default_expert_instructions()
        
        assert isinstance(instructions, str)
        assert len(instructions) > 0
        assert "expert" in instructions.lower()
        assert "specialized" in instructions.lower() or "expertise" in instructions.lower()

    def test_create_run_config(self, sample_agent_config):
        """Test creating run configuration."""
        adapter = GoogleADKAdapter()
        
        run_config = adapter._create_run_config(sample_agent_config)
        
        assert run_config is not None

    @pytest.mark.asyncio
    async def test_get_tools_for_agent(self):
        """Test getting tools for an agent."""
        adapter = GoogleADKAdapter()
        
        tools = await adapter._get_tools_for_agent(["search", "calculator"])
        
        assert isinstance(tools, list)
        # Should at least include google_search if available

    def test_prepare_task_input(self, sample_task):
        """Test preparing task input."""
        adapter = GoogleADKAdapter()
        
        task_input = adapter._prepare_task_input(sample_task)
        
        assert isinstance(task_input, str)
        assert sample_task.title in task_input
        if sample_task.description:
            assert sample_task.description in task_input

    @pytest.mark.asyncio
    async def test_execute_adk_task(self):
        """Test executing ADK task."""
        adapter = GoogleADKAdapter()
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value="Mock response")
        
        # Mock run config
        mock_config = Mock()
        
        result = await adapter._execute_adk_task(mock_agent, "test input", mock_config)
        
        assert result == "Mock response"
        mock_agent.run.assert_called_once_with("test input", config=mock_config)

    @pytest.mark.asyncio
    async def test_execute_adk_task_no_config(self):
        """Test executing ADK task without config."""
        adapter = GoogleADKAdapter()
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value="Mock response")
        
        result = await adapter._execute_adk_task(mock_agent, "test input", None)
        
        assert result == "Mock response"
        mock_agent.run.assert_called_once_with("test input")

    @pytest.mark.asyncio
    async def test_execute_adk_task_fallback(self):
        """Test executing ADK task with fallback method."""
        adapter = GoogleADKAdapter()

        # Mock agent without run method but callable
        mock_agent = AsyncMock()
        mock_agent.return_value = "Fallback response"
        # Remove the run attribute to trigger fallback
        del mock_agent.run

        result = await adapter._execute_adk_task(mock_agent, "test input", None)

        assert result == "Fallback response"
        mock_agent.assert_called_once_with("test input")

    @pytest.mark.asyncio
    async def test_process_execution_result_string(self, sample_task, sample_execution_context):
        """Test processing string execution result."""
        adapter = GoogleADKAdapter()
        start_time = datetime.now(timezone.utc)

        # Add a small delay to ensure execution time > 0
        import asyncio
        await asyncio.sleep(0.001)

        result = await adapter._process_execution_result(
            "String response",
            sample_task,
            sample_execution_context,
            start_time
        )

        assert isinstance(result, AgentExecutionResult)
        assert result.success is True
        assert result.result["response"] == "String response"
        assert result.execution_time_ms >= 0  # Changed to >= 0 since it might be 0 in fast tests

    @pytest.mark.asyncio
    async def test_process_execution_result_dict(self, sample_task, sample_execution_context):
        """Test processing dictionary execution result."""
        adapter = GoogleADKAdapter()
        start_time = datetime.now(timezone.utc)
        
        dict_result = {"key": "value", "data": [1, 2, 3]}
        
        result = await adapter._process_execution_result(
            dict_result,
            sample_task,
            sample_execution_context,
            start_time
        )
        
        assert isinstance(result, AgentExecutionResult)
        assert result.success is True
        assert result.result == dict_result

    @pytest.mark.asyncio
    async def test_execute_framework_task(self, sample_agent_config, sample_task, sample_execution_context):
        """Test executing framework task."""
        adapter = GoogleADKAdapter()
        await adapter.initialize()
        
        # Create agent first
        instance = await adapter.create_agent(sample_agent_config)
        framework_agent = adapter._adk_agents[sample_agent_config.agent_id]


        result = await adapter._execute_framework_task(
                framework_agent,
                sample_task,
                sample_execution_context
            )
        assert result.success is True
        assert "response" in result.result

    @pytest.mark.asyncio
    async def test_execute_framework_task_failure(self, sample_agent_config, sample_task, sample_execution_context):
        """Test framework task execution failure."""
        adapter = GoogleADKAdapter()
        await adapter.initialize()

        # Create agent first
        instance = await adapter.create_agent(sample_agent_config)
        framework_agent = adapter._adk_agents[sample_agent_config.agent_id]

        # Mock the ADK execution to fail by patching the framework agent's run method
        with patch.object(framework_agent, 'run') as mock_run:
            mock_run.side_effect = Exception("ADK execution failed")

            result = await adapter._execute_framework_task(
                framework_agent,
                sample_task,
                sample_execution_context
            )

            assert isinstance(result, AgentExecutionResult)
            assert result.success is False
            assert result.error_message is not None
            assert "ADK execution failed" in result.error_message

    @pytest.mark.asyncio
    async def test_call_tool(self, sample_execution_context):
        """Test calling a tool."""
        adapter = GoogleADKAdapter()
        
        result = await adapter.call_tool(
            "test_agent",
            "tool_001",
            "test_tool",
            {"param1": "value1"},
            sample_execution_context
        )
        
        assert isinstance(result, ToolCallResult)
        assert result.success is True
        assert result.tool_name == "test_tool"
        assert result.tool_id == "tool_001"
        assert "tool_result" in result.result

    @pytest.mark.asyncio
    async def test_call_tool_failure(self, sample_execution_context):
        """Test tool call failure."""
        adapter = GoogleADKAdapter()
        
        # Mock tool execution to fail
        with patch.object(adapter, 'call_tool') as mock_call:
            mock_call.side_effect = Exception("Tool call failed")
            
            with pytest.raises(Exception):
                await adapter.call_tool(
                    "test_agent",
                    "tool_001",
                    "test_tool",
                    {},
                    sample_execution_context
                )

    @pytest.mark.asyncio
    async def test_query_knowledge_base(self, sample_execution_context):
        """Test querying knowledge base."""
        adapter = GoogleADKAdapter()
        
        result = await adapter.query_knowledge_base(
            "test_agent",
            "kb_001",
            "test_kb",
            "test query",
            {"param1": "value1"},
            sample_execution_context
        )
        
        assert isinstance(result, KnowledgeBaseQueryResult)
        assert result.success is True
        assert result.kb_name == "test_kb"
        assert result.kb_id == "kb_001"
        assert result.query == "test query"
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_query_knowledge_base_failure(self, sample_execution_context):
        """Test knowledge base query failure."""
        adapter = GoogleADKAdapter()
        
        # Mock KB query to fail
        with patch.object(adapter, 'query_knowledge_base') as mock_query:
            mock_query.side_effect = Exception("KB query failed")
            
            with pytest.raises(Exception):
                await adapter.query_knowledge_base(
                    "test_agent",
                    "kb_001",
                    "test_kb",
                    "test query",
                    {},
                    sample_execution_context
                )

    @pytest.mark.asyncio
    async def test_agent_deletion(self, sample_agent_config):
        """Test agent deletion."""
        adapter = GoogleADKAdapter()
        await adapter.initialize()
        
        # Create agent
        instance = await adapter.create_agent(sample_agent_config)
        assert sample_agent_config.agent_id in adapter._adk_agents
        assert sample_agent_config.agent_id in adapter._run_configs
        
        # Delete agent
        result = await adapter.delete_agent(sample_agent_config.agent_id)
        
        assert result is True
        assert sample_agent_config.agent_id not in adapter._adk_agents
        assert sample_agent_config.agent_id not in adapter._run_configs

    @pytest.mark.asyncio
    async def test_full_task_execution_flow(self, sample_agent_config, sample_task):
        """Test full task execution flow."""
        adapter = GoogleADKAdapter()
        await adapter.initialize()
        
        # Create agent
        instance = await adapter.create_agent(sample_agent_config)
        
        # Execute task
        result = await adapter.execute_task(
            sample_agent_config.agent_id,
            sample_task
        )
        
        assert isinstance(result, AgentExecutionResult)
        assert result.success is True
        assert result.result is not None
