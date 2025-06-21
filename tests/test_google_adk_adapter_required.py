"""
Unit tests for Google ADK adapter with required dependencies.

This module tests the GoogleADKAdapter class after modifications to make
Google ADK a required dependency instead of optional.
"""

import pytest
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

# Mock Google ADK modules before importing
mock_llm_agent = MagicMock()
mock_google_search = MagicMock()
mock_run_config = MagicMock()

# Create mock modules
sys.modules['google.adk.agents'] = MagicMock()
sys.modules['google.adk.tools'] = MagicMock()
sys.modules['google.adk.runtime'] = MagicMock()

# Set up the mocks
sys.modules['google.adk.agents'].LlmAgent = mock_llm_agent
sys.modules['google.adk.tools'].google_search = mock_google_search
sys.modules['google.adk.runtime'].RunConfig = mock_run_config

from src.adapters.google_adk_adapter import GoogleADKAdapter
from src.core.models import AgentConfig, Task, ExecutionContext, AgentExecutionResult
from src.core.enums import AgentType, FrameworkCapability
from src.core.exceptions import AgentCreationError


class TestGoogleADKAdapterRequired:
    """Test cases for GoogleADKAdapter with required dependencies."""

    def test_adapter_initialization_with_required_adk(self):
        """Test adapter initialization when ADK is required."""
        adapter = GoogleADKAdapter()
        
        assert adapter.name == "google-adk"
        assert adapter.version_info == "1.4.1"
        assert not adapter.is_initialized
        
        # Check that streaming capability is always included now
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
    async def test_framework_initialization_required_adk(self):
        """Test framework initialization with required ADK."""
        adapter = GoogleADKAdapter()
        
        # Mock the _create_llm_agent method to avoid actual ADK calls
        with patch.object(adapter, '_create_llm_agent') as mock_create:
            mock_create.return_value = Mock()
            await adapter.initialize()
        
        assert adapter.is_initialized

    def test_create_run_config_always_uses_adk(self):
        """Test that run config creation always uses ADK RunConfig."""
        adapter = GoogleADKAdapter()

        config = AgentConfig(
            agent_id="test_agent",
            name="Test Agent",
            agent_type=AgentType.EXPERT,
            model="gemini-2.0-flash",
            temperature=0.7,
            max_iterations=5
        )

        # Use patch to mock RunConfig in the actual module
        with patch('src.adapters.google_adk_adapter.RunConfig') as mock_run_config_local:
            mock_run_config_local.return_value = Mock()

            result = adapter._create_run_config(config)

            # Verify RunConfig was called
            mock_run_config_local.assert_called_once()
            # Check that the result is not None
            assert result is not None

    @pytest.mark.asyncio
    async def test_create_manager_agent_always_uses_adk(self):
        """Test that manager agent creation always uses ADK LlmAgent."""
        adapter = GoogleADKAdapter()

        config = AgentConfig(
            agent_id="manager_agent",
            name="Manager Agent",
            agent_type=AgentType.MANAGER,
            model="gemini-2.0-flash"
        )

        # Mock LlmAgent and tools
        mock_llm_agent.reset_mock()
        mock_llm_agent.return_value = Mock()

        # Get tools first
        tools = await adapter._get_tools_for_agent([])

        result = await adapter._create_manager_agent(config, tools)

        # Verify LlmAgent was called
        mock_llm_agent.assert_called_once()
        # Just verify the result is not None
        assert result is not None

    @pytest.mark.asyncio
    async def test_create_expert_agent_always_uses_adk(self):
        """Test that expert agent creation always uses ADK LlmAgent."""
        adapter = GoogleADKAdapter()

        config = AgentConfig(
            agent_id="expert_agent",
            name="Expert Agent",
            agent_type=AgentType.EXPERT,
            model="gemini-2.0-flash"
        )

        # Reset mock to clear previous calls
        mock_llm_agent.reset_mock()
        mock_llm_agent.return_value = Mock()

        # Get tools first
        tools = await adapter._get_tools_for_agent([])

        result = await adapter._create_expert_agent(config, tools)

        # Verify LlmAgent was called
        mock_llm_agent.assert_called_once()
        # Just verify the result is not None
        assert result is not None

    @pytest.mark.asyncio
    async def test_create_llm_agent_always_uses_adk(self):
        """Test that LLM agent creation always uses ADK LlmAgent."""
        adapter = GoogleADKAdapter()

        config = AgentConfig(
            agent_id="llm_agent",
            name="LLM Agent",
            agent_type=AgentType.CUSTOM,
            model="gemini-2.0-flash"
        )

        # Reset mock to clear previous calls
        mock_llm_agent.reset_mock()
        mock_llm_agent.return_value = Mock()

        # Get tools first
        tools = await adapter._get_tools_for_agent([])

        result = await adapter._create_llm_agent(config, tools)

        # Verify LlmAgent was called
        mock_llm_agent.assert_called_once()
        # Just verify the result is not None
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_tools_for_agent(self):
        """Test that tools can be retrieved for an agent."""
        adapter = GoogleADKAdapter()

        # Test getting tools for an agent
        tools = await adapter._get_tools_for_agent(["google_search"])

        # Should return a list
        assert isinstance(tools, list)

    def test_no_mock_classes_used(self):
        """Test that no mock classes are used since ADK is required."""
        adapter = GoogleADKAdapter()

        # Verify that the adapter doesn't have any mock-related attributes
        assert not hasattr(adapter, 'MockAgent')
        assert not hasattr(adapter, '_mock_implementation')

    @pytest.mark.asyncio
    async def test_framework_initialization_success(self):
        """Test that framework initialization succeeds with ADK."""
        adapter = GoogleADKAdapter()

        # Mock the _create_llm_agent to avoid actual ADK calls
        with patch.object(adapter, '_create_llm_agent') as mock_create:
            mock_create.return_value = Mock()

            # Should not raise exception since ADK is required and available
            await adapter.initialize()
            assert adapter.is_initialized

    def test_default_instructions_methods(self):
        """Test that default instruction methods work correctly."""
        adapter = GoogleADKAdapter()

        # Test manager instructions
        manager_instructions = adapter._get_default_manager_instructions()
        assert isinstance(manager_instructions, str)
        assert len(manager_instructions) > 0
        assert "manager" in manager_instructions.lower()

        # Test expert instructions
        expert_instructions = adapter._get_default_expert_instructions()
        assert isinstance(expert_instructions, str)
        assert len(expert_instructions) > 0
        assert "expert" in expert_instructions.lower()

    @pytest.mark.asyncio
    async def test_execute_adk_task_direct_call(self):
        """Test that ADK task execution is called directly."""
        adapter = GoogleADKAdapter()
        
        # Create mock agent with run method
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value="ADK response")
        mock_config = Mock()
        
        result = await adapter._execute_adk_task(mock_agent, "test input", mock_config)
        
        assert result == "ADK response"
        mock_agent.run.assert_called_once_with("test input", config=mock_config)

    @pytest.mark.asyncio
    async def test_execute_adk_task_fallback(self):
        """Test ADK task execution fallback when run method not available."""
        adapter = GoogleADKAdapter()
        
        # Create mock agent without run method but callable
        mock_agent = AsyncMock()
        mock_agent.return_value = "Fallback response"
        # Remove run attribute to trigger fallback
        if hasattr(mock_agent, 'run'):
            delattr(mock_agent, 'run')
        
        result = await adapter._execute_adk_task(mock_agent, "test input", None)
        
        assert result == "Fallback response"
        mock_agent.assert_called_once_with("test input")
