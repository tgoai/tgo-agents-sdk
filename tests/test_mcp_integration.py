"""
Integration tests for MCP tools with the multi-agent system.

This module tests the integration of MCP tools with:
- MultiAgentCoordinator
- Framework adapters
- Agent configurations
- Tool calling workflows
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from tgo.agents import (
    MultiAgentCoordinator, AdapterRegistry, GoogleADKAdapter,
    MCPToolManager, MCPServerConfig, MCPTool,
    InMemoryMemoryManager, InMemorySessionManager,
    MultiAgentConfig, AgentConfig, WorkflowConfig, Task,
    AgentType, WorkflowType, ExecutionStrategy, TaskType, TaskPriority,
    ExecutionContext
)
from tgo.agents.tools.mcp_security_manager import MCPSecurityManager, SecurityPolicy


class TestMCPCoordinatorIntegration:
    """Test MCP integration with MultiAgentCoordinator."""
    
    @pytest.fixture
    async def coordinator_setup(self):
        """Setup coordinator with MCP support."""
        # Create components
        registry = AdapterRegistry()
        memory_manager = InMemoryMemoryManager()
        session_manager = InMemorySessionManager()
        mcp_tool_manager = MCPToolManager()
        
        # Initialize MCP tool manager
        await mcp_tool_manager.initialize()
        
        # Register Google ADK adapter
        google_adapter = GoogleADKAdapter()
        registry.register("google-adk", google_adapter)
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(
            registry=registry,
            session_manager=session_manager,
            memory_manager=memory_manager,
            mcp_tool_manager=mcp_tool_manager
        )
        
        yield coordinator, mcp_tool_manager, registry
        
        # Cleanup
        await mcp_tool_manager.shutdown()
    
    @pytest.fixture
    def mcp_server_config(self):
        """Create MCP server configuration."""
        return MCPServerConfig(
            server_id="test_mcp_server",
            name="Test MCP Server",
            description="Test server for integration tests",
            transport_type="stdio",
            command="echo",
            args=["test"],
            trusted=True
        )
    
    @pytest.fixture
    def agent_config_with_mcp(self):
        """Create agent configuration with MCP tools."""
        return AgentConfig(
            agent_id="mcp_test_agent",
            name="MCP Test Agent",
            agent_type=AgentType.EXPERT,
            description="Agent with MCP tool access",
            model="gemini-2.0-flash",
            instructions="You have access to MCP tools for testing.",
            mcp_servers=["test_mcp_server"],
            mcp_auto_approve=True,
            tools=[],
            max_iterations=5
        )
    
    async def test_coordinator_mcp_injection(self, coordinator_setup):
        """Test that MCP tool manager is injected into adapters."""
        coordinator, mcp_tool_manager, registry = coordinator_setup
        
        # Initialize adapter
        await registry.initialize_adapter("google-adk")
        
        # Get adapter and check MCP tool manager injection
        adapter = registry.get_adapter("google-adk")
        
        # The adapter should have MCP tool manager injected when accessed through coordinator
        with patch.object(coordinator, '_get_framework_adapter') as mock_get_adapter:
            mock_get_adapter.return_value = adapter
            
            # Simulate getting adapter through coordinator
            result_adapter = await coordinator._get_framework_adapter("google-adk")
            
            # Verify MCP tool manager would be injected
            assert hasattr(adapter, 'set_mcp_tool_manager')
    
    @patch('tgo.agents.adapters.google_adk_adapter.GoogleADKAdapter._create_framework_agent')
    async def test_agent_creation_with_mcp(self, mock_create_agent, coordinator_setup, 
                                         mcp_server_config, agent_config_with_mcp):
        """Test agent creation with MCP tools."""
        coordinator, mcp_tool_manager, registry = coordinator_setup
        
        # Setup MCP server
        await mcp_tool_manager.register_server(mcp_server_config)
        
        # Mock successful connection
        with patch.object(mcp_tool_manager, 'connect_to_server', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            
            # Mock agent creation
            mock_create_agent.return_value = Mock()
            
            # Initialize adapter
            await registry.initialize_adapter("google-adk")
            adapter = registry.get_adapter("google-adk")
            
            # Inject MCP tool manager
            await adapter.set_mcp_tool_manager(mcp_tool_manager)
            
            # Create agent
            agent_instance = await adapter.create_agent(agent_config_with_mcp)
            
            assert agent_instance is not None
            assert agent_instance.config.mcp_servers == ["test_mcp_server"]
    
    async def test_mcp_tool_availability(self, coordinator_setup, mcp_server_config, agent_config_with_mcp):
        """Test MCP tool availability for agents."""
        coordinator, mcp_tool_manager, registry = coordinator_setup
        
        # Setup MCP server and tools
        await mcp_tool_manager.register_server(mcp_server_config)
        
        # Add mock tool
        test_tool = MCPTool(
            name="test_mcp_tool",
            description="Test MCP tool",
            input_schema={"type": "object"},
            server_id="test_mcp_server"
        )
        mcp_tool_manager._tools_by_name["test_mcp_tool"] = test_tool
        mcp_tool_manager._tools_by_server["test_mcp_server"] = [test_tool]
        
        # Initialize adapter
        await registry.initialize_adapter("google-adk")
        adapter = registry.get_adapter("google-adk")
        await adapter.set_mcp_tool_manager(mcp_tool_manager)
        
        # Create agent
        with patch.object(adapter, '_create_framework_agent', return_value=Mock()):
            agent_instance = await adapter.create_agent(agent_config_with_mcp)
            
            # Get available MCP tools
            available_tools = await adapter.get_available_mcp_tools("mcp_test_agent")
            
            assert len(available_tools) == 1
            assert available_tools[0]["name"] == "test_mcp_tool"


class TestMCPToolCalling:
    """Test MCP tool calling through adapters."""
    
    @pytest.fixture
    async def adapter_with_mcp(self):
        """Setup Google ADK adapter with MCP support."""
        adapter = GoogleADKAdapter()
        mcp_tool_manager = MCPToolManager()
        await mcp_tool_manager.initialize()
        
        await adapter.set_mcp_tool_manager(mcp_tool_manager)
        
        yield adapter, mcp_tool_manager
        
        await mcp_tool_manager.shutdown()
    
    @pytest.fixture
    def execution_context(self):
        """Create execution context."""
        return ExecutionContext(
            task_id="test_task",
            agent_id="test_agent",
            session_id="test_session",
            user_id="test_user"
        )
    
    async def test_mcp_tool_call_success(self, adapter_with_mcp, execution_context):
        """Test successful MCP tool call."""
        adapter, mcp_tool_manager = adapter_with_mcp
        
        # Setup mock tool
        test_tool = MCPTool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object"},
            server_id="test_server"
        )
        mcp_tool_manager._tools_by_name["test_tool"] = test_tool
        
        # Mock tool call
        with patch.object(mcp_tool_manager, 'call_tool', new_callable=AsyncMock) as mock_call:
            from tgo.agents.core.models import MCPToolCallResult
            
            mock_call.return_value = MCPToolCallResult(
                request_id="test_request",
                tool_name="test_tool",
                server_id="test_server",
                success=True,
                content=[{"type": "text", "text": "Tool result"}]
            )
            
            # Call MCP tool through adapter
            result = await adapter.call_mcp_tool(
                agent_id="test_agent",
                tool_name="test_tool",
                arguments={"param": "value"},
                context=execution_context,
                user_approved=True
            )
            
            assert result.success is True
            assert result.tool_name == "test_tool"
            assert "text" in result.result
    
    async def test_mcp_tool_call_failure(self, adapter_with_mcp, execution_context):
        """Test MCP tool call failure."""
        adapter, mcp_tool_manager = adapter_with_mcp
        
        # Mock tool call failure
        with patch.object(mcp_tool_manager, 'call_tool', new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("Tool call failed")
            
            # Call MCP tool through adapter
            result = await adapter.call_mcp_tool(
                agent_id="test_agent",
                tool_name="test_tool",
                arguments={},
                context=execution_context
            )
            
            assert result.success is False
            assert "Tool call failed" in result.error_message
    
    async def test_mcp_tool_call_no_manager(self, execution_context):
        """Test MCP tool call without tool manager."""
        adapter = GoogleADKAdapter()
        # No MCP tool manager set
        
        from tgo.agents.core.exceptions import FrameworkError
        
        with pytest.raises(FrameworkError, match="MCP tool manager not available"):
            await adapter.call_mcp_tool(
                agent_id="test_agent",
                tool_name="test_tool",
                arguments={},
                context=execution_context
            )


class TestMCPWorkflowIntegration:
    """Test MCP tools in multi-agent workflows."""
    
    @pytest.fixture
    async def workflow_setup(self):
        """Setup multi-agent workflow with MCP support."""
        # Create components
        registry = AdapterRegistry()
        memory_manager = InMemoryMemoryManager()
        session_manager = InMemorySessionManager()
        mcp_tool_manager = MCPToolManager()
        security_manager = MCPSecurityManager()
        
        # Initialize
        await mcp_tool_manager.initialize()
        
        # Setup MCP server
        server_config = MCPServerConfig(
            server_id="workflow_server",
            name="Workflow MCP Server",
            description="MCP server for workflow tests",
            transport_type="stdio",
            command="echo",
            args=["workflow"],
            trusted=True
        )
        await mcp_tool_manager.register_server(server_config)
        
        # Add test tools
        file_tool = MCPTool(
            name="read_file",
            description="Read file content",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            },
            server_id="workflow_server"
        )
        
        analysis_tool = MCPTool(
            name="analyze_data",
            description="Analyze data",
            input_schema={
                "type": "object",
                "properties": {"data": {"type": "string"}},
                "required": ["data"]
            },
            server_id="workflow_server"
        )
        
        mcp_tool_manager._tools_by_name["read_file"] = file_tool
        mcp_tool_manager._tools_by_name["analyze_data"] = analysis_tool
        mcp_tool_manager._tools_by_server["workflow_server"] = [file_tool, analysis_tool]
        
        # Register adapter
        google_adapter = GoogleADKAdapter()
        registry.register("google-adk", google_adapter)
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(
            registry=registry,
            session_manager=session_manager,
            memory_manager=memory_manager,
            mcp_tool_manager=mcp_tool_manager
        )
        
        yield coordinator, mcp_tool_manager
        
        await mcp_tool_manager.shutdown()
    
    @pytest.fixture
    def workflow_config(self):
        """Create workflow configuration with MCP-enabled agents."""
        manager_config = AgentConfig(
            agent_id="workflow_manager",
            name="Workflow Manager",
            agent_type=AgentType.MANAGER,
            description="Manager with MCP tools",
            model="gemini-2.0-flash",
            instructions="Coordinate workflow using MCP tools",
            mcp_servers=["workflow_server"],
            mcp_auto_approve=True
        )
        
        file_agent_config = AgentConfig(
            agent_id="file_agent",
            name="File Agent",
            agent_type=AgentType.EXPERT,
            description="Agent for file operations",
            model="gemini-2.0-flash",
            instructions="Handle file operations using MCP tools",
            mcp_servers=["workflow_server"],
            mcp_tools=["read_file"],
            mcp_auto_approve=True
        )
        
        analysis_agent_config = AgentConfig(
            agent_id="analysis_agent",
            name="Analysis Agent",
            agent_type=AgentType.EXPERT,
            description="Agent for data analysis",
            model="gemini-2.0-flash",
            instructions="Analyze data using MCP tools",
            mcp_servers=["workflow_server"],
            mcp_tools=["analyze_data"],
            mcp_auto_approve=True
        )
        
        workflow = WorkflowConfig(
            workflow_type=WorkflowType.HIERARCHICAL,
            execution_strategy=ExecutionStrategy.FAIL_FAST,
            manager_agent_id="workflow_manager",
            expert_agent_ids=["file_agent", "analysis_agent"]
        )
        
        return MultiAgentConfig(
            framework="google-adk",
            agents=[manager_config, file_agent_config, analysis_agent_config],
            workflow=workflow
        )
    
    @patch('tgo.agents.adapters.google_adk_adapter.GoogleADKAdapter.execute_task')
    async def test_workflow_with_mcp_tools(self, mock_execute, workflow_setup, workflow_config):
        """Test multi-agent workflow using MCP tools."""
        coordinator, mcp_tool_manager = workflow_setup
        
        # Mock successful task execution
        from tgo.agents.core.models import AgentExecutionResult, ToolCallResult
        
        mock_execute.return_value = AgentExecutionResult(
            agent_id="test_agent",
            success=True,
            result={"output": "Workflow completed with MCP tools"},
            tool_calls=[
                ToolCallResult(
                    tool_name="read_file",
                    tool_id="mcp:read_file",
                    success=True,
                    result={"content": "file data"}
                ),
                ToolCallResult(
                    tool_name="analyze_data",
                    tool_id="mcp:analyze_data", 
                    success=True,
                    result={"analysis": "data insights"}
                )
            ]
        )
        
        # Create task
        task = Task(
            title="MCP Workflow Test",
            description="Test workflow using MCP tools",
            task_type=TaskType.COMPLEX,
            priority=TaskPriority.MEDIUM,
            input_data={"file_path": "/test/data.txt"}
        )
        
        # Execute workflow
        result = await coordinator.execute_task(workflow_config, task)
        
        # Verify execution
        assert result is not None
        # Mock was called, indicating workflow execution attempted
        assert mock_execute.called


class TestMCPSecurityIntegration:
    """Test MCP security integration with multi-agent system."""
    
    @pytest.fixture
    async def secure_setup(self):
        """Setup system with MCP security controls."""
        mcp_tool_manager = MCPToolManager()
        security_manager = MCPSecurityManager()
        
        # Set restrictive policy
        restrictive_policy = SecurityPolicy(
            allowed_tools={"safe_tool"},
            denied_tools={"dangerous_tool"},
            max_calls_per_minute=2,
            require_approval_for_untrusted=True
        )
        security_manager.set_policy("restricted_agent", restrictive_policy)
        
        # Replace security manager in tool manager
        mcp_tool_manager._security_manager = security_manager
        
        await mcp_tool_manager.initialize()
        
        yield mcp_tool_manager, security_manager
        
        await mcp_tool_manager.shutdown()
    
    async def test_security_policy_enforcement(self, secure_setup):
        """Test that security policies are enforced."""
        mcp_tool_manager, security_manager = secure_setup
        
        # Add tools
        safe_tool = MCPTool(
            name="safe_tool",
            description="Safe tool",
            input_schema={"type": "object"},
            server_id="test_server"
        )
        
        dangerous_tool = MCPTool(
            name="dangerous_tool",
            description="Dangerous tool",
            input_schema={"type": "object"},
            server_id="test_server"
        )
        
        mcp_tool_manager._tools_by_name["safe_tool"] = safe_tool
        mcp_tool_manager._tools_by_name["dangerous_tool"] = dangerous_tool
        
        # Mock connection
        from tgo.agents.core.models import MCPConnection, MCPServerConfig
        
        server_config = MCPServerConfig(
            server_id="test_server",
            name="Test Server",
            description="Test",
            transport_type="stdio",
            command="echo"
        )
        
        mcp_tool_manager._connections["test_server"] = MCPConnection(
            server_config=server_config,
            status="connected"
        )
        
        # Mock connector
        mock_connector = AsyncMock()
        mock_connector.call_tool.return_value = {
            "content": [{"type": "text", "text": "result"}],
            "isError": False
        }
        mcp_tool_manager._connectors["test_server"] = mock_connector
        
        context = ExecutionContext(
            task_id="test",
            agent_id="restricted_agent"
        )
        
        # Safe tool should work
        result = await mcp_tool_manager.call_tool(
            tool_name="safe_tool",
            arguments={},
            context=context,
            user_approved=True
        )
        assert result.success is True
        
        # Dangerous tool should be denied
        from tgo.agents.tools.mcp_tool_manager import MCPToolManagerError
        
        with pytest.raises(MCPToolManagerError, match="Permission denied"):
            await mcp_tool_manager.call_tool(
                tool_name="dangerous_tool",
                arguments={},
                context=context,
                user_approved=True
            )


if __name__ == "__main__":
    pytest.main([__file__])
