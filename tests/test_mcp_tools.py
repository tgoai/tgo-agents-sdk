"""
Test cases for MCP (Model Context Protocol) tools functionality.

This module contains comprehensive tests for MCP tool integration including:
- Tool manager functionality
- Connection management
- Security controls
- Tool calling and result handling
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from tgo.agents.tools.mcp_tool_manager import MCPToolManager, MCPToolManagerError
from tgo.agents.tools.mcp_connector import MCPConnector, MCPConnectorError
from tgo.agents.tools.mcp_tool_proxy import MCPToolProxy
from tgo.agents.tools.mcp_security_manager import (
    MCPSecurityManager, SecurityPolicy, PermissionAction, SecurityLevel
)
from tgo.agents.core.models import (
    MCPServerConfig, MCPTool, MCPConnection, MCPToolCallRequest, MCPToolCallResult,
    ExecutionContext, AgentConfig, AgentType
)


class TestMCPToolManager:
    """Test cases for MCPToolManager."""
    
    @pytest.fixture
    async def tool_manager(self):
        """Create a test tool manager."""
        manager = MCPToolManager()
        await manager.initialize()
        yield manager
        await manager.shutdown()
    
    @pytest.fixture
    def sample_server_config(self):
        """Create a sample MCP server configuration."""
        return MCPServerConfig(
            server_id="test_server",
            name="Test MCP Server",
            description="Test server for unit tests",
            transport_type="stdio",
            command="echo",
            args=["test"],
            trusted=True
        )
    
    @pytest.fixture
    def sample_tool(self):
        """Create a sample MCP tool."""
        return MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"]
            },
            server_id="test_server",
            requires_confirmation=False
        )
    
    @pytest.fixture
    def execution_context(self):
        """Create a test execution context."""
        return ExecutionContext(
            task_id="test_task",
            agent_id="test_agent",
            session_id="test_session",
            user_id="test_user"
        )
    
    async def test_register_server(self, tool_manager, sample_server_config):
        """Test server registration."""
        result = await tool_manager.register_server(sample_server_config)
        assert result is True
        
        # Check that server is registered
        assert sample_server_config.server_id in tool_manager._server_configs
    
    async def test_register_duplicate_server(self, tool_manager, sample_server_config):
        """Test registering duplicate server."""
        await tool_manager.register_server(sample_server_config)
        
        # Should succeed (update existing)
        result = await tool_manager.register_server(sample_server_config)
        assert result is True
    
    @patch('tgo.agents.tools.mcp_tool_manager.MCPConnector')
    async def test_connect_to_server(self, mock_connector_class, tool_manager, sample_server_config):
        """Test connecting to MCP server."""
        # Setup mock connector
        mock_connector = AsyncMock()
        mock_connector.connect = AsyncMock()
        mock_connector.get_capabilities.return_value = {"tools": {}}
        mock_connector.get_protocol_version.return_value = "2025-06-18"
        mock_connector_class.return_value = mock_connector
        
        # Register server first
        await tool_manager.register_server(sample_server_config)
        
        # Mock tool discovery
        with patch.object(tool_manager, '_discover_tools', new_callable=AsyncMock):
            result = await tool_manager.connect_to_server("test_server")
            assert result is True
        
        # Check connection was created
        assert "test_server" in tool_manager._connections
        assert tool_manager._connections["test_server"].status == "connected"
    
    async def test_connect_to_unregistered_server(self, tool_manager):
        """Test connecting to unregistered server."""
        with pytest.raises(MCPToolManagerError, match="Server .* not registered"):
            await tool_manager.connect_to_server("nonexistent_server")
    
    async def test_get_available_tools(self, tool_manager, sample_tool):
        """Test getting available tools."""
        # Add tool to manager
        tool_manager._tools_by_server["test_server"] = [sample_tool]
        tool_manager._tools_by_name[sample_tool.name] = sample_tool
        
        # Get all tools
        tools = await tool_manager.get_available_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        
        # Get tools for specific server
        server_tools = await tool_manager.get_available_tools("test_server")
        assert len(server_tools) == 1
        assert server_tools[0].name == "test_tool"
        
        # Get tools for nonexistent server
        no_tools = await tool_manager.get_available_tools("nonexistent")
        assert len(no_tools) == 0
    
    @patch('tgo.agents.tools.mcp_tool_manager.MCPConnector')
    async def test_call_tool_success(self, mock_connector_class, tool_manager, sample_server_config, 
                                   sample_tool, execution_context):
        """Test successful tool call."""
        # Setup mocks
        mock_connector = AsyncMock()
        mock_connector.call_tool.return_value = {
            "content": [{"type": "text", "text": "Tool result"}],
            "isError": False
        }
        mock_connector_class.return_value = mock_connector
        
        # Setup tool manager
        await tool_manager.register_server(sample_server_config)
        tool_manager._tools_by_name[sample_tool.name] = sample_tool
        tool_manager._connectors["test_server"] = mock_connector
        tool_manager._connections["test_server"] = MCPConnection(
            server_config=sample_server_config,
            status="connected"
        )
        
        # Call tool
        result = await tool_manager.call_tool(
            tool_name="test_tool",
            arguments={"input": "test"},
            context=execution_context,
            user_approved=True
        )
        
        assert result.success is True
        assert result.tool_name == "test_tool"
        assert len(result.content) == 1
    
    async def test_call_nonexistent_tool(self, tool_manager, execution_context):
        """Test calling nonexistent tool."""
        with pytest.raises(MCPToolManagerError, match="Tool not found"):
            await tool_manager.call_tool(
                tool_name="nonexistent_tool",
                arguments={},
                context=execution_context
            )
    
    async def test_disconnect_from_server(self, tool_manager, sample_server_config):
        """Test disconnecting from server."""
        # Setup connection
        tool_manager._connections["test_server"] = MCPConnection(
            server_config=sample_server_config,
            status="connected"
        )
        tool_manager._connectors["test_server"] = AsyncMock()
        
        result = await tool_manager.disconnect_from_server("test_server")
        assert result is True
        assert "test_server" not in tool_manager._connections
    
    async def test_get_server_metrics(self, tool_manager, sample_server_config):
        """Test getting server metrics."""
        # Setup connection
        connection = MCPConnection(
            server_config=sample_server_config,
            status="connected",
            request_count=10,
            error_count=1
        )
        tool_manager._connections["test_server"] = connection
        
        metrics = tool_manager.get_server_metrics("test_server")
        assert metrics is not None
        assert metrics["server_id"] == "test_server"
        assert metrics["status"] == "connected"
        assert metrics["request_count"] == 10
        assert metrics["error_count"] == 1
    
    async def test_get_global_metrics(self, tool_manager):
        """Test getting global metrics."""
        tool_manager._total_requests = 100
        tool_manager._total_errors = 5
        
        metrics = tool_manager.get_global_metrics()
        assert metrics["total_requests"] == 100
        assert metrics["total_errors"] == 5
        assert metrics["error_rate"] == 0.05


class TestMCPConnector:
    """Test cases for MCPConnector."""
    
    @pytest.fixture
    def server_config(self):
        """Create a test server configuration."""
        return MCPServerConfig(
            server_id="test_connector",
            name="Test Connector",
            description="Test connector",
            transport_type="stdio",
            command="echo",
            args=["test"]
        )
    
    def test_connector_creation(self, server_config):
        """Test connector creation."""
        connector = MCPConnector(server_config)
        assert connector.config == server_config
        assert not connector.is_connected()
    
    @patch('tgo.agents.tools.mcp_connector.StdioTransport')
    async def test_connect_success(self, mock_transport_class, server_config):
        """Test successful connection."""
        # Setup mock transport
        mock_transport = AsyncMock()
        mock_transport.connect = AsyncMock()
        mock_transport.is_connected.return_value = True
        mock_transport_class.return_value = mock_transport
        
        connector = MCPConnector(server_config)
        
        # Mock handshake
        with patch.object(connector, '_handshake', new_callable=AsyncMock):
            await connector.connect()
            assert connector._connected is True
    
    async def test_connect_failure(self, server_config):
        """Test connection failure."""
        connector = MCPConnector(server_config)
        
        # Mock transport creation failure
        with patch.object(connector, '_create_transport', side_effect=Exception("Transport error")):
            with pytest.raises(MCPConnectorError, match="Connection failed"):
                await connector.connect()
    
    @patch('tgo.agents.tools.mcp_connector.StdioTransport')
    async def test_tool_call(self, mock_transport_class, server_config):
        """Test tool calling."""
        # Setup mock transport
        mock_transport = AsyncMock()
        mock_transport.send_message = AsyncMock()
        mock_transport_class.return_value = mock_transport
        
        connector = MCPConnector(server_config)
        connector._connected = True
        connector.transport = mock_transport
        
        # Mock request/response
        future = asyncio.Future()
        future.set_result({
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"content": [{"type": "text", "text": "result"}]}
        })
        
        with patch.object(connector, '_send_request', return_value=future.result()):
            result = await connector.call_tool("test_tool", {"param": "value"})
            assert "content" in result
    
    async def test_health_check(self, server_config):
        """Test health check."""
        connector = MCPConnector(server_config)
        connector._connected = False
        
        # Should return False when not connected
        is_healthy = await connector.health_check()
        assert is_healthy is False


class TestMCPSecurityManager:
    """Test cases for MCPSecurityManager."""
    
    @pytest.fixture
    def security_manager(self):
        """Create a test security manager."""
        return MCPSecurityManager()
    
    @pytest.fixture
    def test_policy(self):
        """Create a test security policy."""
        return SecurityPolicy(
            allowed_tools={"safe_tool"},
            denied_tools={"dangerous_tool"},
            max_calls_per_minute=10,
            security_level=SecurityLevel.MEDIUM
        )
    
    @pytest.fixture
    def test_tool(self):
        """Create a test tool."""
        return MCPTool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object"},
            server_id="test_server",
            requires_confirmation=False
        )
    
    @pytest.fixture
    def test_request(self):
        """Create a test request."""
        return MCPToolCallRequest(
            tool_name="test_tool",
            server_id="test_server",
            arguments={"param": "value"},
            agent_id="test_agent",
            user_id="test_user"
        )
    
    @pytest.fixture
    def execution_context(self):
        """Create a test execution context."""
        return ExecutionContext(
            task_id="test_task",
            agent_id="test_agent"
        )
    
    def test_set_get_policy(self, security_manager, test_policy):
        """Test setting and getting security policy."""
        security_manager.set_policy("test_agent", test_policy)
        retrieved_policy = security_manager.get_policy("test_agent")
        assert retrieved_policy == test_policy
    
    async def test_check_permission_allow(self, security_manager, test_policy, test_tool, 
                                        test_request, execution_context):
        """Test permission check that allows access."""
        test_policy.allowed_tools = {"test_tool"}
        security_manager.set_policy("test_agent", test_policy)
        
        permission = await security_manager.check_permission(test_request, test_tool, execution_context)
        assert permission == PermissionAction.ALLOW
    
    async def test_check_permission_deny(self, security_manager, test_policy, test_tool, 
                                       test_request, execution_context):
        """Test permission check that denies access."""
        test_policy.denied_tools = {"test_tool"}
        security_manager.set_policy("test_agent", test_policy)
        
        permission = await security_manager.check_permission(test_request, test_tool, execution_context)
        assert permission == PermissionAction.DENY
    
    async def test_validate_parameters_success(self, security_manager, test_tool, test_request):
        """Test successful parameter validation."""
        test_tool.input_schema = {
            "type": "object",
            "properties": {
                "param": {"type": "string"}
            },
            "required": ["param"]
        }
        
        validated = await security_manager.validate_parameters(test_request, test_tool)
        assert validated == test_request.arguments
    
    async def test_rate_limiting(self, security_manager, test_policy, test_tool, execution_context):
        """Test rate limiting functionality."""
        test_policy.max_calls_per_minute = 1
        security_manager.set_policy("test_agent", test_policy)
        
        # First request should be allowed
        request1 = MCPToolCallRequest(
            tool_name="test_tool",
            server_id="test_server",
            arguments={},
            agent_id="test_agent"
        )
        permission1 = await security_manager.check_permission(request1, test_tool, execution_context)
        assert permission1 == PermissionAction.ALLOW
        
        # Second request should be denied due to rate limit
        request2 = MCPToolCallRequest(
            tool_name="test_tool",
            server_id="test_server",
            arguments={},
            agent_id="test_agent"
        )
        permission2 = await security_manager.check_permission(request2, test_tool, execution_context)
        assert permission2 == PermissionAction.DENY


class TestMCPToolProxy:
    """Test cases for MCPToolProxy."""
    
    @pytest.fixture
    def tool_proxy(self):
        """Create a test tool proxy."""
        return MCPToolProxy()
    
    @pytest.fixture
    def sample_mcp_tool(self):
        """Create a sample MCP tool."""
        return MCPTool(
            name="test_tool",
            title="Test Tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            },
            server_id="test_server"
        )
    
    @pytest.fixture
    def sample_mcp_result(self):
        """Create a sample MCP result."""
        return MCPToolCallResult(
            request_id="test_request",
            tool_name="test_tool",
            server_id="test_server",
            success=True,
            content=[{"type": "text", "text": "Test result"}]
        )
    
    def test_convert_tool_for_google_adk(self, tool_proxy, sample_mcp_tool):
        """Test converting MCP tool to Google ADK format."""
        converted = tool_proxy.convert_tool_for_framework(sample_mcp_tool, "google-adk")
        
        assert converted["name"] == "test_tool"
        assert converted["description"] == "A test tool"
        assert converted["display_name"] == "Test Tool"
        assert "metadata" in converted
        assert converted["metadata"]["mcp_tool"] is True
    
    def test_convert_result_for_google_adk(self, tool_proxy, sample_mcp_result):
        """Test converting MCP result to Google ADK format."""
        converted = tool_proxy.convert_result_for_framework(sample_mcp_result, "google-adk")
        
        assert converted.tool_name == "test_tool"
        assert converted.success is True
        assert "text" in converted.result
    
    def test_validate_tool_arguments(self, tool_proxy, sample_mcp_tool):
        """Test tool argument validation."""
        arguments = {"input": "test value"}
        
        is_valid = tool_proxy.validate_tool_arguments(sample_mcp_tool, arguments, "google-adk")
        assert is_valid is True
        
        # Test with missing required field
        invalid_arguments = {}
        is_valid = tool_proxy.validate_tool_arguments(sample_mcp_tool, invalid_arguments, "google-adk")
        assert is_valid is False
    
    def test_get_supported_frameworks(self, tool_proxy):
        """Test getting supported frameworks."""
        frameworks = tool_proxy.get_supported_frameworks()
        assert "google-adk" in frameworks
        assert "langgraph" in frameworks
        assert "crewai" in frameworks


# Integration tests
class TestMCPIntegration:
    """Integration tests for MCP functionality."""
    
    @pytest.fixture
    async def integrated_setup(self):
        """Setup integrated MCP components."""
        tool_manager = MCPToolManager()
        await tool_manager.initialize()
        
        server_config = MCPServerConfig(
            server_id="integration_test",
            name="Integration Test Server",
            description="Server for integration tests",
            transport_type="stdio",
            command="echo",
            args=["test"],
            trusted=True
        )
        
        yield tool_manager, server_config
        await tool_manager.shutdown()
    
    async def test_end_to_end_tool_call(self, integrated_setup):
        """Test end-to-end tool call flow."""
        tool_manager, server_config = integrated_setup
        
        # Register server
        await tool_manager.register_server(server_config)
        
        # Mock successful connection and tool discovery
        with patch.object(tool_manager, 'connect_to_server', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            
            # Add a mock tool
            test_tool = MCPTool(
                name="integration_tool",
                description="Integration test tool",
                input_schema={"type": "object"},
                server_id="integration_test"
            )
            tool_manager._tools_by_name["integration_tool"] = test_tool
            
            # Mock connector
            mock_connector = AsyncMock()
            mock_connector.call_tool.return_value = {
                "content": [{"type": "text", "text": "Integration test result"}],
                "isError": False
            }
            tool_manager._connectors["integration_test"] = mock_connector
            tool_manager._connections["integration_test"] = MCPConnection(
                server_config=server_config,
                status="connected"
            )
            
            # Execute tool call
            context = ExecutionContext(
                task_id="integration_test",
                agent_id="test_agent"
            )
            
            result = await tool_manager.call_tool(
                tool_name="integration_tool",
                arguments={},
                context=context,
                user_approved=True
            )
            
            assert result.success is True
            assert result.tool_name == "integration_tool"


if __name__ == "__main__":
    pytest.main([__file__])
