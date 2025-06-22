"""
FastMCP Tool Manager for managing Model Context Protocol connections and tools.

This module provides centralized management of MCP server connections using FastMCP's
multi-server client architecture, tool discovery, and tool execution coordination.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone

from fastmcp import Client
from mcp.types import Content, TextContent
from ..core.models import (
    MCPTool, MCPToolCallRequest, MCPToolCallResult,
    ExecutionContext
)
from ..core.exceptions import MultiAgentError
from .mcp_tool_proxy import MCPToolProxy
from .mcp_security_manager import MCPSecurityManager, PermissionAction

logger = logging.getLogger(__name__)


class MCPToolManagerError(MultiAgentError):
    """MCP Tool Manager specific error."""
    pass


class MCPToolManager:
    """
    Centralized manager for MCP (Model Context Protocol) tools and connections using FastMCP.
    
    This class handles:
    - Multi-server MCP client management using FastMCP
    - Tool discovery and caching
    - Tool execution routing
    - Security and permission management
    
    Usage:
        config = {
            "mcpServers": {
                "math_server": {
                    "command": "python",
                    "args": ["./fastmcp_simple_server.py"],
                    "env": {"DEBUG": "true"}
                },
                "weather": {
                    "url": "https://weather-api.example.com/mcp",
                    "transport": "streamable-http"
                }
            }
        }
        mcp_manager = MCPToolManager(config)
    """
    
    def __init__(self, config: Dict[str, Any], security_manager: Optional[MCPSecurityManager] = None):
        """
        Initialize MCP Tool Manager with FastMCP multi-server configuration.
        
        Args:
            config: Configuration dict with mcpServers section
            security_manager: Optional security manager
        """
        self.config = config
        self._security_manager = security_manager or MCPSecurityManager()
        self._tool_proxy = MCPToolProxy()
        
        # FastMCP multi-server client
        self._client: Optional[Client] = None
        self._initialized = False
        
        # Tool caching and indexing
        self._tools_by_server: Dict[str, List[MCPTool]] = {}
        self._tools_by_name: Dict[str, MCPTool] = {}
        self._tool_permissions: Dict[str, Set[str]] = {}  # tool_name -> allowed_contexts
        
        # Server configurations from config
        self._server_configs: Dict[str, Dict[str, Any]] = config.get("mcpServers", {})
        
        # Metrics
        self._total_requests = 0
        self._total_errors = 0
        # Create FastMCP multi-server client using MCPConfig format
        # FastMCP will automatically handle multiple servers with prefixing
        self._client = Client(self.config)
        self._initialized = True
        
        logger.info(f"MCPToolManager initialized with {len(self._server_configs)} servers")

    async def call_tool(
        self,
        tool: MCPTool,
        arguments: Dict[str, Any],
        context: ExecutionContext,
        user_approved: bool = False
    ) -> MCPToolCallResult:
        """
        Call an MCP tool with security controls using FastMCP.

        Args:
            tool_name: Name of tool to call
            arguments: Tool arguments
            context: Execution context
            user_approved: Whether user approved the call

        Returns:
            Tool call result

        Raises:
            MCPToolManagerError: If tool call fails
        """
        full_tool_name = tool.name
        if tool.server_id:
            full_tool_name = f"{tool.server_id}_{tool.name}"
         
        # Create request
        request = MCPToolCallRequest(
            tool_name=full_tool_name,
            server_id=tool.server_id,
            arguments=arguments,
            agent_id=context.agent_id,
            session_id=context.session_id,
            user_id=context.user_id,
            user_approved=user_approved,
            approval_timestamp=datetime.now(timezone.utc) if user_approved else None
        )

        # Security permission check
        permission = await self._security_manager.check_permission(request, tool, context)

        if permission == PermissionAction.DENY:
            raise MCPToolManagerError(f"Permission denied for tool: {full_tool_name}")

        if permission == PermissionAction.REQUIRE_APPROVAL and not user_approved:
            raise MCPToolManagerError(f"Tool {full_tool_name} requires user approval")

        # Validate and sanitize parameters
        try:
            validated_arguments = await self._security_manager.validate_parameters(request, tool)
            request.arguments = validated_arguments
        except Exception as e:
            raise MCPToolManagerError(f"Parameter validation failed: {e}")

        try:
            # Execute tool call using FastMCP
            if not self._client:
                raise MCPToolManagerError("Client not initialized")

            async with self._client as client:
                # For multi-server setups, FastMCP handles routing automatically
                # based on the tool name prefix
                print("tool.name----->",tool.name)
                result = await client.call_tool(tool.name, request.arguments)

            # Convert FastMCP result to our format
            print("result------>",result)
            mcp_result = self._convert_fastmcp_result(request, result)
            print("mcp_result------>",mcp_result)
            # Apply security filtering to result
            mcp_result = await self._security_manager.filter_result(mcp_result, request)

            # Update metrics
            self._total_requests += 1
            if not mcp_result.success:
                self._total_errors += 1

            # Update tool usage
            tool.usage_count += 1
            tool.last_used = datetime.now(timezone.utc)

            logger.info(f"MCP tool call completed: {full_tool_name} (success: {mcp_result.success})")
            return mcp_result

        except Exception as e:
            self._total_errors += 1
            logger.error(f"MCP tool call failed: {full_tool_name} - {e}")
            raise MCPToolManagerError(f"Tool call failed: {e}")
    
    def _convert_fastmcp_result(self, request: MCPToolCallRequest, result: list[Content]) -> MCPToolCallResult:
        """Convert FastMCP result to MCPToolCallResult format."""
        try:
            if not result:
                raise Exception("No result returned from FastMCP")
            
            if not isinstance(result[0], TextContent):
                raise TypeError("Expected TextContent type")
            textContent: TextContent = result[0]
            
            return MCPToolCallResult(
                request_id=request.request_id,
                tool_name=request.tool_name,
                server_id=request.server_id,
                success= True,
                content=result,
                text = textContent.text,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc)
            )

        except Exception as e:
            logger.error(f"Failed to convert FastMCP result: {e}")
            return MCPToolCallResult(
                request_id=request.request_id,
                tool_name=request.tool_name,
                server_id=request.server_id,
                success=False,
                content=[],
                is_error=True,
                error_message=f"Result conversion failed: {e}",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc)
            )

    def get_available_tools(self) -> List[MCPTool]:
        """Get all available tools."""
        return list(self._tools_by_name.values())

    def get_tools_by_server(self, server_id: str) -> List[MCPTool]:
        """Get tools for a specific server."""
        return self._tools_by_server.get(server_id, [])

    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get a specific tool by name."""
        return self._tools_by_name.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is available."""
        return tool_name in self._tools_by_name

    def get_server_ids(self) -> List[str]:
        """Get all configured server IDs."""
        return list(self._server_configs.keys())

    async def check_tool_permission(
        self,
        agent_id: str,
        tool_name: str,
        context: ExecutionContext
    ) -> str:
        """
        Check if an agent has permission to use a tool.

        Args:
            agent_id: Agent identifier
            tool_name: Tool name
            context: Execution context

        Returns:
            Permission action: "allow", "deny", or "require_approval"
        """
        if tool_name not in self._tools_by_name:
            return "deny"

        tool = self._tools_by_name[tool_name]

        # Create a dummy request for permission check
        request = MCPToolCallRequest(
            tool_name=tool_name,
            server_id=tool.server_id,
            arguments={},
            agent_id=agent_id,
            session_id=context.session_id,
            user_id=context.user_id,
            user_approved=False
        )

        permission = await self._security_manager.check_permission(request, tool, context)
        return permission.value

    def get_metrics(self) -> Dict[str, Any]:
        """Get manager metrics."""
        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / max(self._total_requests, 1),
            "total_tools": len(self._tools_by_name),
            "total_servers": len(self._server_configs),
            "initialized": self._initialized
        }

    async def shutdown(self) -> None:
        """Shutdown the MCP tool manager."""
        try:
            if self._client:
                # FastMCP client will be closed automatically when context exits
                pass

            self._initialized = False
            logger.info("MCPToolManager shutdown completed")

        except Exception as e:
            logger.error(f"Error during MCPToolManager shutdown: {e}")


# Compatibility alias for existing code
FastMCPToolManager = MCPToolManager
