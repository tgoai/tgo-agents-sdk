"""
Tools package for the multi-agent system.

This package provides MCP (Model Context Protocol) integration and tool management
capabilities for the multi-agent system.
"""

from .fastmcp_tool_manager import MCPToolManager
from .mcp_tool_proxy import MCPToolProxy

__all__ = [
    "MCPToolManager",
    "MCPToolProxy",
]
