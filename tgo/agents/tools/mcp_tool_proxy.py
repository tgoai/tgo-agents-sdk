"""
MCP Tool Proxy for adapting MCP tools to different AI framework formats.

This module provides translation between MCP tool definitions and the native
tool formats expected by different AI frameworks like Google ADK, LangGraph, CrewAI, etc.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod

from ..core.models import MCPTool, MCPToolCallResult, ToolCallResult
from ..core.exceptions import MultiAgentError

logger = logging.getLogger(__name__)


class MCPToolProxyError(MultiAgentError):
    """MCP Tool Proxy specific error."""
    pass


class FrameworkToolAdapter(ABC):
    """Base class for framework-specific tool adapters."""
    
    @abstractmethod
    def convert_tool_definition(self, mcp_tool: MCPTool) -> Dict[str, Any]:
        """Convert MCP tool definition to framework-specific format."""
        pass
    
    @abstractmethod
    def validate_arguments(self, mcp_tool: MCPTool, arguments: Dict[str, Any]) -> bool:
        """Validate tool arguments against schema."""
        pass


class GoogleADKToolAdapter(FrameworkToolAdapter):
    """Adapter for Google ADK tools."""
    
    def convert_tool_definition(self, mcp_tool: MCPTool) -> Dict[str, Any]:
        """Convert MCP tool to Google ADK tool format."""
        try:
            # Google ADK tool definition format
            tool_def = {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": self._convert_schema_to_adk(mcp_tool.input_schema),
                "required": mcp_tool.input_schema.get("required", [])
            }
            
            if mcp_tool.title:
                tool_def["display_name"] = mcp_tool.title
            
            # Add metadata
            tool_def["metadata"] = {
                "server_id": mcp_tool.server_id,
                "mcp_tool": True,
                "requires_confirmation": mcp_tool.requires_confirmation,
                "annotations": mcp_tool.annotations
            }
            
            return tool_def
            
        except Exception as e:
            logger.error(f"Failed to convert MCP tool {mcp_tool.name} to Google ADK format: {e}")
            raise MCPToolProxyError(f"Tool conversion failed: {e}")
    
    def validate_arguments(self, mcp_tool: MCPTool, arguments: Dict[str, Any]) -> bool:
        """Validate arguments against MCP tool schema."""
        try:
            schema = mcp_tool.input_schema
            
            # Check required fields
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in arguments:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Basic type checking for properties
            properties = schema.get("properties", {})
            for field, value in arguments.items():
                if field in properties:
                    expected_type = properties[field].get("type")
                    if expected_type and not self._validate_type(value, expected_type):
                        logger.error(f"Invalid type for field {field}: expected {expected_type}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Argument validation failed: {e}")
            return False
    
    def _convert_schema_to_adk(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON Schema to Google ADK parameter format."""
        properties = schema.get("properties", {})
        adk_params = {}
        
        for name, prop in properties.items():
            adk_param = {
                "type": prop.get("type", "string"),
                "description": prop.get("description", "")
            }
            
            # Handle enum values
            if "enum" in prop:
                adk_param["enum"] = prop["enum"]
            
            # Handle default values
            if "default" in prop:
                adk_param["default"] = prop["default"]
            
            # Handle format constraints
            if "format" in prop:
                adk_param["format"] = prop["format"]
            
            adk_params[name] = adk_param
        
        return adk_params
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type against JSON Schema type."""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, allow it


class LangGraphToolAdapter(FrameworkToolAdapter):
    """Adapter for LangGraph tools."""
    
    def convert_tool_definition(self, mcp_tool: MCPTool) -> Dict[str, Any]:
        """Convert MCP tool to LangGraph tool format."""
        return {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "args_schema": mcp_tool.input_schema,
            "metadata": {
                "server_id": mcp_tool.server_id,
                "mcp_tool": True,
                "requires_confirmation": mcp_tool.requires_confirmation
            }
        }
    
    def validate_arguments(self, mcp_tool: MCPTool, arguments: Dict[str, Any]) -> bool:
        """Validate arguments for LangGraph."""
        # Use same validation as Google ADK for now
        return GoogleADKToolAdapter().validate_arguments(mcp_tool, arguments)


class CrewAIToolAdapter(FrameworkToolAdapter):
    """Adapter for CrewAI tools."""
    
    def convert_tool_definition(self, mcp_tool: MCPTool) -> Dict[str, Any]:
        """Convert MCP tool to CrewAI tool format."""
        return {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.input_schema,
            "metadata": {
                "server_id": mcp_tool.server_id,
                "mcp_tool": True
            }
        }
    
    def validate_arguments(self, mcp_tool: MCPTool, arguments: Dict[str, Any]) -> bool:
        """Validate arguments for CrewAI."""
        return GoogleADKToolAdapter().validate_arguments(mcp_tool, arguments)


class MCPToolProxy:
    """
    Proxy for adapting MCP tools to different AI framework formats.
    
    This class provides a unified interface for converting MCP tools and results
    to the native formats expected by different AI frameworks.
    """
    
    def __init__(self):
        self._adapters: Dict[str, FrameworkToolAdapter] = {
            "google-adk": GoogleADKToolAdapter(),
            "langgraph": LangGraphToolAdapter(),
            "crewai": CrewAIToolAdapter()
        }
        
        logger.info("MCPToolProxy initialized with adapters: %s", list(self._adapters.keys()))
    
    def register_adapter(self, framework_name: str, adapter: FrameworkToolAdapter) -> None:
        """Register a custom framework adapter."""
        self._adapters[framework_name] = adapter
        logger.info(f"Registered custom adapter for framework: {framework_name}")
    
    # def convert_tool_for_framework(self, mcp_tool: MCPTool, framework: str) -> Dict[str, Any]:
    #     """
    #     Convert MCP tool definition to framework-specific format.
        
    #     Args:
    #         mcp_tool: MCP tool definition
    #         framework: Target framework name
            
    #     Returns:
    #         Framework-specific tool definition
            
    #     Raises:
    #         MCPToolProxyError: If conversion fails
    #     """
    #     if framework not in self._adapters:
    #         raise MCPToolProxyError(f"No adapter found for framework: {framework}")
        
    #     try:
    #         adapter = self._adapters[framework]
    #         return adapter.convert_tool_definition(mcp_tool)
            
    #     except Exception as e:
    #         logger.error(f"Tool conversion failed for {framework}: {e}")
    #         raise MCPToolProxyError(f"Tool conversion failed: {e}")
    
    def validate_tool_arguments(
        self, 
        mcp_tool: MCPTool, 
        arguments: Dict[str, Any], 
        framework: str
    ) -> bool:
        """
        Validate tool arguments for a specific framework.
        
        Args:
            mcp_tool: MCP tool definition
            arguments: Tool arguments to validate
            framework: Target framework name
            
        Returns:
            True if arguments are valid
        """
        if framework not in self._adapters:
            logger.warning(f"No adapter found for framework {framework}, skipping validation")
            return True
        
        try:
            adapter = self._adapters[framework]
            return adapter.validate_arguments(mcp_tool, arguments)
            
        except Exception as e:
            logger.error(f"Argument validation failed for {framework}: {e}")
            return False
    
    def get_supported_frameworks(self) -> List[str]:
        """Get list of supported frameworks."""
        return list(self._adapters.keys())
