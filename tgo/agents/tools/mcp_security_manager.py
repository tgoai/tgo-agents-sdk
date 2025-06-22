"""
MCP Security Manager for controlling access and validating MCP tool operations.

This module provides security controls for MCP tool usage including:
- Permission checking
- Parameter validation and sanitization
- Result filtering
- Rate limiting
- Audit logging
"""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ..core.models import MCPTool, MCPToolCallRequest, MCPToolCallResult, ExecutionContext
from ..core.exceptions import MultiAgentError

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security levels for MCP operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PermissionAction(str, Enum):
    """Permission actions."""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class SecurityPolicy:
    """Security policy for MCP tools."""
    
    # Tool access control
    allowed_tools: Set[str] = field(default_factory=set)
    denied_tools: Set[str] = field(default_factory=set)
    
    # Server access control
    allowed_servers: Set[str] = field(default_factory=set)
    denied_servers: Set[str] = field(default_factory=set)
    
    # User/Agent access control
    allowed_agents: Set[str] = field(default_factory=set)
    allowed_users: Set[str] = field(default_factory=set)
    
    # Rate limiting
    max_calls_per_minute: int = 60
    max_calls_per_hour: int = 1000
    
    # Parameter validation
    max_parameter_size: int = 1024 * 1024  # 1MB
    allowed_parameter_types: Set[str] = field(default_factory=lambda: {"string", "number", "boolean", "array", "object"})
    
    # Result filtering
    max_result_size: int = 10 * 1024 * 1024  # 10MB
    filter_sensitive_data: bool = True
    
    # Security level
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    
    # Approval requirements
    require_approval_for_untrusted: bool = True
    require_approval_for_high_risk: bool = True
    
    # Audit settings
    audit_all_calls: bool = True
    audit_failures_only: bool = False


@dataclass
class RateLimitEntry:
    """Rate limiting entry."""
    calls_this_minute: int = 0
    calls_this_hour: int = 0
    last_minute_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_hour_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MCPSecurityManagerError(MultiAgentError):
    """MCP Security Manager specific error."""
    pass


class MCPSecurityManager:
    """
    Security manager for MCP tool operations.
    
    Provides comprehensive security controls including:
    - Access control and permissions
    - Parameter validation and sanitization
    - Rate limiting
    - Result filtering
    - Audit logging
    """
    
    def __init__(self, default_policy: Optional[SecurityPolicy] = None):
        self.default_policy = default_policy or SecurityPolicy()
        self._policies: Dict[str, SecurityPolicy] = {}  # agent_id -> policy
        self._rate_limits: Dict[str, RateLimitEntry] = {}  # agent_id -> rate limit
        self._audit_log: List[Dict[str, Any]] = []
        self._sensitive_patterns = [
            r'password', r'secret', r'key', r'token', r'credential',
            r'ssn', r'social.*security', r'credit.*card', r'bank.*account'
        ]
        
        logger.info("MCPSecurityManager initialized")
    
    def set_policy(self, agent_id: str, policy: SecurityPolicy) -> None:
        """Set security policy for a specific agent."""
        self._policies[agent_id] = policy
        logger.info(f"Security policy set for agent: {agent_id}")
    
    def get_policy(self, agent_id: str) -> SecurityPolicy:
        """Get security policy for an agent."""
        return self._policies.get(agent_id, self.default_policy)
    
    async def check_permission(
        self,
        request: MCPToolCallRequest,
        tool: MCPTool,
        context: ExecutionContext
    ) -> PermissionAction:
        """
        Check if a tool call is permitted.
        
        Args:
            request: Tool call request
            tool: Tool definition
            context: Execution context
            
        Returns:
            Permission action (allow, deny, require_approval)
        """
        try:
            policy = self.get_policy(request.agent_id or "")
            
            # Check tool access
            if tool.name in policy.denied_tools:
                await self._audit_log_entry("permission_denied", request, "Tool in denied list")
                return PermissionAction.DENY
            
            if policy.allowed_tools and tool.name not in policy.allowed_tools:
                await self._audit_log_entry("permission_denied", request, "Tool not in allowed list")
                return PermissionAction.DENY
            
            # Check server access
            if tool.server_id in policy.denied_servers:
                await self._audit_log_entry("permission_denied", request, "Server in denied list")
                return PermissionAction.DENY
            
            if policy.allowed_servers and tool.server_id not in policy.allowed_servers:
                await self._audit_log_entry("permission_denied", request, "Server not in allowed list")
                return PermissionAction.DENY
            
            # Check agent access
            if policy.allowed_agents and request.agent_id not in policy.allowed_agents:
                await self._audit_log_entry("permission_denied", request, "Agent not in allowed list")
                return PermissionAction.DENY
            
            # Check user access
            if policy.allowed_users and request.user_id not in policy.allowed_users:
                await self._audit_log_entry("permission_denied", request, "User not in allowed list")
                return PermissionAction.DENY
            
            # Check rate limits
            if not await self._check_rate_limits(request.agent_id or "", policy):
                await self._audit_log_entry("rate_limit_exceeded", request, "Rate limit exceeded")
                return PermissionAction.DENY
            
            # Check if approval is required
            if self._requires_approval(tool, policy, request):
                await self._audit_log_entry("approval_required", request, "Tool requires approval")
                return PermissionAction.REQUIRE_APPROVAL
            
            await self._audit_log_entry("permission_granted", request, "Permission granted")
            return PermissionAction.ALLOW
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            await self._audit_log_entry("permission_error", request, f"Permission check error: {e}")
            return PermissionAction.DENY
    
    async def validate_parameters(
        self,
        request: MCPToolCallRequest,
        tool: MCPTool
    ) -> Dict[str, Any]:
        """
        Validate and sanitize tool parameters.
        
        Args:
            request: Tool call request
            tool: Tool definition
            
        Returns:
            Validated and sanitized parameters
            
        Raises:
            MCPSecurityManagerError: If validation fails
        """
        try:
            policy = self.get_policy(request.agent_id or "")
            parameters = request.arguments.copy()
            
            # Check parameter size
            param_size = len(json.dumps(parameters))
            if param_size > policy.max_parameter_size:
                raise MCPSecurityManagerError(f"Parameters too large: {param_size} bytes")
            
            # Validate against tool schema
            if not self._validate_against_schema(parameters, tool.input_schema):
                raise MCPSecurityManagerError("Parameters do not match tool schema")
            
            # Sanitize parameters
            sanitized_params = await self._sanitize_parameters(parameters, policy)
            
            await self._audit_log_entry("parameters_validated", request, "Parameters validated successfully")
            return sanitized_params
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            await self._audit_log_entry("validation_failed", request, f"Parameter validation failed: {e}")
            raise MCPSecurityManagerError(f"Parameter validation failed: {e}")
    
    async def filter_result(
        self,
        result: MCPToolCallResult,
        request: MCPToolCallRequest
    ) -> MCPToolCallResult:
        """
        Filter and sanitize tool call result.
        
        Args:
            result: Tool call result
            request: Original request
            
        Returns:
            Filtered result
        """
        try:
            policy = self.get_policy(request.agent_id or "")
            
            # Check result size
            result_size = len(result.text)
            if result_size > policy.max_result_size:
                logger.warning(f"Result too large ({result_size} bytes), truncating")
                # Truncate result
                result.text =  "Result truncated due to size limit"
                result.content = []
            
            # Filter sensitive data if enabled
            if policy.filter_sensitive_data:
                result = await self._filter_sensitive_data(result)
            
            await self._audit_log_entry("result_filtered", request, "Result filtered successfully")
            return result
            
        except Exception as e:
            logger.error(f"Result filtering failed: {e}")
            await self._audit_log_entry("filtering_failed", request, f"Result filtering failed: {e}")
            return result  # Return original result if filtering fails
    
    async def _check_rate_limits(self, agent_id: str, policy: SecurityPolicy) -> bool:
        """Check if agent is within rate limits."""
        now = datetime.now(timezone.utc)
        
        if agent_id not in self._rate_limits:
            self._rate_limits[agent_id] = RateLimitEntry()
        
        entry = self._rate_limits[agent_id]
        
        # Reset counters if needed
        if now - entry.last_minute_reset >= timedelta(minutes=1):
            entry.calls_this_minute = 0
            entry.last_minute_reset = now
        
        if now - entry.last_hour_reset >= timedelta(hours=1):
            entry.calls_this_hour = 0
            entry.last_hour_reset = now
        
        # Check limits
        if entry.calls_this_minute >= policy.max_calls_per_minute:
            return False
        
        if entry.calls_this_hour >= policy.max_calls_per_hour:
            return False
        
        # Increment counters
        entry.calls_this_minute += 1
        entry.calls_this_hour += 1
        
        return True
    
    def _requires_approval(
        self,
        tool: MCPTool,
        policy: SecurityPolicy,
        request: MCPToolCallRequest
    ) -> bool:
        """Check if tool call requires approval."""
        # Already approved
        if request.user_approved:
            return False
        
        # Tool requires confirmation
        if tool.requires_confirmation:
            return True
        
        # Policy requires approval for untrusted servers
        if policy.require_approval_for_untrusted and not tool.server_id.endswith("_trusted"):
            return True
        
        # High security level requires approval
        if policy.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            return True
        
        return False
    
    def _validate_against_schema(self, parameters: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate parameters against JSON schema."""
        try:
            # Basic validation - in production, use a proper JSON schema validator
            required_fields = schema.get("required", [])
            properties = schema.get("properties", {})
            
            # Check required fields
            for field in required_fields:
                if field not in parameters:
                    return False
            
            # Check field types
            for field, value in parameters.items():
                if field in properties:
                    expected_type = properties[field].get("type")
                    if expected_type and not self._check_type(value, expected_type):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
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
    
    async def _sanitize_parameters(
        self,
        parameters: Dict[str, Any],
        policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Sanitize parameters to remove potentially harmful content."""
        # In a real implementation, this would include:
        # - SQL injection prevention
        # - XSS prevention
        # - Path traversal prevention
        # - Command injection prevention
        
        # For now, just return the parameters as-is
        return parameters
    
    async def _filter_sensitive_data(self, result: MCPToolCallResult) -> MCPToolCallResult:
        """Filter sensitive data from result."""
        # In a real implementation, this would scan for and redact:
        # - Personal information
        # - Credentials
        # - Financial data
        # - Other sensitive patterns
        
        # For now, just return the result as-is
        return result
    
    async def _audit_log_entry(
        self,
        event_type: str,
        request: MCPToolCallRequest,
        message: str
    ) -> None:
        """Add entry to audit log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "agent_id": request.agent_id,
            "user_id": request.user_id,
            "tool_name": request.tool_name,
            "server_id": request.server_id,
            "message": message,
            "request_id": request.request_id
        }
        
        self._audit_log.append(entry)
        
        # Keep only recent entries (last 1000)
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]
        
        logger.debug(f"Audit log entry: {event_type} - {message}")
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]
    
    def get_rate_limit_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get rate limit status for an agent."""
        if agent_id not in self._rate_limits:
            return None
        
        entry = self._rate_limits[agent_id]
        policy = self.get_policy(agent_id)
        
        return {
            "calls_this_minute": entry.calls_this_minute,
            "calls_this_hour": entry.calls_this_hour,
            "max_calls_per_minute": policy.max_calls_per_minute,
            "max_calls_per_hour": policy.max_calls_per_hour,
            "last_minute_reset": entry.last_minute_reset.isoformat(),
            "last_hour_reset": entry.last_hour_reset.isoformat()
        }
