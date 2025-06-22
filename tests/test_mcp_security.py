"""
Test cases for MCP security functionality.

This module contains comprehensive tests for MCP security controls including:
- Permission checking
- Parameter validation
- Rate limiting
- Result filtering
- Audit logging
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch

from tgo.agents.tools.mcp_security_manager import (
    MCPSecurityManager, SecurityPolicy, PermissionAction, SecurityLevel,
    RateLimitEntry, MCPSecurityManagerError
)
from tgo.agents.core.models import (
    MCPTool, MCPToolCallRequest, MCPToolCallResult, ExecutionContext
)


class TestSecurityPolicy:
    """Test cases for SecurityPolicy."""
    
    def test_default_policy(self):
        """Test default security policy."""
        policy = SecurityPolicy()
        assert policy.security_level == SecurityLevel.MEDIUM
        assert policy.max_calls_per_minute == 60
        assert policy.max_calls_per_hour == 1000
        assert policy.require_approval_for_untrusted is True
        assert policy.filter_sensitive_data is True
    
    def test_custom_policy(self):
        """Test custom security policy."""
        policy = SecurityPolicy(
            allowed_tools={"tool1", "tool2"},
            denied_tools={"dangerous_tool"},
            max_calls_per_minute=30,
            security_level=SecurityLevel.HIGH,
            require_approval_for_untrusted=False
        )
        
        assert "tool1" in policy.allowed_tools
        assert "dangerous_tool" in policy.denied_tools
        assert policy.max_calls_per_minute == 30
        assert policy.security_level == SecurityLevel.HIGH
        assert policy.require_approval_for_untrusted is False


class TestMCPSecurityManager:
    """Test cases for MCPSecurityManager."""
    
    @pytest.fixture
    def security_manager(self):
        """Create a test security manager."""
        return MCPSecurityManager()
    
    @pytest.fixture
    def restrictive_policy(self):
        """Create a restrictive security policy."""
        return SecurityPolicy(
            allowed_tools={"safe_tool"},
            denied_tools={"dangerous_tool"},
            allowed_servers={"trusted_server"},
            denied_servers={"untrusted_server"},
            max_calls_per_minute=2,
            max_calls_per_hour=10,
            security_level=SecurityLevel.HIGH,
            require_approval_for_untrusted=True,
            filter_sensitive_data=True
        )
    
    @pytest.fixture
    def permissive_policy(self):
        """Create a permissive security policy."""
        return SecurityPolicy(
            security_level=SecurityLevel.LOW,
            max_calls_per_minute=1000,
            require_approval_for_untrusted=False,
            filter_sensitive_data=False
        )
    
    @pytest.fixture
    def test_tool(self):
        """Create a test tool."""
        return MCPTool(
            name="test_tool",
            description="Test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                    "param2": {"type": "number"}
                },
                "required": ["param1"]
            },
            server_id="test_server",
            requires_confirmation=False
        )
    
    @pytest.fixture
    def dangerous_tool(self):
        """Create a dangerous tool."""
        return MCPTool(
            name="dangerous_tool",
            description="Dangerous tool",
            input_schema={"type": "object"},
            server_id="untrusted_server",
            requires_confirmation=True
        )
    
    @pytest.fixture
    def test_request(self):
        """Create a test request."""
        return MCPToolCallRequest(
            tool_name="test_tool",
            server_id="test_server",
            arguments={"param1": "value1", "param2": 42},
            agent_id="test_agent",
            user_id="test_user"
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
    
    def test_policy_management(self, security_manager, restrictive_policy):
        """Test setting and getting policies."""
        # Test default policy
        default_policy = security_manager.get_policy("unknown_agent")
        assert default_policy == security_manager.default_policy
        
        # Test setting custom policy
        security_manager.set_policy("test_agent", restrictive_policy)
        retrieved_policy = security_manager.get_policy("test_agent")
        assert retrieved_policy == restrictive_policy
    
    async def test_permission_allow(self, security_manager, permissive_policy, test_tool, 
                                  test_request, execution_context):
        """Test permission check that allows access."""
        security_manager.set_policy("test_agent", permissive_policy)
        
        permission = await security_manager.check_permission(test_request, test_tool, execution_context)
        assert permission == PermissionAction.ALLOW
    
    async def test_permission_deny_tool(self, security_manager, restrictive_policy, test_tool, 
                                      test_request, execution_context):
        """Test permission denied due to tool restrictions."""
        # Tool not in allowed list
        restrictive_policy.allowed_tools = {"other_tool"}
        security_manager.set_policy("test_agent", restrictive_policy)
        
        permission = await security_manager.check_permission(test_request, test_tool, execution_context)
        assert permission == PermissionAction.DENY
        
        # Tool in denied list
        restrictive_policy.allowed_tools = set()
        restrictive_policy.denied_tools = {"test_tool"}
        
        permission = await security_manager.check_permission(test_request, test_tool, execution_context)
        assert permission == PermissionAction.DENY
    
    async def test_permission_deny_server(self, security_manager, restrictive_policy, test_tool, 
                                        test_request, execution_context):
        """Test permission denied due to server restrictions."""
        # Server not in allowed list
        restrictive_policy.allowed_servers = {"other_server"}
        security_manager.set_policy("test_agent", restrictive_policy)
        
        permission = await security_manager.check_permission(test_request, test_tool, execution_context)
        assert permission == PermissionAction.DENY
        
        # Server in denied list
        restrictive_policy.allowed_servers = set()
        restrictive_policy.denied_servers = {"test_server"}
        
        permission = await security_manager.check_permission(test_request, test_tool, execution_context)
        assert permission == PermissionAction.DENY
    
    async def test_permission_require_approval(self, security_manager, restrictive_policy, 
                                             dangerous_tool, execution_context):
        """Test permission that requires approval."""
        security_manager.set_policy("test_agent", restrictive_policy)
        
        request = MCPToolCallRequest(
            tool_name="dangerous_tool",
            server_id="untrusted_server",
            arguments={},
            agent_id="test_agent",
            user_approved=False
        )
        
        permission = await security_manager.check_permission(request, dangerous_tool, execution_context)
        assert permission == PermissionAction.REQUIRE_APPROVAL
    
    async def test_rate_limiting(self, security_manager, restrictive_policy, test_tool, execution_context):
        """Test rate limiting functionality."""
        security_manager.set_policy("test_agent", restrictive_policy)
        
        # First request should be allowed
        request1 = MCPToolCallRequest(
            tool_name="safe_tool",
            server_id="trusted_server",
            arguments={},
            agent_id="test_agent"
        )
        
        safe_tool = MCPTool(
            name="safe_tool",
            description="Safe tool",
            input_schema={"type": "object"},
            server_id="trusted_server"
        )
        
        permission1 = await security_manager.check_permission(request1, safe_tool, execution_context)
        assert permission1 == PermissionAction.ALLOW
        
        # Second request should be allowed (within limit)
        request2 = MCPToolCallRequest(
            tool_name="safe_tool",
            server_id="trusted_server",
            arguments={},
            agent_id="test_agent"
        )
        
        permission2 = await security_manager.check_permission(request2, safe_tool, execution_context)
        assert permission2 == PermissionAction.ALLOW
        
        # Third request should be denied (exceeds limit of 2 per minute)
        request3 = MCPToolCallRequest(
            tool_name="safe_tool",
            server_id="trusted_server",
            arguments={},
            agent_id="test_agent"
        )
        
        permission3 = await security_manager.check_permission(request3, safe_tool, execution_context)
        assert permission3 == PermissionAction.DENY
    
    async def test_parameter_validation_success(self, security_manager, test_tool, test_request):
        """Test successful parameter validation."""
        validated = await security_manager.validate_parameters(test_request, test_tool)
        assert validated == test_request.arguments
        assert "param1" in validated
        assert "param2" in validated
    
    async def test_parameter_validation_missing_required(self, security_manager, test_tool):
        """Test parameter validation with missing required field."""
        request = MCPToolCallRequest(
            tool_name="test_tool",
            server_id="test_server",
            arguments={"param2": 42},  # Missing required param1
            agent_id="test_agent"
        )
        
        with pytest.raises(MCPSecurityManagerError, match="Parameters do not match tool schema"):
            await security_manager.validate_parameters(request, test_tool)
    
    async def test_parameter_validation_size_limit(self, security_manager, test_tool):
        """Test parameter validation with size limit."""
        # Create large parameters
        large_data = "x" * (1024 * 1024 + 1)  # Exceed 1MB limit
        request = MCPToolCallRequest(
            tool_name="test_tool",
            server_id="test_server",
            arguments={"param1": large_data},
            agent_id="test_agent"
        )
        
        with pytest.raises(MCPSecurityManagerError, match="Parameters too large"):
            await security_manager.validate_parameters(request, test_tool)
    
    async def test_result_filtering(self, security_manager, test_request):
        """Test result filtering."""
        result = MCPToolCallResult(
            request_id="test_request",
            tool_name="test_tool",
            server_id="test_server",
            success=True,
            content=[{"type": "text", "text": "Test result"}]
        )
        
        filtered_result = await security_manager.filter_result(result, test_request)
        assert filtered_result.success is True
        assert filtered_result.tool_name == "test_tool"
    
    async def test_result_filtering_size_limit(self, security_manager, test_request):
        """Test result filtering with size limit."""
        # Create large result
        large_content = [{"type": "text", "text": "x" * (10 * 1024 * 1024 + 1)}]  # Exceed 10MB
        result = MCPToolCallResult(
            request_id="test_request",
            tool_name="test_tool",
            server_id="test_server",
            success=True,
            content=large_content
        )
        
        filtered_result = await security_manager.filter_result(result, test_request)
        # Should be truncated
        assert len(filtered_result.content) == 1
        assert "truncated" in filtered_result.content[0]["text"].lower()
    
    def test_audit_logging(self, security_manager, test_request):
        """Test audit logging functionality."""
        # Initially empty
        log_entries = security_manager.get_audit_log()
        initial_count = len(log_entries)
        
        # Add audit entry
        asyncio.run(security_manager._audit_log_entry(
            "test_event", test_request, "Test message"
        ))
        
        # Check log entry was added
        log_entries = security_manager.get_audit_log()
        assert len(log_entries) == initial_count + 1
        
        latest_entry = log_entries[-1]
        assert latest_entry["event_type"] == "test_event"
        assert latest_entry["agent_id"] == "test_agent"
        assert latest_entry["tool_name"] == "test_tool"
        assert latest_entry["message"] == "Test message"
    
    def test_rate_limit_status(self, security_manager, restrictive_policy):
        """Test rate limit status reporting."""
        security_manager.set_policy("test_agent", restrictive_policy)
        
        # Initially no status
        status = security_manager.get_rate_limit_status("test_agent")
        assert status is None
        
        # Create rate limit entry
        security_manager._rate_limits["test_agent"] = RateLimitEntry(
            calls_this_minute=5,
            calls_this_hour=25
        )
        
        status = security_manager.get_rate_limit_status("test_agent")
        assert status is not None
        assert status["calls_this_minute"] == 5
        assert status["calls_this_hour"] == 25
        assert status["max_calls_per_minute"] == 2
        assert status["max_calls_per_hour"] == 10


class TestSecurityIntegration:
    """Integration tests for security functionality."""
    
    @pytest.fixture
    def security_manager(self):
        """Create security manager with test policies."""
        manager = MCPSecurityManager()
        
        # Set up different policies for different agents
        strict_policy = SecurityPolicy(
            allowed_tools={"safe_tool"},
            max_calls_per_minute=1,
            security_level=SecurityLevel.HIGH,
            require_approval_for_untrusted=True
        )
        
        lenient_policy = SecurityPolicy(
            security_level=SecurityLevel.LOW,
            max_calls_per_minute=100,
            require_approval_for_untrusted=False
        )
        
        manager.set_policy("strict_agent", strict_policy)
        manager.set_policy("lenient_agent", lenient_policy)
        
        return manager
    
    async def test_multi_agent_permissions(self, security_manager):
        """Test permissions for multiple agents."""
        tool = MCPTool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object"},
            server_id="test_server"
        )
        
        context = ExecutionContext(task_id="test", agent_id="test")
        
        # Strict agent - should be denied (tool not in allowed list)
        strict_request = MCPToolCallRequest(
            tool_name="test_tool",
            server_id="test_server",
            arguments={},
            agent_id="strict_agent"
        )
        
        permission = await security_manager.check_permission(strict_request, tool, context)
        assert permission == PermissionAction.DENY
        
        # Lenient agent - should be allowed
        lenient_request = MCPToolCallRequest(
            tool_name="test_tool",
            server_id="test_server",
            arguments={},
            agent_id="lenient_agent"
        )
        
        permission = await security_manager.check_permission(lenient_request, tool, context)
        assert permission == PermissionAction.ALLOW
    
    async def test_rate_limit_per_agent(self, security_manager):
        """Test that rate limits are enforced per agent."""
        tool = MCPTool(
            name="safe_tool",
            description="Safe tool",
            input_schema={"type": "object"},
            server_id="test_server"
        )
        
        context = ExecutionContext(task_id="test", agent_id="test")
        
        # Strict agent has limit of 1 per minute
        strict_request = MCPToolCallRequest(
            tool_name="safe_tool",
            server_id="test_server",
            arguments={},
            agent_id="strict_agent"
        )
        
        # First call should be allowed
        permission1 = await security_manager.check_permission(strict_request, tool, context)
        assert permission1 == PermissionAction.ALLOW
        
        # Second call should be denied (rate limit)
        permission2 = await security_manager.check_permission(strict_request, tool, context)
        assert permission2 == PermissionAction.DENY
        
        # Lenient agent should still be allowed (different rate limit)
        lenient_request = MCPToolCallRequest(
            tool_name="safe_tool",
            server_id="test_server",
            arguments={},
            agent_id="lenient_agent"
        )
        
        permission3 = await security_manager.check_permission(lenient_request, tool, context)
        assert permission3 == PermissionAction.ALLOW


if __name__ == "__main__":
    pytest.main([__file__])
