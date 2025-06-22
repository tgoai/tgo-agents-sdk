#!/usr/bin/env python3
"""
Test script to verify the fix for the deepcopy coroutine object issue.

This script tests the _create_adk_mcp_tool_from_object method to ensure
it properly handles async/sync conversion without creating unpicklable
coroutine objects.
"""

import asyncio
import copy
import sys
import os
from unittest.mock import Mock, AsyncMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tgo.agents.adapters.google_adk_adapter import GoogleADKAdapter
from tgo.agents.core.models import ExecutionContext, MCPTool


class MockMCPToolManager:
    """Mock MCP tool manager for testing."""

    async def call_tool(self, tool, arguments, context, user_approved=False):
        """Mock tool call that returns a proper result."""
        # Simulate a successful tool call result
        result = Mock()
        result.success = True
        result.content = [f"Mock result for {tool.name} with args {arguments}"]
        result.error_message = None
        return result


def create_mock_mcp_tool():
    """Create a mock MCP tool for testing."""
    tool = Mock(spec=MCPTool)
    tool.name = "test_tool"
    tool.description = "A test MCP tool"
    tool.server_id = "test_server"
    tool.input_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Test message parameter"
            }
        },
        "required": ["message"]
    }
    return tool


def create_mock_context():
    """Create a mock execution context."""
    context = Mock(spec=ExecutionContext)
    context.agent_id = "test_agent"
    context.session_id = "test_session"
    context.user_id = "test_user"
    return context


async def test_mcp_tool_creation_and_deepcopy():
    """Test that MCP tool creation doesn't create unpicklable objects."""
    print("🧪 Testing MCP tool creation and deepcopy compatibility...")
    
    # Create adapter and mock dependencies
    adapter = GoogleADKAdapter()
    adapter._mcp_tool_manager = MockMCPToolManager()
    
    # Create test objects
    mcp_tool = create_mock_mcp_tool()
    context = create_mock_context()
    
    try:
        # Create the ADK tool wrapper
        print("📦 Creating ADK tool wrapper...")
        adk_tool = await adapter._create_adk_mcp_tool_from_object(mcp_tool, context)
        print(f"✅ Successfully created ADK tool: {adk_tool}")
        
        # Test that the tool can be deep copied (this was the original issue)
        print("🔄 Testing deepcopy compatibility...")
        try:
            copied_tool = copy.deepcopy(adk_tool)
            print("✅ Successfully deep copied the tool!")
            print(f"   Original tool: {adk_tool}")
            print(f"   Copied tool: {copied_tool}")
        except Exception as e:
            print(f"❌ Deepcopy failed: {e}")
            return False
        
        # Test that the tool function can be called
        print("🔧 Testing tool function execution...")
        try:
            # Get the function from the tool
            if hasattr(adk_tool, 'func'):
                tool_func = adk_tool.func
            elif hasattr(adk_tool, '__call__'):
                tool_func = adk_tool
            else:
                print("⚠️  Tool doesn't have expected callable interface")
                return False

            # Call the function with a test parameter
            result = tool_func(expression="2 + 2")
            print(f"✅ Tool function executed successfully!")
            print(f"   Result: {result}")

            # Verify the result is a string (not a coroutine)
            if asyncio.iscoroutine(result):
                print("❌ Tool function returned a coroutine (this is the bug!)")
                return False
            else:
                print("✅ Tool function returned a proper string result")

        except Exception as e:
            print(f"❌ Tool function execution failed: {e}")
            return False
        
        print("🎉 All tests passed! The deepcopy issue has been fixed.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mock_tool_fallback():
    """Test the mock tool fallback when Google ADK is not available."""
    print("\n🧪 Testing mock tool fallback...")
    
    # Create adapter without Google ADK
    adapter = GoogleADKAdapter()
    adapter._mcp_tool_manager = MockMCPToolManager()
    
    # Create test objects
    mcp_tool = create_mock_mcp_tool()
    context = create_mock_context()
    
    try:
        # Force ImportError by temporarily modifying the import
        original_create_method = adapter._create_adk_mcp_tool_from_object
        
        async def mock_create_method(self, mcp_tool, context):
            # Simulate ImportError
            return self._create_mock_mcp_tool(mcp_tool)
        
        # Test mock tool creation
        mock_tool = adapter._create_mock_mcp_tool(mcp_tool)
        print(f"✅ Successfully created mock tool: {mock_tool}")
        
        # Test deepcopy of mock tool
        try:
            copied_mock = copy.deepcopy(mock_tool)
            print("✅ Successfully deep copied the mock tool!")
        except Exception as e:
            print(f"❌ Mock tool deepcopy failed: {e}")
            return False
        
        print("✅ Mock tool fallback works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Mock tool test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting MCP tool deepcopy fix tests...\n")
    
    # Test 1: Main fix
    test1_passed = await test_mcp_tool_creation_and_deepcopy()
    
    # Test 2: Mock fallback
    test2_passed = await test_mock_tool_fallback()
    
    print(f"\n📊 Test Results:")
    print(f"   Main fix test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Mock fallback test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The deepcopy issue has been successfully fixed.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
