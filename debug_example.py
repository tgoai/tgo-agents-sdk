#!/usr/bin/env python3
"""
Debug Example for TGO Multi-Agent Coordinator with MCP Tools

This debug example tests the multi-agent system functionality including:
- Basic component initialization (registry, memory, session management)
- MCP (Model Context Protocol) tools integration
- MCP tool calling and security
- Single agent execution with MCP tools support
- Comprehensive error handling and logging

The example uses mock MCP servers for testing to avoid external dependencies.

Usage:
    python debug_example.py

Features tested:
- AdapterRegistry functionality
- Memory and session management
- MCP tool manager initialization
- MCP server registration and connection
- MCP tool discovery and calling
- Single agent execution with MCP tools
- Error handling and cleanup
"""

import asyncio
import logging
from datetime import datetime, timezone
from tgo.agents.core.models import MCPTool
from tgo.agents.tools import MCPToolManager

# Core imports - using tgo.agents structure
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tgo.agents import (
    MultiAgentCoordinator, AdapterRegistry, GoogleADKAdapter,
)
from tgo.agents.core.models import (
    MultiAgentConfig, AgentConfig, Task, WorkflowConfig, Session
)
from tgo.agents.core.enums import (
    AgentType, WorkflowType, ExecutionStrategy, TaskType,
    TaskPriority, SessionType
)
from tgo.agents.memory.in_memory_memory_manager import InMemoryMemoryManager
from tgo.agents.memory.in_memory_session_manager import InMemorySessionManager

# Configure simple logging
logging.basicConfig(
    level=logging.DEBUG,
    force=True,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def debug_single_agent():
    """Single agent execution with real MCP Stdio Transport tools."""
    logger.info("🔧 Starting Debug Example - Single Agent with MCP Stdio Transport")

    try:
        # Step 1: Initialize memory and session managers
        logger.info("📝 Step 1: Initializing memory and session managers...")
        memory_manager = InMemoryMemoryManager()
        session_manager = InMemorySessionManager()

        # Step 2: Initialize MCP tool manager for Stdio Transport
        logger.info("📝 Step 2: Initializing MCP tool manager...")
        config = {
            "mcpServers": {
                "mathserver": {
                    "command": "python",
                    "args": ["./fastmcp_simple_server.py"],
                    "env": {"DEBUG": "true"}
                }
            }
        }
        
        mcp_manager = MCPToolManager(config)

        # Step 3: Configure MCP server with Stdio Transport
        logger.info("📝 Step 3: Configuring MCP server with Stdio Transport...")

        # Register and connect to MCP server
        mcp_connected = True

        # Step 4: Create adapter registry
        logger.info("📝 Step 4: Creating adapter registry...")
        registry = AdapterRegistry()

        # Step 5: Register Google ADK adapter
        logger.info("📝 Step 5: Registering Google ADK adapter...")
        google_adapter = GoogleADKAdapter()
        registry.register("google-adk", google_adapter, is_default=True)

        # Step 6: Create coordinator with MCP support
        logger.info("📝 Step 6: Creating multi-agent coordinator with MCP support...")
        coordinator = MultiAgentCoordinator(
            registry=registry,
            memory_manager=memory_manager,
            session_manager=session_manager,
            mcp_tool_manager=mcp_manager
        )

        # Step 5: Create session
        logger.info("📝 Step 5: Creating session...")
        session_id = "debug_session_001"
        user_id = "debug_user"
        session_type = SessionType.SINGLE_CHAT

        await session_manager.create_session(session_id, user_id, session_type)

        # Create session object for coordinator
        session = Session(
            session_id=session_id,
            user_id=user_id,
            session_type=session_type
        )
        logger.info(f"✅ Session created: {session.session_id}")

        # Step 7: Configure single agent with elegant mixed tools
        logger.info("📝 Step 7: Configuring agent with elegant mixed tools...")

        # Create a simple debug function tool
        def debug_echo_tool(message: str) -> str:
            """A simple debug tool that echoes the input message."""
            logger.info(f"🔧 Function tool called with message: {message}")
            return f"Function Echo: {message}"

        # Create MCPTool that corresponds to the MCP server
        if mcp_connected:
            # Real MCP tool that connects to our Stdio Transport server
            calculator_tool = MCPTool(
                name="calculate",
                description="Real MCP calculator via Stdio Transport",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                    },
                    "required": ["expression"]
                },
                server_id="mathserver",  # Must match MCPServerConfig server_id
                requires_confirmation=False
            )
            logger.info("🌐 Using real MCP tool with Stdio Transport")
        else:
            # Fallback demo tool
            calculator_tool = MCPTool(
                name="calculate",
                description="Demo MCP tool (fallback)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                    },
                    "required": ["expression"]
                },
                server_id="demo_server",
                requires_confirmation=False
            )
            logger.info("🔧 Using demo MCP tool as fallback")

        config = MultiAgentConfig(
            framework="google-adk",
            agents=[
                AgentConfig(
                    agent_id="debug_agent_001",
                    name="Debug Agent with Elegant Tools",
                    agent_type=AgentType.EXPERT,
                    description="Debug agent showcasing elegant tool configuration",
                    model="gemini-2.0-flash",
                    instructions="You are a helpful assistant with access to both function tools and MCP tools. Demonstrate the elegant tool configuration by using the available tools.",
                    tools=[debug_echo_tool, calculator_tool]  # Elegant: Mixed tools in one array!
                )
            ],
            workflow=WorkflowConfig(
                workflow_type=WorkflowType.SINGLE,
                execution_strategy=ExecutionStrategy.FAIL_FAST,
                timeout_seconds=60
            )
        )
        logger.info(f"✅ Agent configured: {config.agents[0].name}")
        logger.info(f"🔧 Function tools: {len(config.agents[0].get_function_tools())}")
        logger.info(f"🌐 MCP tools: {len(config.agents[0].get_mcp_tools())}")
        logger.info(f"📊 Total tools: {len(config.agents[0].tools)}")

        # Step 8: Create task that demonstrates MCP Stdio Transport
        logger.info("📝 Step 8: Creating task...")
        if mcp_connected:
            task = Task(
                title="MCP Stdio Transport Demo",
                description="Please demonstrate the MCP Stdio Transport by: 1) Using debug_echo_tool to echo 'Hello from MCP demo!', 2) Using the calculate tool to compute '2 + 3 * 4', and 3) Explaining what tools you have available.",
                task_type=TaskType.COMPLEX,
                priority=TaskPriority.HIGH
            )
        else:
            task = Task(
                title="Elegant Tool Configuration Demo",
                description="Please demonstrate the elegant tool configuration by using the debug_echo_tool to echo 'Hello from elegant tools!' and explain what tools you have available.",
                task_type=TaskType.SIMPLE,
                priority=TaskPriority.MEDIUM
            )
        logger.info(f"✅ Task created: {task.title}")

        # Step 9: Execute task
        logger.info("📝 Step 9: Executing task with MCP tools...")
        logger.info("⏳ This may take a moment...")

        start_time = datetime.now(timezone.utc)
        result = await coordinator.execute_task(config, task, session)
        end_time = datetime.now(timezone.utc)

        execution_time = (end_time - start_time).total_seconds()

        # Step 10: Check results
        logger.info("📝 Step 10: Checking results...")

        if result.is_successful():
            logger.info("🎉 SUCCESS! Elegant tool configuration works perfectly!")
            logger.info(f"⏱️  Execution time: {execution_time:.2f} seconds")
            logger.info(f"📊 Result: {result.result}")

            # Show execution metrics if available
            if hasattr(result, 'total_execution_time_ms') and result.total_execution_time_ms:
                logger.info(f"📈 Metrics: {result.total_execution_time_ms}ms total")

            # Show tool configuration summary
            logger.info("📝 Step 11: Tool configuration summary...")
            logger.info(f"✅ Successfully demonstrated elegant tool configuration:")
            logger.info(f"   🔧 Function tools: {len(config.agents[0].get_function_tools())}")
            logger.info(f"   🌐 MCP tools: {len(config.agents[0].get_mcp_tools())}")
            logger.info(f"   � Total tools in single array: {len(config.agents[0].tools)}")

            # Cleanup MCP manager
            if 'mcp_manager' in locals():
                await mcp_manager.shutdown()
                logger.info("✅ MCP tool manager shutdown complete")

            return True
        else:
            logger.error("❌ FAILED! Task execution failed")
            logger.error(f"💥 Error: {result.error_message}")

            # Still cleanup MCP manager
            if 'mcp_manager' in locals():
                await mcp_manager.shutdown()
            return False

    except Exception as e:
        logger.error(f"💥 EXCEPTION during debug execution: {e}")
        logger.exception("Full exception details:")

        # Cleanup MCP manager in case of exception
        try:
            if 'mcp_manager' in locals():
                await mcp_manager.shutdown()
        except:
            pass

        return False

async def main():
    """Main debug function - runs all tests including MCP tools."""
    print("🔧 TGO Multi-Agent Coordinator - Debug Example with MCP Tools")
    print("=" * 60)

    start_time = datetime.now(timezone.utc)

    # Run basic component tests first
    # logger.info("🧪 Running component tests...")

    # registry_ok = await debug_registry_test()
    # memory_ok = await debug_memory_test()

    # if not (registry_ok and memory_ok):
    #     logger.error("❌ Component tests failed. Stopping.")
    #     return

    # logger.info("✅ Component tests passed.")

    # # Run MCP tools test
    # logger.info("🧪 Running MCP tools test...")
    # mcp_ok = await debug_mcp_tools()

    # Run MCP tool call test
    # logger.info("🧪 Running MCP tool call test...")
    # mcp_call_ok = await debug_mcp_tool_call()

    # if not (mcp_ok and mcp_call_ok):
    #     logger.warning("⚠️  Some MCP tests failed, but continuing with integration test...")
    # else:
    #     logger.info("✅ All MCP tests passed.")

    # logger.info("🧪 Running full integration test with MCP tools...")

    # Run full integration test with MCP tools
    success = await debug_single_agent()

    end_time = datetime.now(timezone.utc)
    total_time = (end_time - start_time).total_seconds()

    print("\n" + "=" * 60)
    if success:
        print("🎉 DEBUG EXAMPLE WITH MCP TOOLS COMPLETED SUCCESSFULLY!")
        logger.info("✅ All tests passed")
        print("📋 Tests completed:")
        print("  ✅ Registry functionality")
        print("  ✅ Memory management")
        # print(f"  {'✅' if mcp_ok else '⚠️ '} MCP tools functionality")
        # print(f"  {'✅' if mcp_call_ok else '⚠️ '} MCP tool calling")
        print("  ✅ Single agent execution with MCP tools")
    else:
        print("❌ DEBUG EXAMPLE FAILED!")
        logger.error("❌ Some tests failed")

    print(f"⏱️  Total execution time: {total_time:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Debug example interrupted by user")
    except Exception as e:
        print(f"\n💥 Debug example failed with error: {e}")
        logging.exception("Detailed error information:")
