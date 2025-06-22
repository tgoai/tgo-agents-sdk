#!/usr/bin/env python3
"""
Test MCP in a threading environment similar to debug_example.py
This will help isolate whether the issue is with threading or Google ADK.
"""

import asyncio
import logging
import threading
import concurrent.futures
from tgo.agents.tools.mcp_tool_manager import MCPToolManager
from tgo.agents.core.models import MCPServerConfig, ExecutionContext

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def setup_mcp():
    """Set up MCP tool manager and connection."""
    mcp_manager = MCPToolManager()
    await mcp_manager.initialize()
    
    config = MCPServerConfig(
        server_id='test_server',
        name='Test Server',
        transport_type='stdio',
        command='python3',
        args=['simple_mcp_server.py'],
        trusted=True
    )
    
    mcp_manager.register_server(config)
    await mcp_manager.connect_to_server('test_server')
    
    return mcp_manager


def sync_mcp_call(mcp_manager, main_loop):
    """Simulate the sync function call from Google ADK."""
    logger.info("🌐 Sync MCP call started in worker thread")
    
    context = ExecutionContext(
        task_id='test_task',
        agent_id='test_agent',
        session_id='test_session',
        user_id='test_user'
    )
    
    try:
        # Method 1: Use run_coroutine_threadsafe (like our current approach)
        logger.debug("Using run_coroutine_threadsafe...")
        future = asyncio.run_coroutine_threadsafe(
            mcp_manager.call_tool(
                tool_name='calculate',
                arguments={'expression': '2 + 3 * 4'},
                context=context,
                user_approved=True
            ),
            main_loop
        )
        
        result = future.result(timeout=15)
        logger.info(f"✅ Method 1 success: {result.success}")
        if result.success:
            logger.info(f"Result content: {result.content}")
        else:
            logger.error(f"Result error: {result.error_message}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Method 1 failed: {e}")
        
        # Method 2: Direct connector call in new event loop
        try:
            logger.debug("Trying direct connector call...")
            connector = mcp_manager._connectors['test_server']
            
            def run_direct_call():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        connector.call_tool('calculate', {'expression': '2 + 3 * 4'})
                    )
                finally:
                    new_loop.close()
            
            direct_result = run_direct_call()
            logger.info(f"✅ Method 2 success: {direct_result}")
            return direct_result
            
        except Exception as e2:
            logger.error(f"❌ Method 2 failed: {e2}")
            return None


async def main():
    """Main test function."""
    logger.info("🚀 Starting MCP threading test")
    
    # Set up MCP in main thread
    mcp_manager = await setup_mcp()
    main_loop = asyncio.get_running_loop()
    
    logger.info("📋 Testing direct call in main thread...")
    
    # Test 1: Direct call in main thread (should work)
    context = ExecutionContext(
        task_id='test_task',
        agent_id='test_agent',
        session_id='test_session',
        user_id='test_user'
    )
    
    try:
        result = await mcp_manager.call_tool(
            tool_name='calculate',
            arguments={'expression': '2 + 3 * 4'},
            context=context,
            user_approved=True
        )
        logger.info(f"✅ Main thread call: {result.success}")
        if result.success:
            logger.info(f"Main thread result: {result.content}")
    except Exception as e:
        logger.error(f"❌ Main thread call failed: {e}")
    
    logger.info("🧵 Testing call from worker thread...")
    
    # Test 2: Call from worker thread (simulating Google ADK)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(sync_mcp_call, mcp_manager, main_loop)
        try:
            thread_result = future.result(timeout=30)
            logger.info(f"Worker thread completed: {thread_result}")
        except Exception as e:
            logger.error(f"Worker thread failed: {e}")
    
    # Clean up
    await mcp_manager.shutdown()
    logger.info("🏁 Test completed")


if __name__ == "__main__":
    asyncio.run(main())
