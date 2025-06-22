#!/usr/bin/env python3
"""
Direct test of MCP server without our framework.
This will help isolate whether the issue is in our framework or the MCP server itself.
"""

import asyncio
import json
import subprocess
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_mcp_server_direct():
    """Test MCP server directly using subprocess communication."""
    
    # Start MCP server process
    logger.info("Starting MCP server process...")
    proc = subprocess.Popen(
        ['python3', 'simple_mcp_server.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0  # Unbuffered
    )
    
    try:
        # Test 1: Initialize
        logger.info("Test 1: Initialize")
        init_request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        proc.stdin.write(json.dumps(init_request) + '\n')
        proc.stdin.flush()
        
        # Read response
        response_line = proc.stdout.readline()
        logger.info(f"Initialize response: {response_line.strip()}")
        
        # Test 2: List tools
        logger.info("Test 2: List tools")
        list_request = {
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tools/list",
            "params": {}
        }
        
        proc.stdin.write(json.dumps(list_request) + '\n')
        proc.stdin.flush()
        
        # Read response
        response_line = proc.stdout.readline()
        logger.info(f"List tools response: {response_line.strip()}")
        
        # Test 3: Call tool
        logger.info("Test 3: Call tool")
        call_request = {
            "jsonrpc": "2.0",
            "id": "3",
            "method": "tools/call",
            "params": {
                "name": "calculate",
                "arguments": {"expression": "2 + 3 * 4"}
            }
        }
        
        proc.stdin.write(json.dumps(call_request) + '\n')
        proc.stdin.flush()
        
        # Read response with timeout
        logger.info("Waiting for tool call response...")
        
        # Use asyncio to add timeout
        async def read_with_timeout():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, proc.stdout.readline)
        
        try:
            response_line = await asyncio.wait_for(read_with_timeout(), timeout=5.0)
            logger.info(f"Tool call response: {response_line.strip()}")
            
            # Parse and validate response
            response = json.loads(response_line.strip())
            if "result" in response:
                content = response["result"].get("content", [])
                for item in content:
                    if item.get("type") == "text":
                        logger.info(f"Tool result: {item.get('text')}")
            else:
                logger.error(f"Tool call failed: {response}")
                
        except asyncio.TimeoutError:
            logger.error("Tool call timed out!")
            
        # Test 4: Send initialized notification
        logger.info("Test 4: Send initialized notification")
        init_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        
        proc.stdin.write(json.dumps(init_notification) + '\n')
        proc.stdin.flush()
        
        logger.info("All tests completed")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        
    finally:
        # Clean up
        logger.info("Cleaning up...")
        proc.stdin.close()
        
        # Wait for process to finish or kill it
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            logger.warning("Process didn't exit, killing it")
            proc.kill()
            proc.wait()
        
        # Read any remaining stderr
        stderr_output = proc.stderr.read()
        if stderr_output:
            logger.info(f"Server stderr: {stderr_output}")


if __name__ == "__main__":
    asyncio.run(test_mcp_server_direct())
