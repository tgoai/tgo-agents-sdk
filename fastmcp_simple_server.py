#!/usr/bin/env python3
"""
Simple MCP Server using FastMCP for testing purposes.

This server provides basic mathematical calculation tools via the Model Context Protocol.
It's designed to be lightweight and easy to use for testing MCP client implementations.
"""

import logging
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastMCP server
server = FastMCP(name="simple-mcp-server")


@server.tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., '2 + 3 * 4')
        
    Returns:
        The result of the calculation
    """
    try:
        # Simple evaluation (in production, use a safer math parser)
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        logger.error(f"Calculation error: {e}")
        return f"Calculation error: {str(e)}"


@server.tool
def echo(message: str) -> str:
    """
    Echo back the input message.
    
    Args:
        message: Message to echo back
        
    Returns:
        The echoed message
    """
    return f"Echo: {message}"


@server.tool
def add_numbers(a: float, b: float) -> str:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of the two numbers
    """
    result = a + b
    return f"{a} + {b} = {result}"


@server.tool
def multiply_numbers(a: float, b: float) -> str:
    """
    Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The product of the two numbers
    """
    result = a * b
    return f"{a} × {b} = {result}"


@server.tool
def get_server_info() -> str:
    """
    Get information about this MCP server.
    
    Returns:
        Server information
    """
    return "FastMCP Simple Server v1.0.0 - Provides basic math and utility tools"


if __name__ == "__main__":
    logger.info("Starting FastMCP Simple Server")
    server.run()
