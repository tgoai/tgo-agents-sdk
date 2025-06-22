# #!/usr/bin/env python3
# """
# Simple MCP Server for testing Stdio Transport.

# This is a minimal MCP server that implements the basic MCP protocol
# for testing purposes.
# """

# import json
# import sys
# import logging

# # Configure logging to stderr so it doesn't interfere with MCP communication
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     stream=sys.stderr
# )
# logger = logging.getLogger(__name__)


# def handle_initialize(request_id, params):
#     """Handle initialize request."""
#     logger.debug(f"Handling initialize request: {params}")
    
#     return {
#         "jsonrpc": "2.0",
#         "id": request_id,
#         "result": {
#             "protocolVersion": "2025-06-18",
#             "capabilities": {
#                 "tools": {}
#             },
#             "serverInfo": {
#                 "name": "simple-mcp-server",
#                 "version": "1.0.0"
#             }
#         }
#     }


# def handle_tools_list(request_id, params):
#     """Handle tools/list request."""
#     logger.debug(f"Handling tools/list request: {params}")
    
#     return {
#         "jsonrpc": "2.0",
#         "id": request_id,
#         "result": {
#             "tools": [
#                 {
#                     "name": "calculate",
#                     "description": "Perform mathematical calculations",
#                     "inputSchema": {
#                         "type": "object",
#                         "properties": {
#                             "expression": {
#                                 "type": "string",
#                                 "description": "Mathematical expression to evaluate"
#                             }
#                         },
#                         "required": ["expression"]
#                     }
#                 }
#             ]
#         }
#     }


# def handle_tools_call(request_id, params):
#     """Handle tools/call request."""
#     logger.debug(f"Handling tools/call request: {params}")
    
#     tool_name = params.get("name")
#     arguments = params.get("arguments", {})
    
#     if tool_name == "calculate":
#         expression = arguments.get("expression", "")
#         try:
#             # Safe evaluation of basic math expressions
#             import math
#             result = eval(expression, {"__builtins__": {}, "math": math})
            
#             return {
#                 "jsonrpc": "2.0",
#                 "id": request_id,
#                 "result": {
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": f"Result: {result}"
#                         }
#                     ]
#                 }
#             }
#         except Exception as e:
#             return {
#                 "jsonrpc": "2.0",
#                 "id": request_id,
#                 "result": {
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": f"Error: {str(e)}"
#                         }
#                     ],
#                     "isError": True
#                 }
#             }
#     else:
#         return {
#             "jsonrpc": "2.0",
#             "id": request_id,
#             "error": {
#                 "code": -32601,
#                 "message": f"Unknown tool: {tool_name}"
#             }
#         }


# def handle_request(request):
#     """Handle incoming MCP request."""
#     try:
#         request_id = request.get("id")
#         method = request.get("method")
#         params = request.get("params", {})
        
#         logger.debug(f"Received request: method={method}, id={request_id}")
        
#         if method == "initialize":
#             return handle_initialize(request_id, params)
#         elif method == "tools/list":
#             return handle_tools_list(request_id, params)
#         elif method == "tools/call":
#             return handle_tools_call(request_id, params)
#         elif method == "notifications/initialized":
#             # Notification - no response needed
#             logger.debug("Received initialized notification")
#             return None
#         else:
#             return {
#                 "jsonrpc": "2.0",
#                 "id": request_id,
#                 "error": {
#                     "code": -32601,
#                     "message": f"Method not found: {method}"
#                 }
#             }
            
#     except Exception as e:
#         logger.error(f"Error handling request: {e}")
#         return {
#             "jsonrpc": "2.0",
#             "id": request.get("id") if isinstance(request, dict) else None,
#             "error": {
#                 "code": -32603,
#                 "message": f"Internal error: {str(e)}"
#             }
#         }


# def main():
#     """Main MCP server loop."""
#     logger.info("Starting Simple MCP Server")

#     try:
#         while True:
#             # Read line from stdin
#             logger.debug("Waiting for input...")
#             line = sys.stdin.readline()
#             if not line:
#                 logger.info("EOF received, shutting down")
#                 break

#             line = line.strip()
#             if not line:
#                 logger.debug("Empty line received, continuing...")
#                 continue

#             logger.debug(f"Received line: {line}")

#             try:
#                 # Parse JSON-RPC request
#                 request = json.loads(line)
#                 logger.debug(f"Parsed request: {request}")

#                 # Handle request
#                 logger.debug("Handling request...")
#                 response = handle_request(request)
#                 logger.debug(f"Generated response: {response}")

#                 # Send response if needed
#                 if response is not None:
#                     response_json = json.dumps(response)
#                     logger.debug(f"Sending response: {response_json}")
#                     print(response_json)
#                     sys.stdout.flush()
#                     logger.debug("Response sent and flushed")
#                 else:
#                     logger.debug("No response to send (notification)")

#             except json.JSONDecodeError as e:
#                 logger.error(f"Invalid JSON received: {e}")
#                 error_response = {
#                     "jsonrpc": "2.0",
#                     "id": None,
#                     "error": {
#                         "code": -32700,
#                         "message": "Parse error"
#                     }
#                 }
#                 print(json.dumps(error_response))
#                 sys.stdout.flush()
#             except Exception as e:
#                 logger.error(f"Error processing request: {e}")
#                 logger.exception("Full exception details:")

#     except KeyboardInterrupt:
#         logger.info("Received interrupt, shutting down")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         logger.exception("Full exception details:")
#     finally:
#         logger.info("Simple MCP Server shutdown")


# if __name__ == "__main__":
#     main()
