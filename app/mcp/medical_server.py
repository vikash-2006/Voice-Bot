"""
Model Context Protocol (MCP) Server for Medical & Search Tools.
Implements the standard MCP JSON-RPC protocol over stdio / HTTP:
- tools/list: Returns the list of tools with JSON schemas
- tools/call: Executes a tool and returns formatted text/json content
"""
import sys
import json
import asyncio
from typing import Dict, Any, List
from app.services.medical_tools import MEDICAL_TOOLS_DEFINITIONS, execute_tool

SERVER_NAME = "medical-voice-mcp-server"
SERVER_VERSION = "1.0.0"

def get_mcp_tools_list() -> List[Dict[str, Any]]:
    """Converts tools into standard MCP tool schema."""
    mcp_tools = []
    for t in MEDICAL_TOOLS_DEFINITIONS:
        fn = t["function"]
        mcp_tools.append({
            "name": fn["name"],
            "description": fn["description"],
            "inputSchema": fn["parameters"]
        })
    return mcp_tools

def handle_mcp_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handles an incoming JSON-RPC MCP request."""
    method = request.get("method")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                }
            }
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": get_mcp_tools_list()
            }
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        result_data = execute_tool(tool_name, arguments)

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result_data, indent=2, ensure_ascii=False)
                    }
                ],
                "isError": "error" in result_data
            }
        }

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method '{method}' not found"
            }
        }

def run_stdio_server():
    """Runs standard MCP stdio loop for Cursor, Claude Desktop, or Antigravity integration."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_mcp_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(e)}
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    run_stdio_server()
