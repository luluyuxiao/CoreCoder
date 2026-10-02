"""Tiny stdlib-only MCP stdio server for sandbox smoke tests."""

import json
import sys

TOOL = {
    "name": "echo",
    "description": "Echo one text value",
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}

for line in sys.stdin:
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "id" not in request:
        continue
    method = request.get("method")
    if method == "initialize":
        result = {
            "protocolVersion": request.get("params", {}).get("protocolVersion"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "minimal-sandbox-example", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [TOOL]}
    elif method == "tools/call":
        arguments = request.get("params", {}).get("arguments", {})
        result = {
            "content": [{"type": "text", "text": f"echo: {arguments.get('text', '')}"}],
            "isError": False,
        }
    else:
        response = {
            "jsonrpc": "2.0",
            "id": request["id"],
            "error": {"code": -32601, "message": "method not found"},
        }
        print(json.dumps(response), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
