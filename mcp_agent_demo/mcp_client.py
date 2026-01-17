from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys
from pathlib import Path



@dataclass
class McpTool:
    name: str
    description: str
    input_schema: Dict[str, Any]


class McpConnection:
    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None

    async def connect_stdio(self, server_script_path: str) -> None:
        server_params = StdioServerParameters(
            command=sys.executable,                     
            args=[str(Path(server_script_path).resolve())],
            env=None,
        )
        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        self.session = await self._exit_stack.enter_async_context(ClientSession(stdio, write))
        await self.session.initialize()

    async def list_tools(self) -> List[McpTool]:
        assert self.session is not None
        resp = await self.session.list_tools()
        return [
            McpTool(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema or {},
            )
            for t in resp.tools
        ]

    async def list_prompts(self) -> List[Dict[str, Any]]:
        assert self.session is not None
        resp = await self.session.list_prompts()
        return [{"name": p.name, "description": p.description or ""} for p in resp.prompts]

    async def list_resources(self) -> List[Dict[str, Any]]:
        assert self.session is not None
        resp = await self.session.list_resources()
        return [{"uri": r.uri, "name": r.name or "", "mimeType": r.mimeType or ""} for r in resp.resources]

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        assert self.session is not None
        result = await self.session.call_tool(tool_name, tool_args)
        # Many MCP tools return "content" blocks;
        return "\n".join(block.text for block in result.content if getattr(block, "text", None))

    async def close(self) -> None:
        await self._exit_stack.aclose()
