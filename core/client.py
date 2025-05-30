import asyncio
from contextlib import AsyncExitStack
from typing import List, Optional

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, name: str, command: str, arguments: list):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.write = None
        self.exit_stack = AsyncExitStack()

        self.name = name
        self.command = command
        self.arguments = arguments
        self.tools: List[Tool] = []

    async def connect_to_server(self):
        server_params = StdioServerParameters(
            command=self.command, args=self.arguments, env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.tools = response.tools
        print("连接到工具:", [tool.name for tool in self.tools])

    def get_all_tools(self):
        return self.tools

    def call_tool(self, name: str, arguments: dict):
        if self.session is None:
            raise RuntimeError("Session not initialized. Call connect_to_server first.")
        return self.session.call_tool(name, arguments)

    async def close_connection(self):
        await self.exit_stack.aclose()
        print("\nConnection closed.")
