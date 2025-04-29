import asyncio
from contextlib import AsyncExitStack
from typing import List, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client

load_dotenv()  # load environment variables from .env


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
        print("\nConnected to server with tools:", [tool.name for tool in self.tools])

    def get_all_tools(self):
        return self.tools

    async def close_connection(self):
        await self.exit_stack.aclose()
        print("\nConnection closed.")
