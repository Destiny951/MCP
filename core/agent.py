import asyncio
import json
from typing import Any, List, Optional

from dotenv import load_dotenv

from mcp import Tool

from .client import MCPClient
from .llm import LLM, ToolCall
from .utils.util import log_title

load_dotenv()


def transform_tools_format(tool_calls: ToolCall, tools: List[Tool]):
    for tool in tools:
        # 创建目标数据格式
        function_info = {
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": json.dumps(tool.inputSchema),
            }
        }

        # 将转换后的数据添加到 ToolCall
        tool_calls.add_tool_call(function_info)


class Agent:
    def __init__(
        self,
        model: str,
        api_url: str,
        clients: List[MCPClient],
        sys_prompt: Optional[str] = None,
        enable_memory: bool = True,
    ):
        self.model = model
        self.api_url = api_url
        self.clients = clients
        self.sys_prompt = sys_prompt
        self.enable_memory = enable_memory
        self.llm = None
        self.tool_calls = ToolCall()

    async def init(self):
        log_title("初始化智能体")
        try:
            # 使用 asyncio.gather 并行执行所有客户端的连接操作，并设置 return_exceptions=True
            results = await asyncio.gather(
                *[client.connect_to_server() for client in self.clients],
                return_exceptions=True,
            )

            # 处理连接过程中可能出现的异常
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"客户端 {self.clients[i].name} 连接失败: {result}")
                    self.clients[i] = None  # 将失败的客户端标记为 None

            # 过滤掉失败的客户端
            self.clients = [client for client in self.clients if client is not None]

            # 获取所有工具
            for client in self.clients:
                transform_tools_format(self.tool_calls, client.get_all_tools())

            # 初始化 LLM
            self.llm = LLM(self.api_url, self.model, self.sys_prompt, self.tool_calls)

        except Exception as e:
            # 捕获并记录异常
            print(f"初始化失败: {e}")
            raise  # 重新抛出异常以便调用者处理

    async def close(self):
        await asyncio.gather(*[client.close_connection() for client in self.clients])
        self.tool_calls.clear()

    async def invoke(self, prompt: str):
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        response = await self.llm.chat(prompt)
        while True:
            if response.tool_call.get_tools_num() > 0:
                tool_calls = response.tool_call.get_all_tools()
                for tool_call in tool_calls:
                    for client in self.clients:
                        if tool_call.name == client.name:
                            log_title(f"调用工具: {tool_call.name}")
                            print(tool_call.arguments)
                            client.call_tool(tool_call.name, tool_call.arguments)
                            self.llm.add_tool_message(response.tool_call)
                            break
                response = await self.llm.chat()
                continue
            break
        if self.enable_memory:
            self.llm.add_assistant_message(response.content)

        return response.content
