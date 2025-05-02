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
                "parameters": tool.inputSchema,
            }
        }

        # 将转换后的数据添加到 ToolCall
        tool_calls.add_tool_call(function_info)


class Agent:
    def __init__(
        self,
        model: str,
        api_url: str,
        clients: Optional[List[MCPClient]] = None,
        sys_prompt: Optional[str] = None,
        context: Optional[str] = None,
        enable_memory: bool = True,
    ):
        self.model = model
        self.api_url = api_url
        self.clients = clients
        self.sys_prompt = sys_prompt
        self.context = context
        self.enable_memory = enable_memory
        self.llm = None
        self.tool_calls = ToolCall()

    async def init(self):
        log_title("初始化智能体")
        try:
            if self.clients is not None:
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
            self.llm = LLM(
                self.api_url, self.model, self.sys_prompt, self.context, self.tool_calls
            )

        except Exception as e:
            # 捕获并记录异常
            print(f"初始化失败: {e}")
            raise  # 重新抛出异常以便调用者处理

    async def close(self):
        await asyncio.gather(*[client.close_connection() for client in self.clients])
        self.tool_calls.clear()

    def clear_memory(self):
        self.llm.clear_messages()
        print("Memory cleared.")

    async def invoke(self, prompt: str):
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        response = await self.llm.chat(prompt)
        while True:
            for tool_call in response.tool_call.get_all_tools():
                func = tool_call.get("function")
                func_name = func.get("name")
                func_args = func.get("parameters")

                # 查找匹配的工具和对应的 client
                match = next(
                    (
                        (client, tool)
                        for client in self.clients
                        for tool in client.get_all_tools()
                        if tool.name == func_name
                    ),
                    None,
                )

                if match:
                    client, tool = match
                    print(f"调用工具: {tool.name}，参数: {func_args}")
                    result = await client.call_tool(tool.name, func_args)
                    tool_message = (
                        f"工具名称: {tool.name}\n状态: 成功\n结果: {result.content}"
                    )
                else:
                    tool_message = (
                        f"工具名称: {func_name}\n状态: 错误\n结果: 工具未找到"
                    )

                self.llm.add_tool_message(tool_message)

                response = await self.llm.chat()
                continue
            break

        if self.enable_memory:
            self.llm.add_assistant_message(response.content)

        return response.content
