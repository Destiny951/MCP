import json
import os
from typing import List, NamedTuple, Optional, Union

import aiohttp
from mcp import Tool

from .utils.util import log_title


class ToolCall:

    def __init__(self):
        self.function_calls = []

    def add_tool_call(self, tool_calls: Union[dict, List[dict]]):
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            description = func.get("description")
            name = func.get("name")
            args_str = func.get("parameters", {})  # 获取 parameters

            # 如果 parameters 为空，尝试获取 arguments
            if args_str == {}:
                args_str = func.get("arguments", {})

            self.function_calls.append(
                {
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": args_str,
                    }
                }
            )

    def get_all_tools(self):
        return self.function_calls

    def get_tools_num(self):
        return len(self.function_calls)

    def clear(self):
        self.function_calls = []

    def to_json(self) -> str:

        return json.dumps(self.function_calls)


class LLMResponseData(NamedTuple):
    content: str
    tool_call: ToolCall


class LLM:

    def __init__(
        self,
        api_url: str,
        model: str,
        sys_prompt: Optional[str] = None,
        context: Optional[str] = None,
        tools: Optional[ToolCall] = None,
    ):
        self.url = api_url
        self.model = model
        self.sys_prompt = sys_prompt
        self.context = context
        self.tools = tools
        self.messages = []

        if sys_prompt:
            self.messages.append({"role": "system", "content": self.sys_prompt})
        if context:
            self.add_user_message(context)

    def add_user_message(self, content: str):
        """添加用户消息到上下文"""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """添加助手消息（可用于模拟前一次回复）"""
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_message(self, content: str):
        self.messages.append({"role": "tool", "content": content})

    async def chat(self, prompt: Optional[str] = None):
        if prompt:
            self.add_user_message(prompt)

        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "tools": self.tools.get_all_tools() if self.tools else [],
        }

        print("模型: ", end="", flush=True)
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=payload) as response:
                if response.status != 200:
                    log_title(f"请求失败: {response.status}")
                    return

                full_response = ""
                tool_call_obj = ToolCall()

                async for raw_line in response.content:
                    try:
                        data = json.loads(raw_line.decode("utf-8"))
                        content = data.get("message", {}).get("content")
                        print(content, end="", flush=True)

                        full_response += content

                        # 工具调用处理
                        tool_calls = data.get("message", {}).get("tool_calls", [])
                        if tool_calls:
                            tool_call_obj.add_tool_call(tool_calls)

                        # 判断是否结束
                        if data.get("done"):
                            break

                    except Exception as e:
                        print(f"解析错误: {e}")

                return LLMResponseData(full_response, tool_call_obj)

    def get_all_tools(self):
        return [{"type": "function", "function": tool} for tool in self.tools]

    def clear_messages(self):
        self.messages.clear()
