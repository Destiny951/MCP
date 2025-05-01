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
            args_str = func.get("parameters", "{}")

            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = {}

            self.function_calls.append(
                {
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": arguments,
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
        tools: Optional[ToolCall] = None,
    ):
        self.url = api_url
        self.model = model
        self.sys_prompt = sys_prompt
        self.tools = tools
        self.messages = []

        if sys_prompt:
            self.messages.append({"role": "system", "content": self.sys_prompt})

    def add_user_message(self, content: str):
        """添加用户消息到上下文"""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """添加助手消息（可用于模拟前一次回复）"""
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_message(self, tool_call: ToolCall):
        self.messages.append({"role": "tool", "content": tool_call.to_json()})

    async def chat(self, prompt: Optional[str] = None):
        log_title("模型回复")
        if prompt:
            self.add_user_message(prompt)

        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "tools": self.tools.get_all_tools() if self.tools else [],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url, json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    log_title(f"请求失败: {response.status}")
                    return

                print("模型: ", end="", flush=True)
                full_response = ""
                tool_call_obj = ToolCall()

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        msg = data.get("message", {})
                        content = msg.get("content", "")
                        tool_calls = msg.get("tool_calls", [])
                        print(content, end="", flush=True)

                        full_response += content
                        if tool_calls:
                            tool_call_obj.add_tool_call(tool_calls)

                        if data.get("done"):
                            break
                    except Exception as e:
                        print(f"解析错误: {e}")

                return LLMResponseData(full_response, tool_call_obj)

    def get_all_tools(self):
        return [{"type": "function", "function": tool} for tool in self.tools]
