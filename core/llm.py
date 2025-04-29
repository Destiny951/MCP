import json
import os
from typing import List, Optional, Union

import aiohttp
from dotenv import load_dotenv
from mcp import Tool
from utils.util import log_title

# 加载.env文件
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")


class ToolCall:
    """
    封装工具调用的类。

    该类用于表示工具调用的结构，包含函数名称和参数。

    属性:
    - function_calls (list): 包含多个函数调用的列表。

    方法:
    - add_function(name: str, arguments: dict): 添加一个新的函数调用。
    - to_json(): 将对象转换为 JSON 格式。
    """

    def __init__(self):
        self.function_calls = []

    def add_tool_call(self, tool_calls: Union[dict, List[dict]]):
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            name = func.get("name")
            args_str = func.get("arguments", "{}")

            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = {}

            self.function_calls.append(
                {"function": {"name": name, "arguments": arguments}}
            )

    def to_json(self) -> str:

        return json.dumps(self.function_calls)


class LLM:
    """
    与大型语言模型(LLM)交互的类。

    该类初始化了与LLM通信所需的基本参数和消息格式,提供了灵活性以适应不同的模型配置和需求。

    参数:
    - model (str): 使用的模型名称。
    - sys_prompt (Optional[str]): 系统提示，用于引导模型的行为和响应风格。
    - tools (Optional[List[Tool]]): 模型可以使用的工具列表，扩展了模型的能力。
    - context (Optional[str]): 与模型交互的上下文信息，帮助模型更好地理解对话或任务背景。
    """

    def __init__(
        self,
        model: str,
        sys_prompt: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
    ):
        self.url = f"{OLLAMA_HOST}:{OLLAMA_PORT}{OLLAMA_API_URL}"
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
        """
        与模型进行异步聊天。

        该方法接受一个可选的prompt参数,代表用户的输入。如果提供了prompt,
        则将其添加到消息历史中，并使用这些信息来生成对模型的请求。
        响应以流式处理，以便逐步显示模型的回复。

        参数:
        - prompt (Optional[str]): 用户的输入，如果提供，则基于此输入生成请求。

        注意:
        - 该方法为异步方法，需要在异步环境中使用。
        - 响应以流式处理，逐步输出模型的回复。
        """
        log_title("模型回复")
        if prompt:
            self.add_user_message(prompt)

        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "tools": self.tools,
        }

        try:
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
                            if data.get("done"):
                                break
                            msg = data.get("message", {})
                            content = msg.get("content", "")
                            tool_calls = msg.get("tool_calls", [])

                            print(content, end="", flush=True)

                            full_response += content
                            tool_call_obj.add_tool_call(tool_calls)
                        except Exception as e:
                            log_title(f"解析错误: {e}")

                    self.add_assistant_message(full_response)

                    if tool_call_obj:
                        self.add_tool_message(tool_call_obj)

        except Exception as e:
            log_title(f"请求或流式处理出错: {e}")

    def get_all_tools(self):
        return [{"type": "function", "function": tool} for tool in self.tools]
