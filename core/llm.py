import os
import requests
import json
from dotenv import load_dotenv
from utils.util import log_title
from mcp import Tool
from typing import Optional, List

# 加载.env文件
load_dotenv()

OLLAMA_HOST = os.getenv('OLLAMA_HOST')
OLLAMA_PORT = os.getenv('OLLAMA_PORT')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL')


class LLM:
    """
    与大型语言模型（LLM）交互的类。
    
    该类初始化了与LLM通信所需的基本参数和消息格式，提供了灵活性以适应不同的模型配置和需求。
    
    参数:
    - model (str): 使用的模型名称。
    - sys_prompt (Optional[str]): 系统提示，用于引导模型的行为和响应风格。
    - tools (Optional[List[Tool]]): 模型可以使用的工具列表，扩展了模型的能力。
    - context (Optional[str]): 与模型交互的上下文信息，帮助模型更好地理解对话或任务背景。
    """
    def __init__(self, model: str, sys_prompt: Optional[str] = None, tools: Optional[List[Tool]] = None, context: Optional[str] = None):
        self.url = f"{OLLAMA_HOST}:{OLLAMA_PORT}{OLLAMA_API_URL}"
        self.model = model
        self.sys_prompt = sys_prompt
        self.tools = tools
        self.context = context
        self.messages = []

        # 如果有上下文信息，加入用户消息
        if context:
            self.messages.append({
                "role": "user",
                "content": context
            })

    async def chat(self, prompt:Optional[str]=None):
        """
        与模型进行异步聊天。
    
        该方法接受一个可选的prompt参数，代表用户的输入。如果提供了prompt，
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
            self.messages.append({
                "role": "user",
                "content": prompt
            })
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "system": self.sys_prompt,   
        }
    
        try:
            response = requests.post(self.url, json=payload, stream=True, timeout=30)
            response.raise_for_status()
    
            print("Model: ", end="", flush=True)
    
            full_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        content = data.get("response", "")
                        print(content, end="", flush=True)
                        full_response += content
                    except Exception as e:
                        log_title(f"解析错误: {e}")

    
            print()

        except Exception as e:
            log_title(f"请求或流式处理出错: {e}")

