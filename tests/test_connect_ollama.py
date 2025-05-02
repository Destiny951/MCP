import asyncio
import json
import os

import aiohttp
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
REQUEST_URL = f"{OLLAMA_HOST}:{OLLAMA_PORT}{OLLAMA_API_URL}"


async def test_ollama():
    url = REQUEST_URL

    # 构建请求的 payload
    payload = {
        "model": CHAT_MODEL,  # 模型名称
        "messages": [
            {"role": "user", "content": "Why is the sky blue?"},
            {
                "role": "system",
                "content": "除非用户指定，否则默认回复中文,包括思考过程",
            },
        ],
        "tools": [],
        # "tools": [
        #     {
        #         "type": "function",
        #         "function": {
        #             "name": "get_current_weather",
        #             "description": "Get the current weather for a location",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "location": {
        #                         "type": "string",
        #                         "description": "The location to get the weather for, e.g. San Francisco, CA",
        #                     },
        #                     "format": {
        #                         "type": "string",
        #                         "description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
        #                         "enum": ["celsius", "fahrenheit"],
        #                     },
        #                 },
        #                 "required": ["location", "format"],
        #             },
        #         },
        #     }
        # ],
    }

    # 使用 aiohttp 发送 POST 请求
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload) as response:
                # 检查返回状态
                if response.status == 200:
                    # 处理 ndjson 格式的响应
                    # 逐行读取 ndjson 数据
                    async for line in response.content:
                        try:
                            result = json.loads(line.decode("utf-8"))
                            print(
                                result.get("message").get("content"), end="", flush=True
                            )
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                else:
                    print(f"Request failed with status: {response.status}")
        except Exception as e:
            print(f"Error occurred: {e}")


# 运行测试
asyncio.run(test_ollama())
