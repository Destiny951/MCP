import asyncio
import json

import aiohttp


async def test_ollama():
    url = (
        "http://localhost:11434/api/chat"  # 假设本地 Ollama 服务运行在 localhost:11434
    )

    # 构建请求的 payload
    payload = {
        "model": "qwen3:1.7b",  # 模型名称
        "messages": [{"role": "user", "content": "Why is the sky blue?"}],  # 请求的内容
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
                            # 尝试解析每行 JSON 数据
                            result = json.loads(line.decode("utf-8"))
                            print(
                                "Response from Ollama API:",
                                json.dumps(result, indent=2),
                            )
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                else:
                    print(f"Request failed with status: {response.status}")
        except Exception as e:
            print(f"Error occurred: {e}")


# 运行测试
asyncio.run(test_ollama())
