import asyncio
import os
import sys

from client import MCPClient
from dotenv import load_dotenv
from llm import LLM
from utils.util import log_title

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL")


async def main():
    # llm = LLM(CHAT_MODEL, sys_prompt="除非用户指定，否则默认回复中文")
    # while True:
    #     log_title("用户提问")
    #     prompt = input("用户：")
    #     if prompt.lower() == "exit":
    #         break
    #     await llm.chat(prompt)
    client = MCPClient("fetch", "uvx", ["mcp-server-fetch"])
    try:
        await client.connect_to_server()
        tools = client.get_all_tools()
        print(tools)
    finally:
        await client.close_connection()


if __name__ == "__main__":
    llm = LLM(CHAT_MODEL, sys_prompt="除非用户指定，否则默认回复中文")
    asyncio.run(main())
