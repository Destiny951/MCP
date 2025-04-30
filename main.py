import asyncio
import os

from dotenv import load_dotenv

from core.agent import Agent
from core.client import MCPClient
from core.utils.util import log_title

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
REQUEST_URL = f"{OLLAMA_HOST}:{OLLAMA_PORT}{OLLAMA_API_URL}"


async def main():

    currentDir = os.getcwd()
    url = "https://news.ycombinator.com/"
    outPath = os.path.join(currentDir, "output")
    prompt = f"请从{url}获取新闻，并保存到{outPath}/antonette.md,输出一个漂亮md文件"

    fetchMCP = MCPClient("mcp-server-fetch", "uvx", ["mcp-server-fetch"])
    fileMCP = MCPClient(
        "mcp-server-file",
        "npx",
        [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            currentDir,
        ],
    )

    agent = Agent(
        model=CHAT_MODEL,
        api_url=REQUEST_URL,
        clients=[fileMCP],
        sys_prompt="除非用户指定，否则默认回复中文",
        enable_memory=True,
    )

    await agent.init()
    await agent.invoke(prompt)
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
