import asyncio
import os
import subprocess
import sys

from dotenv import load_dotenv

from core.agent import Agent
from core.client import MCPClient
from core.utils.embedding_retriever import EmbeddingRetriever
from core.utils.util import log_title

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
REQUEST_URL = f"{OLLAMA_HOST}:{OLLAMA_PORT}{OLLAMA_API_URL}"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_REQUEST_URL = f"{OLLAMA_HOST}:{OLLAMA_PORT}{EMBEDDING_API_URL}"


async def embed_documents():
    # RAG
    log_title("编码文档")
    embedding_retriever = EmbeddingRetriever(EMBEDDING_MODEL, EMBEDDING_REQUEST_URL)
    knowledge_dir = os.path.join(os.getcwd(), "knowledge")
    files = os.listdir(knowledge_dir)

    for file in files:
        file_path = os.path.join(knowledge_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            await embedding_retriever.embed_document(content)

    return embedding_retriever


async def retrieve_context(embedding_retriever: EmbeddingRetriever, prompt: str):
    context_list = await embedding_retriever.retrieve(prompt, 3)
    context = "\n".join(context_list)

    log_title("CONTEXT")
    print(context)
    return context


def get_npx_path():
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["where", "npx"], check=True, capture_output=True, text=True
            )
            lines = result.stdout.strip().splitlines()
            if not lines:
                raise RuntimeError("npx not found in system PATH")
            npx_path = lines[-1]
        else:
            npx_path = "npx"
        return npx_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to locate npx executable via 'where' command") from e

    except IndexError as e:
        raise RuntimeError("Unexpected output from 'where npx' command") from e


async def main():
    # 初始化客户端
    currentDir = os.getcwd()
    fetchMCP = MCPClient("mcp-server-fetch", "uvx", ["mcp-server-fetch"])
    fileMCP = MCPClient(
        "mcp-server-file",
        get_npx_path(),
        [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            currentDir,
        ],
    )

    vector_database = await embed_documents()
    # 初始化智能体
    agent = Agent(
        model=CHAT_MODEL,
        api_url=REQUEST_URL,
        clients=[fileMCP, fetchMCP],
        sys_prompt="除非用户指定，否则默认回复中文",
        vector_database=vector_database,
        enable_memory=True,
    )

    await agent.init()

    # 对话循环
    log_title("✅ 智能体已启动，输入你的问题（输入 'exit' 或 'quit' 退出）:")

    while True:
        try:
            log_title("用户提问：")
            prompt = input(
                "🧠 你："
            ).strip()  # 请从https://baijiahao.baidu.com/s?id=1830983245152297044获取新闻，并整理结果保存到E:\mmproject\test\mcp-client/output/antonette.md,输出一个漂亮md文件,如果指定目录不存在，则创建目录
            if prompt.lower() in ("exit", "quit"):
                print("👋 正在关闭智能体...")
                break
            if prompt.lower() == "clear":
                agent.clear_memory()
                continue
            if not prompt:
                continue  # 忽略空输入
            log_title("模型回复")
            await agent.invoke(prompt)
        except KeyboardInterrupt:
            print("\n🛑 检测到中断，正在关闭...")
            break
        except Exception as e:
            print(f"❌ 出现错误: {e}")

    await agent.close()
    log_title("✅ 智能体已关闭")


if __name__ == "__main__":
    asyncio.run(main())
