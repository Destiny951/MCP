import os
import subprocess
import sys

# print(os.environ["PATH"])


def test_npx_execution():
    # 测试命令
    command = ["npx", "-v"]  # 通过 npx 查看版本号，确认 npx 是否可用

    try:
        # 执行命令并捕获输出
        result = subprocess.run(command, check=True, capture_output=True, text=True)

        # 打印命令执行的标准输出
        print("npx 输出:", result.stdout)

    except subprocess.CalledProcessError as e:
        # 捕获错误并输出
        print(f"错误发生: {e.stderr}", file=sys.stderr)

    except FileNotFoundError:
        # 如果找不到 npx 执行文件，输出错误
        print(
            "找不到 'npx' 可执行文件。请确保 Node.js 和 npx 已正确安装，并添加到 PATH 环境变量中。",
            file=sys.stderr,
        )


if __name__ == "__main__":
    test_npx_execution()
