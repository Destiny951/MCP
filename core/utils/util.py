from rich import print


def log_title(message: str):
    total_length = 80
    message_length = len(message)
    padding = max(0, total_length - message_length - 4)  # -4 是为了留出空格和符号
    half_padding = padding // 2
    padded_message = f"{'=' * half_padding} {message} {'=' * half_padding}"

    # 如果总长度不够，补一个 "=" 右边
    if len(padded_message) < total_length:
        padded_message += "="

    print(f"[bold cyan]{padded_message}[/bold cyan]")
