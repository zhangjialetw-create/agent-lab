# Python 虚拟环境

虚拟环境为一个项目提供独立的 Python 包安装位置。Windows 项目里的解释器通常位于 .venv\Scripts\python.exe。

创建环境使用 python -m venv .venv。运行项目时明确使用 .\.venv\Scripts\python.exe，可以避免把包装到另一个 Python 里。复制项目后应重新创建虚拟环境。
