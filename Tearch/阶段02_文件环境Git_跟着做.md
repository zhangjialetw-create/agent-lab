# 阶段 02：文件、环境和 Git 都有明确用途

[返回总导航](00_从这里开始_跟着做.md) · 前置：阶段 01 的四个小练习。

## 第 5 节：把数据从代码搬到 JSON 文件

在 `agent-lab` 下新建 `data` 文件夹，在里面新建 `sessions.json`：

```json
[
  {"topic": "Python", "minutes": 25},
  {"topic": "终端", "minutes": 35},
  {"topic": "复习", "minutes": 15}
]
```

JSON 是保存结构化数据的文本格式。它和 Python 字典长得像，但键和字符串必须用双引号；不能写注释，最后一项后不能加多余逗号。

新建 `basics/04_files.py`：

```python
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
path = root / "data" / "sessions.json"
text = path.read_text(encoding="utf-8-sig")
records = json.loads(text)
print("文件：", path.name)
print("记录数：", len(records))
print("第一条主题：", records[0]["topic"])
```

```powershell
.\.venv\Scripts\python.exe basics/04_files.py
```

预期：

```text
文件： sessions.json
记录数： 3
第一条主题： Python
```

逐句解释：

- `import json`：加载 Python 自带的 JSON 工具。
- `from pathlib import Path`：从路径工具模块中拿出 Path。
- `__file__` 是当前代码文件位置；第一层 `.parent` 到 basics，第二层到 agent-lab。
- `root / "data" / "sessions.json"` 用路径对象拼接路径，这里的 `/` 不是数学除法。
- `read_text` 把文件读成一段字符串。
- `json.loads(text)` 把 JSON 字符串转换成 Python 列表和字典；`loads` 的 s 可理解为 string。
- `records[0]` 取第一条，列表下标从 0 开始；再用 `["topic"]` 取其主题。
- `encoding="utf-8-sig"` 兼容普通 UTF-8 和带 BOM 的 UTF-8 文本。

小练习：只修改数据文件，把第一条的 topic 改成“函数”，代码不动。重跑应看到新主题。这就是代码与数据分离。

路径检查：`Get-Location` 显示终端当前目录；`Get-ChildItem` 列出这里的文件。文件找不到时，先核对这两项，再核对 VS Code 左侧文件名有没有变成 `.json.txt`。

## 第 6 节：导入模块，理解包安装在哪儿

把阶段 01 的 `basics/03_functions.py` 中的函数定义复制到项目根目录 `stats_demo.py`。也可以复制整个文件，那个 `__main__` 判断会避免导入时额外打印演示。

在根目录新建 `import_demo.py`：

```python
from stats_demo import summarize

answer = summarize([{"topic": "函数", "minutes": 30}])
print(answer)
```

```powershell
.\.venv\Scripts\python.exe import_demo.py
```

应只显示这次的统计字典。`from stats_demo import summarize` 从同文件夹的 `stats_demo.py` 取得函数，文件名不写 `.py`。不要把模块命名成 `json.py`、`pathlib.py`，否则可能遮住同名标准库。

现在检查环境：

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -m pip --version
.\.venv\Scripts\python.exe -m pip list
```

第一条打印的路径应落在新项目 `agent-lab\.venv`。`-c` 表示执行后面的一小段 Python 代码。`pip` 管理第三方包；这里指定哪个 Python，就使用哪个 Python 的 pip。

本课程主线只用标准库，不需要安装第三方包。`json`、`pathlib`、`unittest` 无须 pip install。以前项目安装的 pytest 不会因此自动装进新环境，这正是隔离的效果。

小练习：用自己的话写出“Python 解释器”“.py 文件”“第三方包”“虚拟环境”各负责什么，保存到 `学习记录.md`。

## 第 7 节：Git 保存一次你看得懂的修改

在项目根目录创建 `.gitignore`：

```text
.venv/
.venv-lora/
__pycache__/
*.pyc
.env
runs/
artifacts/
adapter/
```

每行是本项目不需要 Git 跟踪的东西。环境可以重建；运行日志和训练产物先保留在本地。

```powershell
git init
git config user.name "Agent Learner"
git config user.email "learner@example.com"
git status
git add .gitignore basics data stats_demo.py import_demo.py
git diff --cached
git commit -m "完成基础和文件读取练习"
git log --oneline -3
```

名字和邮箱是教学占位值；可换成自己希望显示的提交身份。这两条没有 `--global`，只作用于当前练习仓库。

命令与用途：

| 命令 | 作用 |
|---|---|
| init | 在当前文件夹建立本地版本历史 |
| status | 看哪些文件新增、修改、暂存 |
| add 指定文件 | 把这些文件当前内容选入下一次提交 |
| diff --cached | 看即将提交的具体差异 |
| commit | 把暂存区保存成一个版本 |
| log | 查看已保存的版本 |

提交以后，把 `01_values.py` 的 extra 从 10 改成 15，保存。运行程序确认输出，再执行：

```powershell
git diff
git add basics/01_values.py
git commit -m "把额外学习时间改为15分钟"
git log --oneline -3
```

你应看到两次提交，具体哈希是自动产生的，不会与别人相同。

`nothing to commit` 通常表示没有新变化或忘了保存；`pathspec did not match` 先确认文件是否按前面步骤创建；`not a git repository` 检查是否在 agent-lab 内。

此节先掌握本地保存，不需要 GitHub、push、多人分支合并。

## 第 8 节：拆开“复制项目”和“重建环境”

这一节专门补你以前第七天遇到的坎。先回到父文件夹，只复制 Git 已提交的源文件：

```powershell
Set-Location ..
git clone .\agent-lab .\agent-lab-rebuild
Set-Location .\agent-lab-rebuild
Get-ChildItem -Force
```

成功时能看到代码和 `.git`，看不到 `.venv`。原因：`.gitignore` 排除了环境。

如果 `agent-lab-rebuild` 已存在，不覆盖它；换成新的名字，例如 `agent-lab-rebuild-2`，后续进入同一名字即可。

接下来为这份代码建环境：

```powershell
& 'D:\Society\1\Harness\Agent学习教程_10天\实验操作区\study-log\.venv\Scripts\python.exe' -m venv .venv
.\.venv\Scripts\python.exe basics/04_files.py
```

主线无第三方包，因此建完就能运行。以后有依赖时，才增加 `python -m pip install -r requirements.txt` 这一步。

最后回到主练习项目，后续都在那里继续：

```powershell
Set-Location -LiteralPath 'D:\Society\1\Harness\Agent学习教程_10天\实验操作区\agent-lab'
```

验收：能解释“clone 取回源代码和提交历史；venv 创建运行环境”。能在新环境运行程序，就已经完成了此前第七天的核心目标。

下一步：[阶段 03](阶段03_工具和最小循环_跟着做.md)。
