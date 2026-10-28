from tools import execute

print(execute("study_stats","统计分钟数"))
try:
    execute("powershell","任意指令")
except ValueError as error:
    print("已拒绝：", error)
