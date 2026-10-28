# 假模型用于观察流程，不含大语言模型。
actions = [
    {"action":"add", "a":25, "b":35},
    {"action":"finish"}
]
history = []
for step, decision in enumerate(actions, start=1):
    print("第", step, "步：", decision["action"])
    if decision["action"] == "finish":
        print("结束，历史结果：", history)
        break
    if decision["action"] == "add":
        result = decision["a"] + decision["b"]
        history.append(result)
        print("工具返回：", result)