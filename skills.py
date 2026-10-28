import json
from paths import ROOT, DATA

# 为了看清按需加载，初版目录清单由我们维护。
CATALOG = {
    "study-coach": "查询学习资料，或统计学习时长",
}


def load_skill(name):
    if name not in CATALOG:
        raise ValueError("Skill 不在本项目清单中")
    return (ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")


def load_memory():
    data = json.loads((DATA / "memory.json").read_text(encoding="utf-8-sig"))
    style = data.get("style", "")
    if not isinstance(style, str) or len(style) > 200:
        raise ValueError("style 必须是 200 字以内的文字")
    return style


if __name__ == "__main__":
    print("可用 Skill：", CATALOG)
    print(load_skill("study-coach"))