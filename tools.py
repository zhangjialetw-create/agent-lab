import json
from paths import DATA


def study_stats(query=""):
    # 参数 query 留给统一接口使用；数据路径固定，模型不能指定任意文件。
    records = json.loads((DATA / "sessions.json").read_text(encoding="utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError("学习记录必须是列表")
    total = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("每条记录必须是字典")
        minutes = record.get("minutes")
        if type(minutes) is not int or minutes < 0:
            raise ValueError("minutes 必须是非负整数")
        total += minutes
    count = len(records)
    return {
        "count": count,
        "total": total,
        "average": total / count if count else 0.0,
        "source": "sessions.json",
    }


def search_notes(query):
    from knowledge import search
    if not isinstance(query, str) or not query.strip():
        raise ValueError("搜索词不能为空")
    return search(query)


TOOLS = {"study_stats": study_stats, "search_notes": search_notes}


def execute(name, query):
    if name not in TOOLS:
        raise ValueError(f"未允许的工具：{name}")
    return TOOLS[name](query)


if __name__ == "__main__":
    print(study_stats())