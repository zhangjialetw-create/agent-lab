import math
import re
from collections import Counter
from paths import NOTES


def tokens(text):
    #相邻两个字符组成一个片段，这便于观察的中文词法检索。
    text = re.sub(r"\s+", "", text.lower())
    if len(text) < 2:
        return [text] if text else []
    return [text[i:i+2] for i in range(len(text) - 1)]


def similarity(left, right):
    a, b = Counter(tokens(left)), Counter(tokens(right))
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    length = math.sqrt(sum(x*x for x in a.values()) *
                       sum(x*x for x in b.values()))
    return dot / length if length else 0.0


def search(query, folder=NOTES):
    hits = []
    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8-sig")
        # 每块最多 240 个字符，相邻块重叠 40 个字符。
        for start in range(0, len(text), 200):
            chunk = text[start:start + 240]
            score = similarity(query, chunk)
            if score > 0:
                hits.append({
                    "source": f"{path.name}#{start}",
                    "text": chunk,
                    "score": round(score, 4),
                })
    hits.sort(key=lambda item: (-item["score"], item["source"]))
    return hits[:2]


if __name__ == "__main__":
    import json
    print(json.dumps(search("如何创建虚拟环境"), ensure_ascii=False, indent=2))