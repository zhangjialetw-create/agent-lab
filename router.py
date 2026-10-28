"""用标准库训练二分类路由器：0=统计工具，1=知识检索。"""
import json
import math
from knowledge import tokens
from paths import DATA, ROOT


def read_rows(name):
    text = (DATA / name).read_text(encoding="utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def vector(text, vocab):
    present = set(tokens(text))
    return [1.0] + [float(word in present) for word in vocab]


def probability(weights, values):
    z = sum(w*x for w, x in zip(weights, values))
    z = max(-40.0, min(40.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def fit(rows, epochs=180):
    # 词表仅从训练集产生，不读取测试集。
    vocab = sorted({word for row in rows for word in tokens(row["question"])})
    weights = [0.0] * (len(vocab) + 1)
    examples = [(vector(row["question"], vocab),
                 int(row["tool"] == "search_notes")) for row in rows]
    losses = []
    for epoch in range(epochs):
        for values, target in examples:
            predicted = probability(weights, values)
            error = predicted - target
            for i in range(len(weights)):
                weights[i] -= 0.15 * error * values[i]
        loss = 0.0
        for values, target in examples:
            p = probability(weights, values)
            loss -= target*math.log(max(p, 1e-9))
            loss -= (1-target)*math.log(max(1-p, 1e-9))
        losses.append(loss / len(examples))
    return {"vocab": vocab, "weights": weights, "loss": [losses[0], losses[-1]]}


def predict(model, question):
    p = probability(model["weights"], vector(question, model["vocab"]))
    # 零权重时 p=0.5；固定选择统计，形成可复现的初始基线。
    tool = "search_notes" if p > 0.5 else "study_stats"
    return tool, p


def train():
    rows = read_rows("router_train.jsonl")
    model = fit(rows)
    folder = ROOT / "artifacts"
    folder.mkdir(exist_ok=True)
    path = folder / "router.json"
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print("训练损失：", [round(x, 4) for x in model["loss"]])
    print("保存：", path.name)
    return model


def load():
    return json.loads((ROOT / "artifacts" / "router.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    train()