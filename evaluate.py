import json
import argparse
from paths import ROOT
from harness import run
import router


def evaluate_router():
    rows = router.read_rows("router_test.jsonl")
    model = router.load()
    baseline = {**model, "weights": [0.0] * len(model["weights"])}
    for name, candidate in [("before", baseline), ("after", model)]:
        correct = 0
        for row in rows:
            prediction, probability = router.predict(candidate, row["question"])
            correct += prediction == row["tool"]
            print(name, row["question"], prediction, round(probability, 3))
        print(name, f"路由准确率：{correct}/{len(rows)}")


def evaluate_harness(backend):
    rows = [
        {"question": "统计学习总分钟", "contains": "75", "source": "sessions.json"},
        {"question": "如何创建虚拟环境", "contains": "venv", "source": "python-env.md#0"},
        {"question": "Git 如何保存版本", "contains": "commit", "source": "git.md#0"},
        {"question": "量子海豚XYZ", "contains": "未找到", "source": None},
    ]
    results = []
    for row in rows:
        result = run(row["question"], backend)
        passed = result["status"] == "ok" and row["contains"] in result["answer"]
        if row["source"]:
            passed = passed and row["source"] in result["citations"]
        else:
            passed = passed and result["citations"] == []
        results.append({"question": row["question"], "passed": passed, **result})
    folder = ROOT / "artifacts"
    folder.mkdir(exist_ok=True)
    path = folder / f"eval-{backend}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"教学冒烟评测：{sum(r['passed'] for r in results)}/{len(results)}")
    print("报告：", path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", action="store_true")
    parser.add_argument("--backend", choices=["mock", "trained", "ollama"], default="mock")
    args = parser.parse_args()
    if args.router:
        evaluate_router()
    else:
        evaluate_harness(args.backend)