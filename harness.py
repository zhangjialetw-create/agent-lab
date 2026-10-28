import json
import time
import uuid
import argparse
from paths import RUNS
from model import decide
from skills import load_skill, load_memory
from tools import execute, TOOLS


def validate(decision):
    if not isinstance(decision, dict):
        raise ValueError("决策必须是字典")
    if set(decision) != {"action", "query", "answer", "citations"}:
        raise ValueError("决策字段不完整或有多余字段")
    if decision["action"] not in set(TOOLS) | {"finish"}:
        raise ValueError("动作不在允许列表中")
    for key in ["query", "answer"]:
        if not isinstance(decision[key], str) or len(decision[key]) > 3000:
            raise ValueError(f"{key} 必须是 3000 字以内的文字")
    if not isinstance(decision["citations"], list) or not all(
        isinstance(item, str) for item in decision["citations"]
    ):
        raise ValueError("citations 必须是字符串列表")


def allowed_sources(observations):
    sources = set()
    for observation in observations:
        data = observation["result"]
        if observation["tool"] == "study_stats":
            sources.add(data["source"])
        else:
            sources.update(row["source"] for row in data)
    return sources


def run(question, backend="mock", max_steps=3, log_dir=RUNS, decision_fn=None):
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 300:
        raise ValueError("问题须为 1 到 300 字")
    if not 1 <= max_steps <= 5:
        raise ValueError("max_steps 必须为 1 到 5")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{uuid.uuid4().hex}.jsonl"
    started = time.monotonic()
    observations = []
    chooser = decision_fn or decide
    trace = []
    status, answer, citations = "failed", "", []
    try:
        skill, memory = load_skill("study-coach"), load_memory()
        for step in range(1, max_steps + 1):
            decision, usage = chooser(question, observations, skill, memory, backend)
            validate(decision)
            trace.append({"step": step, "decision": decision, "usage": usage})
            if decision["action"] == "finish":
                if not observations:
                    raise ValueError("尚未获取工具结果")
                sources = allowed_sources(observations)
                if not set(decision["citations"]).issubset(sources):
                    raise ValueError("引用了工具未返回的来源")
                if sources and not decision["citations"]:
                    raise ValueError("已有资料时需要引用来源")
                if not decision["answer"].strip():
                    raise ValueError("最终答案不能为空")
                answer, citations = decision["answer"], decision["citations"]
                status = "ok"
                break
            result = execute(decision["action"], decision["query"])
            observation = {"tool": decision["action"], "result": result}
            observations.append(observation)
            trace.append({"observation": observation})
        else:
            raise ValueError("达到步数预算，停止运行")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        answer = f"运行失败：{error}"
    summary = {
        "question": question, "backend": backend, "status": status,
        "answer": answer, "citations": citations,
        "tool_calls": len(observations),
        "seconds": round(time.monotonic() - started, 3),
    }
    with log.open("w", encoding="utf-8") as file:
        for event in trace + [{"summary": summary}]:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
    return {**summary, "log": str(log)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地学习助手 Harness")
    parser.add_argument("question")
    parser.add_argument("--backend", choices=["mock", "trained", "ollama", "lora"], default="mock")
    args = parser.parse_args()
    result = run(args.question, args.backend)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "ok" else 1)
