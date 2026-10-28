# 阶段 07：更新语言模型参数，再决定扩展什么

[返回总导航](00_从这里开始_跟着做.md) · 第 23—24 节为选做。前六阶段已构成能运行、能评测、能训练路由组件的完整主线。

## 第 23 节：做一次小型语言模型 LoRA 实验

### 先明确这次更新什么

阶段 06 更新的是一个二分类路由器。这里改为在一个 0.5B 语言模型上增加可训练的 LoRA 参数，让它更倾向按要求输出正确工具名。

LoRA 在部分层中添加小的可训练矩阵，基础模型参数保持冻结。训练后保存 adapter；推理时需要基础模型加 adapter。它不等于把知识库写进模型，也不等于重新训练整个大模型。参考 [PEFT 官方入门](https://huggingface.co/docs/peft/quicktour)。

本实验选 [Qwen2.5-0.5B-Instruct 官方模型](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，理由是体量较小、常见 Python 库可加载，适合观察训练过程。它不是最新性能排名的推荐。

你的 4GB 显卡优先留给推理。此实验显式在 CPU 上进行，batch=1、短输入、少步数、LoRA rank=4。它会慢，但便于绕开 CUDA、量化库和显存配置的多重难点。没有实测这台机器的 LoRA 吞吐量，先做 1 步测量再决定是否继续。

### 1. 创建独立实验环境

在 agent-lab 根目录的 PowerShell 中：

```powershell
.\.venv\Scripts\python.exe -m venv .venv-lora
.\.venv-lora\Scripts\python.exe -m pip install --upgrade pip
.\.venv-lora\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv-lora\Scripts\python.exe -m pip install "transformers>=4.45,<5" "peft>=0.13,<0.19" accelerate
.\.venv-lora\Scripts\python.exe -m pip check
```

这些步骤会联网下载库，占用额外磁盘。先关闭不需要的大程序，确认系统盘与项目盘均有余量；建议为整个实验预留约 8GB，实际下载与缓存用量随版本变化。不要为了本节去升级原有 MachineLearning 等 Conda 环境。

为什么用新的 `.venv-lora`：主线只需标准库，训练新增的依赖比较多，把它们隔开更容易排错。官方安装入口：[PyTorch 本地安装](https://pytorch.org/get-started/locally/)。

本处给出 4.x Transformers / PEFT 的兼容版本范围，不把未验证的新主版本直接混入。安装后记录本次真正使用的版本：

```powershell
.\.venv-lora\Scripts\python.exe -c "import torch,transformers,peft; print(torch.__version__,transformers.__version__,peft.__version__)"
.\.venv-lora\Scripts\python.exe -m pip freeze | Out-File -FilePath requirements-lora.txt -Encoding utf8
```

以后重建时仍先按官方 CPU 源安装 torch，再从 requirements-lora.txt 安装其余固定版本。`pip freeze` 只记录包版本，不记录全部下载源和硬件条件，所以要把本节命令一同保存。

### 2. 新建训练程序

在根目录创建 `lora_lab.py`：

```python
"""可选 CPU LoRA 实验。需要独立安装 torch、transformers、peft。"""
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from paths import ROOT
from router import read_rows

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = ROOT / "adapter"


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
    )
    model.to("cpu")
    model.config.use_cache = False
    return tokenizer, model


def prompt(tokenizer, question):
    messages = [
        {"role": "system", "content":
         "选择工具。统计学习记录用 study_stats，查询概念用 search_notes。"
         "只输出工具名，不解释。"},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def train(steps):
    torch.manual_seed(42)
    torch.set_num_threads(4)
    tokenizer, base = load_base()
    config = LoraConfig(
        task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base, config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.001,
    )
    rows = read_rows("router_train.jsonl")
    for step in range(steps):
        row = rows[step % len(rows)]
        prefix = tokenizer(
            prompt(tokenizer, row["question"]), add_special_tokens=False,
        )["input_ids"]
        target = tokenizer(
            row["tool"] + tokenizer.eos_token, add_special_tokens=False,
        )["input_ids"]
        if len(prefix) + len(target) > 256:
            raise ValueError("样本超过 256 token，请缩短问题")
        ids = torch.tensor([prefix + target], dtype=torch.long)
        # -100 表示不对提示部分算损失，只学习正确工具名。
        labels = torch.tensor([[-100]*len(prefix) + target], dtype=torch.long)
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"step={step+1}/{steps} loss={loss.item():.4f}", flush=True)
    model.save_pretrained(ADAPTER)
    tokenizer.save_pretrained(ADAPTER)
    print("已保存 LoRA adapter：", ADAPTER)


def generate_tool(tokenizer, model, question):
    encoded = tokenizer(prompt(tokenizer, question), return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        output = model.generate(
            **encoded, max_new_tokens=16, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()


def evaluate():
    torch.set_num_threads(4)
    tokenizer, model = load_base()
    rows = read_rows("router_test.jsonl")
    results = {}
    for phase in ["before", "after"]:
        if phase == "after":
            model = PeftModel.from_pretrained(model, ADAPTER)
        model.eval()
        answers = []
        for row in rows:
            output = generate_tool(tokenizer, model, row["question"])
            answers.append({**row, "prediction": output, "correct": output == row["tool"]})
        results[phase] = answers
        print(phase, sum(item["correct"] for item in answers), "/", len(answers))
    folder = ROOT / "artifacts"
    folder.mkdir(exist_ok=True)
    (folder / "lora-eval.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )


_loaded = None


def predict_tool(question):
    global _loaded
    if _loaded is None:
        torch.set_num_threads(4)
        tokenizer, base = load_base()
        model = PeftModel.from_pretrained(base, ADAPTER)
        model.eval()
        _loaded = tokenizer, model
    tool = generate_tool(*_loaded, question)
    if tool not in {"study_stats", "search_notes"}:
        raise ValueError(f"LoRA 模型未返回合法工具名：{tool}")
    return tool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    if args.mode == "train":
        if not 1 <= args.steps <= 1000:
            parser.error("steps 必须为 1 到 1000")
        train(args.steps)
    else:
        evaluate()
```

读代码可以拆成五个小任务：

1. **加载。** tokenizer 把文字转换成模型接受的 token 编号；model 加载预训练参数。from_pretrained 第一次会下载并缓存模型，之后使用缓存。
2. **加 adapter。** r=4 是低秩维度，target_modules 指定加在 q_proj 和 v_proj。它们是注意力模块中的投影层；现在先理解“只训练新增的小矩阵”。
3. **准备正确答案。** prompt 告诉模型可用工具；target 是带标签的工具名。labels 的提示部分设成 -100，使损失只计算答案部分。
4. **更新。** loss 衡量当前输出与正确答案的差距；backward 计算梯度；optimizer.step 更新参数；zero_grad 清除本轮梯度，准备下一次。
5. **保存与重载。** save_pretrained 保存 adapter；PeftModel.from_pretrained 在基础模型上加载 adapter。

### 3. 先做 1 步

```powershell
Measure-Command { .\.venv-lora\Scripts\python.exe lora_lab.py train --steps 1 | Out-Host }
```

预期：先显示可训练参数数量，再显示 step=1/1 与一个 loss，最后生成 adapter 文件夹。`Measure-Command` 计时，`Out-Host` 让训练输出同时显示在终端。首次计时包括下载和加载，不可当作单步训练耗时。loss 的数值不预先保证。

如果内存持续接近满、系统明显卡顿，可以 Ctrl+C 停止。没有产生最终 adapter 时不要继续 eval；当前主线项目仍可以运行。

1 步成功后再运行：

```powershell
.\.venv-lora\Scripts\python.exe lora_lab.py train --steps 12
```

注意：脚本每次从基础模型重新开始训练，覆盖同目录 adapter，并非从上一次训练接着走。想保留 1 步版本，用文件管理器把整个 adapter 文件夹复制成 adapter-step1 后再运行。12 步覆盖 12 个训练样本一遍，足够观察机制，通常不足以证明效果提升。

## 第 24 节：评估 adapter，并接回同一个 Harness

```powershell
.\.venv-lora\Scripts\python.exe lora_lab.py eval
```

程序先用基础模型预测，再加载 adapter 用同样的保留题预测，保存 `artifacts/lora-eval.json`。打开报告看具体输出。可能变好、持平或变差，都如实记录；训练损失下降不能代替保留题评测。

接回已写好的 Harness：

```powershell
.\.venv-lora\Scripts\python.exe harness.py "解释虚拟环境的用途" --backend lora
```

本模式先由语言模型选择工具，再用确定性的模板组织最终答案；这能单独评估“训练是否改善了工具选择”。LoRA 模型若没有返回合法工具名，框架会拒绝，不会悄悄换成规则路由并冒充训练成功。

这里选择的是 Hugging Face / PEFT 加载路径。之前的 Ollama 模型文件用于推理，不能直接当作这段训练代码的 Transformers 权重；本节也没有把 adapter 自动转换、导入 Ollama。

常见问题：

| 问题 | 定位 |
|---|---|
| No module named torch/peft | 检查运行的是 `.venv-lora` 中的 Python |
| 下载超时或无法连 Hugging Face | 检查网络与官方模型页可访问性；训练尚未开始时不能归咎于 GPU |
| 找不到 adapter | 先完成 train，检查最终保存提示 |
| 内存不足 | 关闭其他模型与大程序，先 1 步；仍不足则保留主线，不在当前机器扩大模型 |
| 分数没有提升 | 数据量、标签和训练步数都可能影响；先查看具体错题，另留测试集 |
| 完整句子而非工具名 | 格式遵循失败；保留原输出，补训练示例和开发测试 |

本节脚本已做语法核验；当前交付没有安装训练依赖、下载 HF 权重或实际跑 LoRA，因此没有给出虚构的训练耗时和分数。主线的小路由训练已实际完成。

## 第 25 节：现代 Agent 怎样逐步扩展

没有一个框架能单凭名字保证“先进”。可持续扩展的做法是：明确状态和工具接口，把上下文组织好，用证据和评测决定新增什么。下面区分本项目已经实现的内容与之后的升级。

| 能力 | 本课程实现 | 后续升级与触发条件 |
|---|---|---|
| 结构化决策 | JSON Schema + Python 校验 | 工具多时增加独立参数 schema |
| 状态与预算 | 先查证据后回答、步数限制、请求超时 | 长任务增加持久化检查点和总时间预算 |
| 知识检索 | 字符片段检索、分块、来源 | 同义表达漏检时加 embedding，与词法结果混合 |
| Skill | 显式清单、按需读一个 SKILL.md | 技能增多再加可靠的选择器和格式校验 |
| 记忆 | 手动确认的风格偏好 | 经验条目附来源、有效期、删除机制 |
| 可观测性 | decision/observation/summary 日志 | 增加模型版本、token 总量、错误类型、成本 |
| 训练 | CPU 路由训练；可选 LoRA | 积累真实失败与正确轨迹后再做更大训练 |
| 多 Agent | 暂未实现 | 能划分独立子任务且单 Agent 评测已稳定时再引入 |
| MCP | 暂未实现 | 要复用外部工具服务时接入标准协议 |
| Coding Agent | 暂未实现自动改代码 | 先做复制仓库中的人工确认补丁，随后再讨论隔离执行 |

### 第一个升级实验：比较更多检索片段是否更好

1. 先在 Git 提交当前可运行版本，记录四题评测和人工核对结果。
2. 把 knowledge.py 最后的 `return hits[:2]` 改为 `return hits[:1]`。
3. 运行 knowledge.py，看来源变化；再运行 `evaluate.py --backend ollama`。
4. 比较相关来源是否更集中、回答是否遗漏信息、耗时是否变化。
5. 改回 2，再比较。只有开发集支持改善，才把改动作为新版本保留。

这就是一个完整的小型系统实验：假设、单变量修改、评测、结论。

### 从当前项目走向更大的 Harness

按下面顺序做，前一步有实际需求才升级：

1. **数据与评测先扩充。** 加 30—50 个真实学习问题，人工核对答案；固定版本和输入，统计失败类型。
2. **提升检索。** 建立“这个问题应该命中哪个文档块”的标注集。先评测检索召回，再选择 embedding 模型；不要直接把生成答案好坏全部归给模型。
3. **状态持久化。** 保存 task_id、下一步、工具结果和状态版本。重启后从检查点恢复。对于发送、付款、文件写入等动作，需要幂等键和确认点，防止重试重复执行。
4. **工具协议标准化。** MCP 提供主机、客户端与服务器之间的工具/资源通信方式。它解决接入问题，权限和任务决策仍由 Harness 负责。先读 [MCP 官方架构](https://modelcontextprotocol.io/docs/learn/architecture)，只接一个只读服务。
5. **需要图与恢复时再迁移框架。** LangGraph 的持久化机制可以承接检查点场景。先阅读 [官方 Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)，再把现有“选工具→执行→回答”画成节点；用原评测比较迁移前后。
6. **可控的 Coding Agent。** 用以前课程的 mean 小题，在副本中让模型提出 diff，人工检查，只运行预先固定的测试命令。将来执行不可信代码时增加系统级隔离；普通 Python 子进程和超时并不等于沙箱。
7. **轨迹训练。** 从日志提取“问题→工具选择→观察→正确终局”。先标注错误原因与正确示范，再扩充 SFT。偏好训练需要同题优劣答案；在线 RL 还需要可信奖励和训练环境，不能只按“程序退出 0”给奖励。

这些是后续项目规格，不是本教程里已实现的功能。当前可运行的代码边界是学习助手与两种训练实验。

### 只选需要的免费资料

以下公开文档已于 2026-10-20 打开核对。只看与你当前实验相关的小节；无需看完整门课才能继续。

| 内容 | 官方入口 | 如何用 |
|---|---|---|
| Python 环境 | [中文 venv 教程](https://docs.python.org/zh-cn/3.12/tutorial/venv.html) | 核对阶段 01—02 的命令 |
| Git | [Pro Git 中文版](https://git-scm.com/book/zh/v2) | 只看第 2 章获取仓库与记录更新 |
| Agent 基本概念 | [HF Agents 中文课程](https://huggingface.co/learn/agents-course/zh-CN/unit0/introduction) | 读介绍与工具概念，暂不依赖托管实验 |
| 工程设计 | [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) | 对照工作流与 Agent 的区别 |
| 上下文组织 | [Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | 思考哪些工具结果值得放进下一轮 |
| Skill | [Agent Skills 规范](https://agentskills.io/specification) | 查 name、description、目录组织 |
| 本地推理 | [Ollama Windows](https://docs.ollama.com/windows) | 查服务、模型路径、日志 |
| 参数微调 | [PEFT Quicktour](https://huggingface.co/docs/peft/quicktour) | 对照 adapter 创建、保存、加载 |

官方文档是免费阅读入口；第三方托管推理、云算力和课程平台实验的费用不属于这些文档的免费承诺。本课程主线不依赖它们。

## 最终验收

你能现场演示以下任务，就已经从“只会复制”跨到“能解释并修改自己的 Harness”：

- 修改学习记录，工具统计随之改变。
- 新增笔记，检索能够找到并返回来源。
- 修改 Skill 或确认偏好，用真实模型观察影响。
- 运行 mock 与 ollama，解释两种决策来源的区别。
- 展示被拒绝的非法工具和来源，解释失败停在哪一步。
- 重训路由器，保存权重，用保留题报告前后结果。
- 从日志挑一个真实错误，提出可验证的改进假设。
- 选做：保存并加载 LoRA adapter，诚实报告它是否改善了工具选择。

回到原 10 天课程时，先看 Harness、工具与验证器相关部分；大型后训练、SWE-bench 全量复现与多 Agent 系统留给后续专题，不再把它们压进同一个十天入门周期。
