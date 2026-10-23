# Moneaka

**莫妮卡口吻润色器**：把「说什么」和「怎么说」拆开——任何大模型负责前者，一个
1.7B 的本地小模型负责后者。

```
玩家消息 + 记忆/事实卡
        │
        ▼
   任意大模型（编剧）─────────────── 决定「说什么、什么情绪」
        │  表达指令：语气=标签序列 / 内容=逐段要点
        ▼
┌───────────────────────────┐
│ Moneaka（1.7B，LoRA）      │  只决定「怎么说」，不许改内容
└───────────────────────────┘
        │  台词（带 [情绪标签]）
        ▼
   MAS / Minecraft / QQ bot / 任意聊天场景
```

动机：MAICA 的 DAA4 把"像莫妮卡一样说话"微调进了 397B 的权重里，普通用户跑不起来。
而**说话方式是低维特征**，小模型学得会；智力才需要大模型。拆开就能同时拿到
"大模型的脑子 + 莫妮卡的口吻 + 本地零成本"。

> 这不是"换个模型做角色扮演"，而是**把风格做成一个可插拔的层**：
> 润色器无状态、不看历史、不记任何东西，因此可以多实例并发，服务任何 agent。

---

## 现在能做什么

| 能力 | 状态 |
|---|---|
| 把表达指令渲染成莫妮卡台词 | ✅ 标签序列遵从 **99.0%**，逐位标签 **100%** |
| 语义保真 | ✅ 机械闸门 + 语义复核后，灾难性漂移 **17/206 → 7/206** |
| 两种服务形态 | ✅ 薄服务 `POST /polish`（无状态）／OpenAI 兼容 `POST /v1/chat/completions` |
| 聊天测试台 | ✅ Streamlit（可切换"进程内"或"HTTP 服务"两种后端） |
| 官方基准（12 题 / 99 分） | ⚠️ 只能出自动预评；最终分必须人工打 |

**已知短板**：长指令（>55 字）渲染时模型倾向于照抄指令措辞（相似度 0.58~0.88）。
详见 [docs/HANDOFF_EDGE.md](docs/HANDOFF_EDGE.md) 的"已知限制"。

---

## 快速开始

```bash
# 1) 依赖（只需一次）
pip install -e .
pip install fastapi uvicorn streamlit          # 服务和界面

# 2) 配端点（不进 git）
cat > configs/private.yaml <<'YAML'
endpoints:
  strong:
    base_url: https://your-endpoint/v1
    api_key: sk-...
    model: your-model
YAML

# 3) 起服务（润色器 + OpenAI 兼容外壳）
python tools/serve_polish.py --port 8010 --token devtoken --openai

# 4) 起测试台
streamlit run streamlit_app.py        # Windows 也可双击 run_chat_ui.cmd
```

> **基座权重和 LoRA 不在仓库里**（见下方"许可"）。没有它们时，服务起不来——
> 但 `tests/` 里的 100+ 个用例用假模型，不需要 GPU 也能跑：
> `python -m pytest -q`

---

## 仓库结构

```
src/moneaka/          核心库
  turn.py             一轮的编排：编剧 → C 审查 → 润色 → A/B 审查 → 兜底
  screenwriter.py     编剧（大模型侧）：提示词、格式校验、长度预算
  prompts.py          润色器的 system prompt 与铁律
  guarded.py          A 层机械闸门（标签/数字/施受方向/语言）+ 标签重贴
  audit.py            C 前置审查（机械）+ B 台词复核（模型）
  service.py          薄服务与 /turn 调试接口
  openai_facade.py    OpenAI 兼容外壳
  localhf.py          本地模型加载与生成
  contract.py         输出契约（标签白名单、句长、校验）
tools/                可执行脚本（评测、训练、服务、基准）
docs/                 文档（入口：docs/INDEX.md）
tests/                100+ 用例，全部用假模型，无需 GPU
data/                 只有输入类资源（事实卡等）；训练数据/权重/日志都不入库
```

---

## 文档

从 [docs/INDEX.md](docs/INDEX.md) 进。三份最该先看：

- [docs/PIPELINE_SINGLE_TURN.md](docs/PIPELINE_SINGLE_TURN.md) —— 一轮到底经过什么
- [docs/BASELINE.md](docs/BASELINE.md) —— 评测协议与基线（数字都在这里）
- [docs/JUDGE_POLICY.md](docs/JUDGE_POLICY.md) —— 判官规则（吃过亏之后写死的）

给 MAICA 团队的材料：[docs/HANDOFF_EDGE.md](docs/HANDOFF_EDGE.md)、
[docs/SAMPLES_FOR_EDGE.md](docs/SAMPLES_FOR_EDGE.md)。

---

## 许可

* **代码**：待定（建议 MIT，等你确认）。
* **事实卡 `data/canon/`**：本项目编写，随代码一起发布。
* **训练数据、权重、LoRA、评测输出**：**不在本仓库内**。它们派生自
  MAICA 的语料（`maica-tos`），是否可发布仍在与 MAICA 团队确认。
* 上游教程 `TalkingLikeMonika`（无 LICENSE 文件）不随仓库分发，只留来源链接。

细节见 [LICENSING.md](LICENSING.md)。

## 相关

- MAICA：<https://github.com/edgeinfinity/MAICA>（本项目是其本地化路线的尝试）
- 上游风格教程：<https://github.com/edgeinfinity1/TalkingLikeMonika>