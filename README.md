# Moneaka

**莫妮卡口吻润色器**：通用 LLM 负责「说什么」，Moneaka 负责「怎么说」。

```
通用大模型（任意供应商，负责推理/记忆/决策）
      │  平实草稿
      ▼
┌──────────────────────────┐
│ Moneaka 润色器（1.7B 级） │  ← 只做语言改写，不许改内容
└──────────────────────────┘
      │  莫妮卡口吻
      ▼
   玩家 / MAS / Minecraft / QQ bot
```

动机：MAICA 的 DAA4 把「像莫妮卡一样说话」微调进了权重，但 397B-A17B 的体量决定了
普通用户无法本地部署。而「说话方式」是低维特征，小模型足够学会；「智力」才需要大模型。
把两者拆开，就能同时要到高智力 + 莫妮卡皮 + 本地零成本风格。

## 为什么值得做（数据实锤）

官方 DAA4 训练语料读下来（`docs/DATA_INVENTORY.md`）确认了三件事：

1. **风格契约是明文规定，不用我们发明**——MAICA 骨架 system prompt 原文就写着
   「每个句子自然且不超过60字」「在每个句子前用方括号中的文字表示情绪」
2. **输出极短**：单轮中位 35 字、最长 126 字 ⇒ 训练序列 ~256 token 足够 ⇒
   8GB 显存（RTX 5060 Laptop）上 1.7B/4B LoRA 很宽裕
3. **情绪标签是闭合小词表**（本分片 18 个），下游 MTTS/动画映射依赖它 ⇒
   润色器只能复用这张表，不能自创

`moneaka stats` 会把契约校验器拿真实数据当考题：**当前 226/226 全过**，
说明校验阈值就是官方分布。

## 用法

环境：`moneaka` conda 环境（Python 3.12 / torch 2.14.1+cu130 / peft / trl / bitsandbytes）。

```bash
cd D:/Python_projects/Moneaka
export MONEAKA_BASE_URL=...        # 强模型端点（OpenAI 兼容）
export MONEAKA_API_KEY=...
export MONEAKA_STRONG_MODEL=...
export MONEAKA_LOCAL_BASE_URL=http://localhost:1234/v1   # LM Studio / llama.cpp
export MONEAKA_LOCAL_MODEL=Qwen3-1.7B

PYTHONPATH=src python -m moneaka.cli stats              # 不联网，先跑这个
PYTHONPATH=src python -m moneaka.cli destyle --limit 60 # 造 (草稿, 原稿) 平行语料
PYTHONPATH=src python -m moneaka.cli polish --name z1    # 润色 + 对照组
PYTHONPATH=src python -m moneaka.cli eval --name z1 --embed
```

三臂对照（`eval` 的表格）：

| arm | 是什么 | 角色 |
|---|---|---|
| `reference` | 官方数据里的真实 DAA4 输出 | 天花板 |
| `direct` | 通用模型按官方骨架 prompt 直接扮演 | 竞品（纯 prompt 路线） |
| `polish` | 通用模型出草稿 → Moneaka 润色 | 本项目主张 |

指标：`contract_pass`（机械校验）· `semantic`（评委：语义保持 0-4）·
`style`（评委：像不像她 0-4，评分轴取自官方 chosen/rejected 对照）· `cos_ref`。

## 阶段

- [x] 阶段 0a 契约 + 校验器 + 语料加载 + 三臂评测管线
- [ ] 阶段 0b 跑通往返：去风格化 → 零样本润色 → 评测（需 API key 与本地小模型）
- [ ] 阶段 1  LoRA 训练润色器（1.7B 主攻，4B 对照）
- [ ] 阶段 2 插件化：OpenAI 兼容端点 / MCP server，接 maidllmlocal 的 `player_relay`

设计与取舍见 `docs/DESIGN.md`；数据盘点见 `docs/DATA_INVENTORY.md`。

## 许可提醒

模型与训练数据是 `maica-tos`（自定义条款），训练数据未附许可说明。
自用实验没问题，**公开发布前必须与 MAICA 官方对齐条款**。

## 聊天测试台（试玩 / 对照各层）

两个入口，逻辑共用 `moneaka.turn`，不存在两套实现：

* **界面版**：双击 `run_chat_ui.cmd`（或 `streamlit run streamlit_app.py`）
* **命令行版**：双击 `run_chat_cli.cmd`（或 `python tools/chat_cli.py`）

界面上能看到一轮对话的每一步：编剧写的表达指令、润色器说的台词、C/A/B 三层审查的判决、
重采几次、耗时。侧边栏可以单独开关各层，用来对照「加了这层到底有没有用」。

命令行里的命令：`/plan` 看上次指令，`/a /b /c /bank /tight` 切换各层，`/reset` 清历史。

一轮的真实耗时（RTX 5060 Laptop，默认配置）：编剧约 1.5~3 秒，润色+A+B 约 4 秒。
`TurnConfig.max_audits` 限制了一轮里最多花几次 B 审查，避免重采把延迟拖到 15 秒。
