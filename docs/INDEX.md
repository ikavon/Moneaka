# 文档索引

按「想干什么」找文档，而不是按时间。**带 ✅ 的是当前有效结论，带 ⚠️ 的是历史记录。**

## 想了解这是什么

| 文档 | 内容 |
|---|---|
| [`../README.md`](../README.md) | 项目概览、快速开始、repo 结构 |
| [`DESIGN.md`](DESIGN.md) | 为什么把「说什么」和「怎么说」拆开 |
| [`PIPELINE_SINGLE_TURN.md`](PIPELINE_SINGLE_TURN.md) ✅ | 单轮流程规格：每一步、每个字段、每层各自看得到什么 |
| [`SHOWCASE_12Q.md`](SHOWCASE_12Q.md) ✅ | 官方基准 12 题的实际输出（玩家 → 莫妮卡） |

## 看数字

| 文档 | 内容 |
|---|---|
| [`BASELINE.md`](BASELINE.md) ✅ | **评测协议**（greedy + 206 条冻结留出 + 双指标）与基线数字 |
| [`CHAIN_RESULTS.md`](CHAIN_RESULTS.md) ✅ | 全链路 A+B+C 实测：语义 ≤2 从 17 条降到 7 条 |
| [`JUDGE_POLICY.md`](JUDGE_POLICY.md) ✅ | **判官规则**：角色扮演模型不能当判官；哪类判据该用谁 |
| [`STYLE_GUIDE.md`](STYLE_GUIDE.md) | 从官方语料统计出来的风格特征 |
| [`results/`](results/) | 各次评测的 json 汇总（选点记录） |
| [`training_logs/`](training_logs/) | 训练 loss 曲线（文字日志） |

## 看过什么路、为什么放弃

| 文档 | 结论 |
|---|---|
| [`STAGE3_DPO.md`](STAGE3_DPO.md) | ⛔ **DPO 无效**（三组超参、语义只在噪声内变化） |
| [`DISTILL_PILOT.md`](DISTILL_PILOT.md) | ⛔ **教师蒸馏不划算**（同指令下 DAA4 盲评输给我们） |
| [`STAGE_C_DATA_SIDE.md`](STAGE_C_DATA_SIDE.md) | ⛔ 改 plan 写法会让模型照抄；✅ 推理侧机械闸门有效 |
| [`COMPARE_PROMPTS.md`](COMPARE_PROMPTS.md) | ⛔ **「格式必须靠微调」判断作废**：3 臂原始输出并列；精写提示词的标签合法率与润色器持平（各 114/115）。用户判断：润色器输出未显著优于精炼提示词直出 |
| [`COMPARE_SIZE_4B.md`](COMPARE_SIZE_4B.md) | ⛔ **参数不是瓶颈**：同数据同超参下 4B 指令遵从更差（标签序列 84.5% vs 99.0%，抄写 0.82 vs 0.61） |

## 审查层

| 文档 | 内容 |
|---|---|
| [`BYPASS_B_C_D.md`](BYPASS_B_C_D.md) | B 台词复核 / C 前置审查 / D 编者范例库的实现与数字 |
| [`PIPELINE_SINGLE_TURN.md`](PIPELINE_SINGLE_TURN.md) | C 的两层、A 的四类、B 的触发条件 |

## 服务与接入

| 文档 | 内容 |
|---|---|
| [`SERVICE_POLISH.md`](SERVICE_POLISH.md) ✅ | `/polish` 契约、错误码、并发策略、OpenAI 兼容外壳 |
| [`../streamlit_app.py`](../streamlit_app.py) | 测试台（两种后端：进程内 / HTTP） |

## 事故记录（都是真的踩过）

| 文档 | 事故 |
|---|---|
| [`INCIDENT_STALE_BASE.md`](INCIDENT_STALE_BASE.md) | DPO 训在错误的合并基座上，四组结论作废 |
| [`JUDGE_POLICY.md`](JUDGE_POLICY.md) | 用角色扮演模型当判官，一次对照实验作废 |

## 历史（数字已过时，别引用）

| 文档 | 说明 |
|---|---|
| [`STAGE0_RESULTS.md`](STAGE0_RESULTS.md) | 零样本阶段的探索（评测协议当时还没定） |
| [`STAGE1_RESULTS.md`](STAGE1_RESULTS.md) | 第一版 LoRA（数据泄漏 bug 之前） |
| [`STAGE2_SCHEMA.md`](STAGE2_SCHEMA.md) | 受控 schema 的设计（仍然有效，但数字以 BASELINE 为准） |
| [`HUMAN_EVAL.md`](HUMAN_EVAL.md) / [`HUMAN_EVAL_CONVO.md`](HUMAN_EVAL_CONVO.md) | 人工盲测表（n=20 时代的数字已作废） |
| [`DATA_INVENTORY.md`](DATA_INVENTORY.md) | 早期对官方数据的盘点 |
| [`PLAN_NEXT.md`](PLAN_NEXT.md) | 阶段计划与执行状态 |