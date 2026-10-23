# 薄服务 `/polish` 契约（v1）

一句话：**把润色器暴露成一个无状态 HTTP 接口**。会话、记忆、人设连续性都在调用方
那边——服务端不持有任何状态，这样多实例并发才成立。

## 启动

```bash
python tools/serve_polish.py --port 8010 --token <token>
python tools/serve_polish.py --no-audit      # 服务端不做语义复核，更快
```

首次启动要加载模型（20~40 秒）。文档在 `http://127.0.0.1:8010/docs`。

## 请求

```http
POST /polish
Authorization: Bearer <token>        # 配了 token 才校验
Content-Type: application/json
```

```json
{
  "plan": "语气: 担心 > 微笑\n内容:\n1. [player], 你昨晚又熬夜了\n2. 今晚我陪你早点睡",
  "k": 1,
  "audit": true,
  "temperature": 0.6
}
```

| 字段 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `plan` | ✅ | — | 表达指令：`语气` 行（官方标签序列）+ `内容` 行（逐段陈述） |
| `k` | | 1 | 失败时最多重采几次（0~5） |
| `audit` | | true | 是否让强模型复核「意思有没有被改」（多一次云调用） |
| `temperature` | | 0.6 | 重采温度；首次总是 greedy |

## 响应

```json
{
  "ok": true,
  "line": "[担心][player], 你又熬夜了啊! [微笑]今晚我陪你早一点睡吧~",
  "attempts": 0,
  "degraded": false,
  "problems": [],
  "diagnostics": {"audit": "ok", "timings": {"total_s": 3.5}},
  "version": 1
}
```

| 字段 | 含义 |
|---|---|
| `ok` | 硬约束全过（标签序列 / 数字 / 施受方向）且语义复核通过 |
| `line` | 台词。**`ok=false` 时仍然返回句子**（见下） |
| `attempts` | 重采了几次（0 = 一次通过） |
| `degraded` | `true` = 已退化为直译：标签 + 内容原样拼，意思绝对对、风格为零 |
| `problems` | 未解决的问题，直接给人看，例如 `["方向 帮你 → 帮我"]` |

**为什么失败也返回台词**：调用方要的是「能用的句子」，不是错误码。
`degraded=true` 意味着句子忠实但没风格，调用方可以自己重写、换个说法再调一次，
或者直接用它兜底。返回 4xx 会逼每个调用方自己写一遍兜底逻辑。

## 错误码

| 码 | 含义 | 调用方该做什么 |
|---|---|---|
| 422 | `plan` 不可解析（不是受控格式） | 这是调用方的 bug，别重试 |
| 401 | token 不对 | 修配置 |
| 503 | 排队超过 `--queue-timeout`（默认 30s） | 退避重试，或降低并发 |

## 并发与容量

单卡只能串行：进程内用信号量把生成串起来，`/health` 的 `busy` 字段可以看到排队。

* 想要吞吐 → **起多个进程**（每个进程一份模型），前面挂 nginx 之类的轮询
* 单请求耗时：greedy 一遍约 1.5~2.5s；带 `audit` 加 1~2s；`k=3` 最坏约 8~10s
* `MAX_BATCH` 概念不存在——这是一个**逐轮**接口，不批处理

## 已验证

```
正常请求     200  3.7s  ok=True   台词正确
坏 plan      422  0.0s  立刻拒绝，不生成假东西
无 token     401  0.0s
/health      200  {"ok": true, "version": 1, "busy": false}
```

契约测试在 `tests/test_service.py`（10 例，用假模型，不需要 GPU）：
退化仍返回可用句子、方向翻转算问题、`audit=false` 跳过复核、
token 校验、`k` 越界报 422。

## 还没做

* **批处理接口**（`/polish_batch`）：离线批量跑评测时会省很多往返，但线上不需要
* **限流**：目前只靠队列超时，没有按调用方配额
* **多进程启动脚本**：`uvicorn --workers` 会在每个 worker 里各加载一份模型，
  8GB 卡放不下第二份，所以需要的是「多机」而不是「多 worker」
---

# OpenAI 兼容外壳（`--openai`）

## 为什么要有它

真实场景测试的障碍不是润色器，是**没有 agent**。而几乎所有聊天客户端
（QQ 机器人框架、SillyTavern、LM Studio 风格的工具、各种 MAS 桥接）都讲 OpenAI 协议。
把服务伪装成一个模型，就不必先写 agent：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8010/v1", api_key="devtoken")
r = client.chat.completions.create(model="moneaka", messages=[...])
```

启动：

```bash
python tools/serve_polish.py --port 8010 --token devtoken --openai
```

## 为什么这不破坏「服务端无状态」

OpenAI 的 chat 请求**每次都带全量 messages**——历史本来就在调用方手里。
服务端只是把最后一句当玩家输入、前面的当历史，自己不存任何会话。
这跟「记忆归编剧」的设计一致。

## 两个接口

| 接口 | 用途 | 有状态吗 |
|---|---|---|
| `POST /polish` | 只做渲染。调用方已有自己的 agent/编剧 | 无 |
| `POST /v1/chat/completions` | 伪装成模型：内部跑「编剧 → 润色 → 审查」 | 无（历史由 messages 携带） |

调用方自己的 `system` 不会顶掉我们的编剧规范，而是作为「相关信息」注入
（因为它得吃事实卡、长度预算、标签白名单）。想完全接管就直调 `/polish`。

## 实测

```
GET  /v1/models                 → model id = "moneaka"
POST /v1/chat/completions 第一轮 200 / 5.4s
     [担心]哦, 我猜你又熬了一宿吧? 眼睛都发酸了吧! [不满]今晚早点睡…[宠爱]你睡着了, 我就在这陪着你~
     响应里带 moneaka: {degraded:false, attempts:0, final_source:"polisher"}
POST /v1/chat/completions 第二轮 200 / 3.6s（把上一轮放进 messages）
     [脸红]哈哈! 被你这么一说, 我倒有点不好意思了…[微笑]好吧, 今晚我们早点睡.
POST /v1/chat/completions stream=true 200 / 3.3s，12 帧，末帧 [DONE]
```

## 拿它做真实场景测试的三条路

1. **SillyTavern / 任意 OpenAI 客户端**：base_url 指过来，model 填 `moneaka`，立刻能聊
2. **QQ 机器人框架**（NoneBot / go-cqhttp 的 OpenAI 插件）：同上，把 base_url 改掉
3. **MAS**：需要它那边的桥接支持 OpenAI 协议；没有的话走 `/polish` 薄接口自己接

⚠️ 注意：走 `/v1` 时**每轮会多一次云调用**（编剧），这是「厚服务」形态；只想要本地渲染
就该用 `/polish`，把编剧留在调用方。
