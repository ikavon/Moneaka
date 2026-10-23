"""启动薄服务：`/polish`。

    python tools/serve_polish.py                      # 默认 127.0.0.1:8010
    python tools/serve_polish.py --port 8010 --token xxx
    python tools/serve_polish.py --no-audit           # 服务端不做语义复核（更快）

只暴露润色器（无状态）。agent、历史、人设连续性都在调用方那边——这是刻意的，
否则服务端就得承担会话状态，多实例并发立刻变成分布式一致性问题。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka.service import Polisher, create_app  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/models/Qwen3-1.7B")
    ap.add_argument("--adapter", default="data/adapters/run8")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--token", default=None, help="留空则不校验 Authorization")
    ap.add_argument("--nsfw", action="store_true",
                    help="OpenAI 外壳里给编剧加 NSFW 补丁")
    ap.add_argument("--no-audit", action="store_true",
                    help="服务端不调用强模型复核（请求里的 audit 会被忽略为 false）")
    ap.add_argument("--queue-timeout", type=float, default=30.0)
    ap.add_argument("--openai", action="store_true",
                    help="额外挂 /v1/chat/completions，让现成聊天客户端把它当模型用")
    args = ap.parse_args()

    import uvicorn
    from moneaka.llm import LLM
    from moneaka.localhf import get_model

    print("加载润色器…（首次约 20~40 秒）")
    model = get_model(args.base, adapter=args.adapter)
    llm = None if args.no_audit else LLM()
    polisher = Polisher(model=model, llm=llm)
    if args.no_audit:
        print("注意：--no-audit 时请求里的 audit 会失效（服务端没有 LLM 客户端）")

    app = create_app(polisher, token=args.token, queue_timeout=args.queue_timeout)
    if args.openai:
        from moneaka.openai_facade import add_openai_routes
        from moneaka.turn import Runtime, TurnConfig

        rt = Runtime(model=model, llm=llm, bank=None)
        add_openai_routes(app, rt, TurnConfig(base=args.base, adapter=args.adapter,
                                              nsfw=args.nsfw), token=args.token)
        print("已挂 OpenAI 兼容接口: /v1/chat/completions  (model 名填 moneaka)")
    print(f"就绪 → http://{args.host}:{args.port}/polish  （文档 /docs）")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())