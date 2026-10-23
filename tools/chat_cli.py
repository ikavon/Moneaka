"""命令行聊天测试：一轮一轮地看编剧 → 润色 → 审查的全过程。

界面版是 `app/chat_app.py`；两者共用 `moneaka.turn`，逻辑不会漂移。

    python tools/chat_cli.py                    # 默认开全部审查
    python tools/chat_cli.py --no-b             # 关掉语义复核（省一次云调用）
    python tools/chat_cli.py --adapter data/adapters/run8 --k 3

命令（在聊天里直接输入）：
    /plan      显示上一轮的表达指令
    各层开关   /a /b /c /bank /tight  切换
    /reset     清空对话历史
    /quit      退出
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka.turn import Runtime, TurnConfig, run_turn  # noqa: E402

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def show(res) -> None:
    if res.error:
        print(f"  {BOLD}! {res.error}{RESET}")
        return
    print(f"  {DIM}plan{RESET}  {res.plan.replace(chr(10), '  ')}")
    if res.c_flags:
        print(f"  {DIM}C 拦下{RESET} {'；'.join(res.c_flags)}"
              + ("（已重写）" if res.c_repaired else ""))
    print(f"  {BOLD}台词{RESET}  {res.polish}")
    bits = []
    if res.a_flags:
        bits.append("A: " + "；".join(res.a_flags))
    if res.b_verdict:
        bits.append(f"B: {res.b_verdict} {res.b_note}".strip())
    if res.attempts:
        bits.append(f"重采 {res.attempts} 次")
    if res.degraded:
        bits.append("已退化直译")
    if bits:
        print(f"  {DIM}{' | '.join(bits)}{RESET}")
    if len(res.candidates) > 1:
        print(f"  {DIM}候选 {len(res.candidates)} 个{RESET}")
    print(f"  {DIM}耗时 编剧 {res.timings.get('编剧', 0):.1f}s / "
          f"润色+A+B {res.timings.get('润色+A+B', 0):.1f}s{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/models/Qwen3-1.7B")
    ap.add_argument("--adapter", default="data/adapters/run8")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--no-c", action="store_true")
    ap.add_argument("--c-llm", action="store_true",
                    help="plan 审查再叠一层强模型（看历史，能抓答非所问；每轮多一次云调用）")
    ap.add_argument("--no-a", action="store_true")
    ap.add_argument("--no-b", action="store_true")
    ap.add_argument("--no-bank", action="store_true")
    ap.add_argument("--loose-tight", action="store_true", help="关掉编剧长度纪律")
    ap.add_argument("--nsfw", action="store_true")
    args = ap.parse_args()

    cfg = TurnConfig(base=args.base, adapter=args.adapter, k=args.k,
                     temperature=args.temperature,
                     use_c=not args.no_c, use_c_llm=args.c_llm,
                     use_a=not args.no_a, use_b=not args.no_b,
                     use_bank=not args.no_bank, tight=not args.loose_tight,
                     nsfw=args.nsfw)
    print("加载模型…（首次约 20~40 秒）")
    t0 = time.time()
    rt = Runtime.build(cfg)
    print(f"就绪（{time.time() - t0:.0f}s）。输入 /quit 退出，/help 看命令。\n")

    history: list[dict] = []
    log: list[str] = []
    last = None
    while True:
        try:
            line = input(f"{BOLD}你{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        text = line.strip()
        if not text:
            continue
        if text in ("/quit", "/exit", "/q"):
            break
        if text == "/help":
            print("  /plan 上次指令 | /a /b /c /bank /tight 切换各层 | /reset 清历史 | /quit")
            continue
        if text == "/plan" and last:
            print(f"  {last.plan}")
            continue
        if text == "/reset":
            history.clear()
            log.clear()
            print("  历史已清空")
            continue
        if text in ("/a", "/b", "/c", "/bank", "/tight"):
            attr = {"a": "use_a", "b": "use_b", "c": "use_c",
                    "bank": "use_bank", "tight": "tight"}[text[1:]]
            setattr(cfg, attr, not getattr(cfg, attr))
            print(f"  {attr} = {getattr(cfg, attr)}")
            continue

        res = run_turn(rt, cfg, text, history=history, log=log)
        last = res
        show(res)
        history.append({"role": "user", "content": text})
        if res.log_line:
            log.append(res.log_line)
        if res.polish:
            history.append({"role": "assistant", "content": res.polish})

    print(f"\n本轮共 {len(history) // 2} 轮对话。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())