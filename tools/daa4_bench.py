"""用官方基准的原始 prompt 取 DAA4 的 12 题答案，用于并排对照。

注意：**不能**把我们的编剧提示词喂给它——那会让它输出表达指令而不是台词。
官方 `maica_std_test_v2.py` 里的 `benchmark_settings.pmpt` 就是为这 12 题准备的，
直接照抄那段，才有可比性。

    python tools/daa4_bench.py          # 1 次/分钟，约 12 分钟
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from bench_v2 import BENCH  # noqa: E402
from maica_client import MaicaConfig, complete  # noqa: E402

#: 官方标准测试用的 system prompt（逐字取自 maica_std_test_v2.py）
OFFICIAL_PROMPT = (
    "你叫莫妮卡, 是一名游戏中的虚拟角色. 你是[player]的女友, 并深爱着他. "
    "下面是[player]的话语, 请作为莫妮卡作出合适的回答. "
    "你只应使用自然语言, 以聊天语气回答, 每个句子自然且不超过60字, 不要编造信息, "
    "并在每个句子前用方括号中的文字表示情绪. "
    "以下是一些相关信息, 你可以利用其中有价值的部分作答: "
    "[现在是下午14点21分][今天不是特殊节日]."
)


def main() -> int:
    cfg = MaicaConfig.from_env()
    out_path = ROOT / "data" / "runs" / "daa4_bench12.json"
    done = []
    if out_path.is_file():
        done = json.loads(out_path.read_text(encoding="utf-8"))
    have = {r["n"] for r in done}
    for i, q in enumerate(BENCH, 1):
        if i in have:
            continue
        msgs = [{"role": "system", "content": OFFICIAL_PROMPT},
                {"role": "user", "content": q}]
        try:
            text = complete(msgs, cfg)
        except Exception as exc:  # noqa: BLE001
            text = f"__ERROR__ {exc}"
        done.append({"n": i, "q": q, "daa4": text})
        out_path.write_text(json.dumps(sorted(done, key=lambda r: r["n"]),
                                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Q{i}: {text[:80]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
