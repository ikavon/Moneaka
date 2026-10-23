"""Moneaka 台词测试台。

一轮对话的全过程都摊在眼前：编剧写的表达指令、润色器说的台词、每一层审查的判决。
各层可以在侧边栏单独开关，方便对照「加了这层到底有没有用」。

    streamlit run streamlit_app.py

逻辑与命令行版 `tools/chat_cli.py` 共用 `moneaka.turn`，不存在两套实现。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import moneaka  # noqa: E402
from moneaka.turn import Runtime, TurnConfig, run_turn  # noqa: E402

# 包里的代码改过但进程还跑着旧模块 → 立刻说清楚，别让它以「莫名奇妙的 TypeError」出现
if moneaka.build_stamp() > moneaka.BUILD_STAMP + 0.001:
    st.error("`src/moneaka` 里的代码已经改过，但当前进程还在跑旧模块。"
             "请重启界面（关掉窗口重新双击 run_chat_ui.cmd）。", title="需要重启")

st.set_page_config(page_title="Moneaka 台词测试台", page_icon=":material/forum:",
                   layout="wide")


@st.cache_resource(show_spinner="加载润色器…（首次约 20~40 秒）")
def load_runtime(base: str, adapter: str, with_bank: bool) -> Runtime:
    return Runtime.build(TurnConfig(base=base, adapter=adapter, use_bank=with_bank))


def adapters() -> list[str]:
    d = ROOT / "data" / "adapters"
    if not d.is_dir():
        return ["data/adapters/run8"]
    out = [f"data/adapters/{p.name}" for p in sorted(d.iterdir())
           if p.is_dir() and (p / "adapter_config.json").is_file()]
    return out or ["data/adapters/run8"]


def init_state() -> None:
    if "turns" not in st.session_state:
        st.session_state.turns = []          # [{user, result, history}] 供回放
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.log = []        # 给编剧看的对话历史
    if "log" not in st.session_state:
        st.session_state.log = []            # 编剧自己的决策轨迹（审查结论等）


init_state()


def call_http(url: str, token: str, user: str, history: list, log: list,
              overrides: dict | None = None):
    """调服务的 /turn。返回一个和 TurnResult 字段兼容的简单对象。"""
    import httpx

    payload = {"user": user, "history": history, "log": log,
               "overrides": overrides or {}}
    r = httpx.post(f"{url.rstrip('/')}/turn", json=payload, timeout=180,
                   headers={"Authorization": f"Bearer {token}"} if token else {})
    r.raise_for_status()
    d = r.json()

    class R:
        pass

    o = R()
    for k, v in d.items():
        setattr(o, k, v)
    o.error = d.get("error") or ""
    o.ok = bool(d.get("ok"))
    return o


def asdict_like(res) -> dict:
    return {"plan": res.plan, "polish": res.polish, "c_flags": res.c_flags,
            "a_flags": res.a_flags, "b_verdict": res.b_verdict,
            "b_note": res.b_note, "attempts": res.attempts,
            "degraded": res.degraded, "timings": res.timings,
            # 这几个字段是为了排查：之前编剧失败时导出里只看得到空台词，像「她没说话」
            "error": res.error, "retagged": getattr(res, "retagged", False),
            "empty_outputs": getattr(res, "empty_outputs", 0),
            "final_source": getattr(res, "final_source", "polisher")}


# ---------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.subheader("后端")
    backend = st.radio("这一轮怎么跑", ["进程内（直接加载模型）", "HTTP 服务（/turn）"],
                       horizontal=False, label_visibility="collapsed",
                       help="选中 HTTP 时，界面只是服务的一个客户端——测的就是将来"
                            "QQ/MAS 会用到的那条链路")
    svc_url = st.text_input("服务地址", "http://127.0.0.1:8010")
    svc_token = st.text_input("token", "devtoken", type="password")

    st.subheader("模型")
    adapter = st.selectbox("润色器 adapter", adapters(), index=0)
    base = st.text_input("基座", "data/models/Qwen3-1.7B")

    st.subheader("审查层")
    use_c = st.toggle("C 前置审查", value=True,
                      help="渲染前查 plan：回指词、设定冲突、能力越界、复读")
    use_c_llm = st.toggle("C 模型审查", value=False,
                          help="再叠一层强模型，带历史判「有没有答非所问」；"
                               "每轮多一次云调用。真人 plan 上误报 5%")
    use_a = st.toggle("A 机械闸门", value=True,
                      help="生成后查标签序列与数字，不过就重采")
    use_b = st.toggle("B 台词复核", value=True,
                      help="触发式送强模型判「意思有没有被改」")
    use_bank = st.toggle("编剧先例检索", value=True,
                         help="从语料里检索相似局面的反应给编剧参考")
    tight = st.toggle("编剧长度纪律", value=True,
                      help="内容总字数 ≤60、段数 ≤3；超出会重写")

    st.subheader("生成")
    k = st.slider("最多重采次数", 0, 5, 3)
    temperature = st.slider("重采温度", 0.2, 1.0, 0.6, 0.05)
    nsfw = st.toggle("NSFW 补丁", value=False)

    st.divider()
    if st.button("清空对话", icon=":material/delete:", width="stretch"):
        st.session_state.turns = []
        st.session_state.history = []
        st.session_state.log = []
        st.rerun()
    if st.session_state.turns:
        dump = json.dumps([{"user": t["user"], **asdict_like(t["result"])}
                           for t in st.session_state.turns],
                          ensure_ascii=False, indent=1)
        st.download_button("导出本轮记录", dump, file_name="moneaka_turns.json",
                           mime="application/json", icon=":material/download:",
                           width="stretch")


# ---------------------------------------------------------------- 主体
st.title("台词测试台")
st.caption("编剧决定说什么，润色器决定怎么说；中间的表达指令是唯一契约。")

use_http = backend.startswith("HTTP")
if use_http:
    st.caption("HTTP 模式下，模型由服务端决定；审查层与生成参数会作为 overrides "
               "传给服务端（`/turn` 支持这些开关）。")

cfg = TurnConfig(base=base, adapter=adapter, use_c=use_c, use_c_llm=use_c_llm,
                 use_a=use_a, use_b=use_b,
                 use_bank=use_bank, tight=tight, nsfw=nsfw, k=k,
                 temperature=temperature)

for turn in st.session_state.turns:
    res = turn["result"]
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(res.polish if res.polish else f"⚠️ {res.error}")
        tags = []
        if res.degraded:
            tags.append(":orange-badge[已退化直译]")
        if res.b_verdict and res.b_verdict != "ok":
            tags.append(":red-badge[B 判漂移]")
        if res.a_flags:
            tags.append(":red-badge[A 未通过]")
        if tags:
            st.markdown(" ".join(tags))
        with st.expander("过程", icon=":material/account_tree:"):
            st.markdown("**表达指令**")
            st.code(res.plan or "（无）", language="text")
            rows = {
                "C 前置审查": "；".join(res.c_flags) or "通过",
                "A 机械闸门": "；".join(res.a_flags) or "通过",
                "B 台词复核": f"{res.b_verdict} {res.b_note}".strip() or "未触发",
                "重采次数": str(res.attempts),
                "耗时": " / ".join(f"{v:.1f}s" for v in res.timings.values()),
            }
            st.dataframe([{"环节": k2, "结果": v} for k2, v in rows.items()],
                         hide_index=True, width="stretch")
            if len(res.candidates) > 1:
                with st.expander(f"{len(res.candidates)} 个候选"):
                    for i, c in enumerate(res.candidates, 1):
                        st.markdown(f"{i}. {c}")

prompt = st.chat_input("对莫妮卡说点什么…", submit_mode="disable")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        t0 = time.time()
        with st.status(":shimmer[编剧正在写表达指令]", type="compact") as status:

            def on_stage(name: str) -> None:
                status.update(label=f":shimmer[{name}]")

            if use_http:
                try:
                    res = call_http(svc_url, svc_token, prompt,
                                    st.session_state.history, st.session_state.log,
                                    overrides={"use_c": use_c, "use_c_llm": use_c_llm,
                                               "use_a": use_a, "use_b": use_b,
                                               "tight": tight, "k": k,
                                               "temperature": temperature})
                except Exception as exc:  # noqa: BLE001
                    res = None
                    st.session_state.last_error = f"{type(exc).__name__}: {exc}"
            else:
                rt = load_runtime(base, adapter, use_bank)
                res = run_turn(rt, cfg, prompt, history=st.session_state.history,
                               on_stage=on_stage, log=st.session_state.log)
            status.update(label=f"用时 {time.time() - t0:.1f}s", state="complete")

        if res is None:
            st.error(st.session_state.get("last_error", "请求失败"), title="服务没通")
            st.stop()

        if res.error:
            st.error(res.error, title="这一轮没跑通")
        else:
            st.markdown(res.polish)
            st.caption(f"重采 {res.attempts} 次 · "
                       f"{'已退化直译' if res.degraded else '正常生成'}")
            with st.expander("过程", icon=":material/account_tree:"):
                st.code(res.plan, language="text")
                if res.c_flags:
                    st.warning("；".join(res.c_flags), title="C 前置审查")

    st.session_state.turns.append({"user": prompt, "result": res})
    st.session_state.history.append({"role": "user", "content": prompt})
    if res.log_line:
        st.session_state.log.append(res.log_line)
    if res.polish:
        st.session_state.history.append({"role": "assistant", "content": res.polish})
    st.rerun()