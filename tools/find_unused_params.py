"""静态检查：找出「签名里有、函数体里没用」的参数。

## 为什么要专门写这个

今天连续踩了三次同一个坑：`build_system()` 的 `canon` / `tight` / `examples`
三个参数都写在签名里、函数体里一次都没用到。调用方传了，什么也没发生，
而且**不报错**——于是实验照跑、分数照出，结论全错（`examples` 那次直接废掉了一个实验）。

这类 bug 用测试很难抓（外部行为只是"没变化"），用 lint 也抓不到（参数被"使用"在
签名里就是合法的）。所以单独做一次静态扫描。

    python tools/find_unused_params.py                # 全项目
    python tools/find_unused_params.py src/moneaka    # 只看包

判定：函数体里**一次都没出现**该参数名（含 f-string、闭包、嵌套函数里的引用）。
故意不用的参数用 `_` 开头命名，或者放进下面的白名单。
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 已知「确实不用」的参数：(函数名, 参数名)。每加一条都要写理由。
ALLOW = {
    ("parse_and_check", "raw"),        # 用了 text = raw.strip()，会被算作使用
    ("_verify_merge", "adapter"),      # 见函数体：只用于取 config
}


def used_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            # f-string 里的字段名也算（ast 会把 "…{var}…" 拆成常量 + Name，
            # 但 format 字符串不是；这里只做宽松补充）
            names.add(sub.value)
        elif isinstance(sub, ast.keyword) and sub.arg:
            names.add(sub.arg)
    return names


def check_file(path: Path) -> list[tuple[int, str, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [(exc.lineno or 0, "<syntax>", str(exc))]
    out: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = [a.arg for a in node.args.args + node.args.kwonlyargs]
        args += [a.arg for a in (node.args.posonlyargs or [])]
        names = used_names(node)
        for a in args:
            if a in ("self", "cls") or a.startswith("_"):
                continue
            if a not in names and (node.name, a) not in ALLOW:
                out.append((node.lineno, node.name, a))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=[])
    ap.add_argument("--imports", action="store_true", help="改查未使用的 import")
    args = ap.parse_args()
    targets = [Path(p) for p in args.paths] or [ROOT / "src" / "moneaka", ROOT / "tools"]

    if args.imports:
        report_imports(targets)
        return 0

    total = 0
    for t in targets:
        files = sorted(t.rglob("*.py")) if t.is_dir() else [t]
        for f in files:
            for lineno, fn, param in check_file(f):
                total += 1
                print(f"{f.relative_to(ROOT)}:{lineno}  {fn}()  未使用参数: {param}")
    print(f"\n共 {total} 处")
    return 0


# ---------------------------------------------------------------------------
# 顺带查未使用的 import（同样是"写了但没生效"的一类，会误导读者）
# ---------------------------------------------------------------------------

def unused_imports(path: Path) -> list[tuple[int, str]]:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    body_lines = src.splitlines()
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
            for a in node.names:
                name = (a.asname or a.name).split(".")[0]
                if name in ("*", "annotations"):
                    continue
                # 在**其他行**里出现过就算用了（同一行的 import 不算）
                used = any(name in line for i, line in enumerate(body_lines, 1)
                           if i != node.lineno)
                if not used:
                    out.append((node.lineno, name))
    return out


def report_imports(targets: list[Path]) -> None:
    total = 0
    for t in targets:
        files = sorted(t.rglob("*.py")) if t.is_dir() else [t]
        for f in files:
            for lineno, name in unused_imports(f):
                total += 1
                try:
                    rel = f.relative_to(ROOT)
                except ValueError:
                    rel = f
                print(f"{rel}:{lineno}  未使用的 import: {name}")
    print(f"\n未使用 import 共 {total} 处")

if __name__ == "__main__":
    raise SystemExit(main())
