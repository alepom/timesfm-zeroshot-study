"""Check every number written by hand in the paper against the study's outputs.

    python -m tfmstudy.check_numbers [--all]

Numbers in `paper/generated/` come from the code. This script reads the hand-written `paper/sections/*.tex`, extracts
each numeric literal (decimals, percentages, counts of 10 or more) and looks for it among the values of the analysis
outputs: `paper/generated/*.tex`, `reports/*.json`, `reports/*.md`, `results/analysis/**/*.json`, in the forms the text
uses (3, 2 or 1 decimals, ratio as a percent change). Unmatched literals are printed with their context, to be checked
by hand; `paper/number_allowlist.txt` lists the ones checked and accepted (design constants, citations of other work),
one `literal | reason` per line. Exit code 1 if any literal is unmatched and not allowed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ROOT / "paper" / "sections"
ALLOW = ROOT / "paper" / "number_allowlist.txt"

NUM = re.compile(r"(?<![\w.:\\-])(\d{1,3}(?:\{,\}\d{3})+|\d+(?:\.\d+)?)(\\?%)?(?![\w]|:\d)")  # not clock times
SKIP_CMDS = re.compile(r"\\(?:cite[tp]?|ref|label|url|input|includegraphics|eqref|autoref|href)\*?(?:\[[^\]]*\])*\{[^}]*\}")
SKIP_ENVS = re.compile(r"\\begin\{(tikzpicture|axis)\}.*?\\end\{\1\}", re.S)


def _walk(obj, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, out)


def _forms(x: float) -> set[str]:
    """Strings under which a value may appear in the text."""
    f = {f"{x:.3f}", f"{x:.2f}", f"{x:.1f}", f"{x:.0f}", f"{x:.4f}"}
    if 0 < abs(x) < 20:  # ratios and shares as percentages
        for p in (x * 100, (1 - x) * 100, (x - 1) * 100):
            f |= {f"{p:.0f}%", f"{p:.1f}%", f"{p:.2f}%", f"{abs(p):.0f}%", f"{abs(p):.1f}%", f"{abs(p):.2f}%"}
    if abs(x) >= 100:
        f.add(f"{x:.0f}")
    return {s.lstrip("-") if s.startswith("-") else s for s in f} | f


def known_values() -> dict[str, str]:
    """value string -> first source file where it occurs."""
    known: dict[str, str] = {}
    files = (sorted((ROOT / "paper" / "generated").glob("*.tex")) + sorted((ROOT / "reports").glob("*.json"))
             + sorted((ROOT / "reports").glob("*.md")) + sorted((ROOT / "results" / "analysis").rglob("*.json"))
             + sorted((ROOT / "reports" / "seals").glob("*.json")))
    for f in files:
        text = f.read_text(errors="ignore")
        vals: list[float] = []
        if f.suffix == ".json":
            _walk(json.loads(text), vals)
        else:
            text = text.replace("{,}", "")
            for m in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", text):
                known.setdefault(m.group().lstrip("-").replace(",", ""), f.relative_to(ROOT).as_posix())  # as written
                try:
                    vals.append(float(m.group().replace(",", "")))
                except ValueError:
                    pass
            for m in re.finditer(r"(\d+(?:\.\d+)?)\\?%", text):  # percentages written as such
                known.setdefault(m.group(1) + "%", f.name)
        for v in vals:
            for s in _forms(v) | ({x + "%" for x in _forms(v) if "%" not in x} if f.name == "paper_facts.json" else set()):
                known.setdefault(s, f.relative_to(ROOT).as_posix())  # paper_facts.json stores percents as numbers
    return known


def literals(tex: str) -> list[tuple[str, str]]:
    """(literal, context) for each number written in the text."""
    body = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())  # LaTeX comments
    body = SKIP_ENVS.sub(" ", SKIP_CMDS.sub(" ", body))
    out = []
    for m in NUM.finditer(body):
        raw, pct = m.group(1), m.group(2)
        val = raw.replace("{,}", "")
        if not pct and "." not in val and int(val) < 10:
            continue  # small counts ("two issues") are read, not checked
        if not pct and "." not in val and 1900 <= int(val) <= 2100:
            continue  # years
        lit = val + ("%" if pct else "")
        ctx = body[max(0, m.start() - 50):m.end() + 30].replace("\n", " ")
        out.append((lit, ctx))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="also print matched literals with their source")
    args = parser.parse_args()
    known = known_values()
    allowed = {}
    if ALLOW.exists():
        for line in ALLOW.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                lit, _, reason = line.partition("|")
                allowed[lit.strip()] = reason.strip()
    bad = 0
    for f in sorted(SECTIONS.glob("*.tex")):
        if f.name == "figure_style.tex":  # colour definitions, no reported numbers
            continue
        for lit, ctx in literals(f.read_text()):
            src = known.get(lit)
            if src and args.all:
                print(f"ok    {f.name:28s} {lit:>8s}  <- {src}")
            elif not src and lit in allowed:
                if args.all:
                    print(f"allow {f.name:28s} {lit:>8s}  ({allowed[lit]})")
            elif not src:
                bad += 1
                print(f"CHECK {f.name:28s} {lit:>8s}  ...{ctx}...")
    print(f"{bad} literal(s) to check by hand")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
