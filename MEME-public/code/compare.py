#!/usr/bin/env python
"""Aggregate after-phase per-task scores for the matched AutoDream vs OmniMemory comparison
from MeME judge outputs, and print the table. Run after run_compare.sh.

Both systems must have been run through the SAME harness/model (see run_compare.sh):
run_agent (single-strategy shared prompt) + judge, --model claude-code/sonnet, and
OMNI_CONSTRUCTION_MODEL=claude-code/sonnet — so the only variable is the memory architecture.
"""
import glob, json, os

TASKS = ["ER", "Agg", "Tr", "Del", "Cas", "Abs"]
HERE = os.path.dirname(os.path.abspath(__file__))
SYS = {"AutoDream": "output/auto_memory_dreaming/judge",
       "OmniMemory": "output/omni_memory/judge"}


def agg(judge_dir):
    a = {t: [0, 0] for t in TASKS}; tot = [0, 0]; n = 0
    for f in glob.glob(os.path.join(judge_dir, "eval_*.json")):
        d = json.load(open(f)); n += 1
        for ans in d.get("after_answers", []):
            t = ans["task_type"].split()[0]
            ok = 1 if ans.get("u_pass") else 0
            if t in a:
                a[t][0] += ok; a[t][1] += 1
            tot[0] += ok; tot[1] += 1
    return a, tot, n


def pct(x):
    return f"{x[0]}/{x[1]}" + (f" ({round(100 * x[0] / x[1])}%)" if x[1] else "")


def main():
    res = {name: agg(os.path.join(HERE, d)) for name, d in SYS.items()}
    names = list(SYS)
    print("Matched comparison (after-phase): construction + answer = claude-code/sonnet, "
          "run_agent harness, full filler32k.\n")
    print(f"{'task':5} " + " ".join(f"{n:>18}" for n in names))
    for t in TASKS:
        print(f"{t:5} " + " ".join(f"{pct(res[n][0][t]):>18}" for n in names))
    print(f"{'TOTAL':5} " + " ".join(f"{pct(res[n][1]):>18}" for n in names))
    print()
    for n in names:
        print(f"  {n}: {res[n][2]} episodes judged")


if __name__ == "__main__":
    main()
