"""Exact local scorer for Santa 2026 - The Elf Calculator.

    python verify.py submission.csv [puzzles.csv]

Implements the calculator rules from the competition's Data/Evaluation pages.
Exits non-zero (and names the puzzle) if any program is invalid.
"""
import csv
import json
import sys

csv.field_size_limit(10**8)
OPS = {"+", "-", "*", "/"}
SUBSET = "--subset" in sys.argv
if SUBSET:
    sys.argv.remove("--subset")


def load_puzzles(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["id"]] = dict(
                start=json.loads(r["starting_values"]),
                targets=json.loads(r["targets"]),
                limit=int(r["value_limit"]),
                nreg=int(r["registers"]),
            )
    return out


def check_program(pz, prog):
    """Return (ok, message). Checks every rule and every target."""
    if not isinstance(prog, list):
        return False, "program is not a list"
    if len(prog) > 4096:
        return False, "too many instructions"
    nreg, limit = pz["nreg"], pz["limit"]
    reg = [None] * nreg
    for i, v in enumerate(pz["start"]):
        reg[i] = v
    need = set(pz["targets"]) - set(pz["start"])
    for k, ins in enumerate(prog):
        if not (isinstance(ins, list) and len(ins) == 4):
            return False, f"ins {k}: bad shape"
        op, i, j, d = ins
        if op not in OPS:
            return False, f"ins {k}: bad op {op!r}"
        for x in (i, j, d):
            if not (type(x) is int and 0 <= x < nreg):
                return False, f"ins {k}: bad register {x!r}"
        a, b = reg[i], reg[j]
        if a is None or b is None:
            return False, f"ins {k}: undefined source"
        if op == "+":
            r = a + b
        elif op == "-":
            if a <= b:
                return False, f"ins {k}: non-positive subtraction {a}-{b}"
            r = a - b
        elif op == "*":
            r = a * b
        else:
            if a % b:
                return False, f"ins {k}: inexact division {a}/{b}"
            r = a // b
        if r > limit:
            return False, f"ins {k}: value {r} > limit {limit}"
        reg[d] = r
        need.discard(r)
    if need:
        return False, f"{len(need)} targets missing, e.g. {sorted(need)[:3]}"
    return True, "ok"


def score_file(sub_path, puz_path):
    puzzles = load_puzzles(puz_path)
    seen = set()
    total = 0
    with open(sub_path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd)
        if header != ["id", "program"]:
            raise SystemExit(f"bad header {header}")
        for row in rd:
            pid, cell = row
            if pid not in puzzles or pid in seen:
                raise SystemExit(f"unknown or duplicate id {pid}")
            if len(cell.encode("utf-8")) > 131072:
                raise SystemExit(f"{pid}: cell too large")
            seen.add(pid)
            prog = json.loads(cell)
            ok, msg = check_program(puzzles[pid], prog)
            if not ok:
                raise SystemExit(f"INVALID {pid}: {msg}")
            total += len(prog)
    if seen != set(puzzles) and not SUBSET:
        raise SystemExit(f"missing {len(set(puzzles) - seen)} ids")
    return total


if __name__ == "__main__":
    sub = sys.argv[1]
    puz = sys.argv[2] if len(sys.argv) > 2 else "data/puzzles.csv"
    print(score_file(sub, puz))
