"""Turn the logs of collect.py into the tables of experiments/RESULTS.md.

usage: python experiments/summarize.py experiments/data/*.jsonl > experiments/RESULTS.md
"""
import sys, json, math
from collections import defaultdict

games = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip()]
SIZE = "SML"


def wdl(gs):
    w = sum(g["result"] == 1 for g in gs); d = sum(g["result"] == 0.5 for g in gs); l = len(gs) - w - d
    pct = 100 * (w + d / 2) / len(gs) if gs else 0
    return f"{len(gs)} | {w} | {d} | {l} | {pct:.1f}%"


def cell_class(rc):
    r, c = rc
    if (r in (0, 3)) and (c in (0, 3)):
        return "corner"
    if r in (1, 2) and c in (1, 2):
        return "centre"
    return "edge"


def table(title, key, order=None):
    print(f"\n### {title}\n")
    opps = sorted({g["opp"] for g in games})
    print("| " + title.split(" by ")[-1] + " | opponent | games | W | D | L | score |\n|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for g in games:
        groups[(key(g), g["opp"])].append(g)
    keys = order or sorted({k for k, _ in groups})
    for k in keys:
        for o in opps:
            if groups[(k, o)]:
                print(f"| {k} | {o} | {wdl(groups[(k, o)])} |")


print("# Measurements for the report\n")
print("Produced by `experiments/collect.py` and `experiments/summarize.py`. The student agent plays every game")
print("on the 2-second clock; each seed (obstacle and removed size) is played twice with the seats swapped.\n")
print("## Results\n")
print("| opponent | games | W | D | L | score | mean plies | late moves | fallback moves |\n|---|---|---|---|---|---|---|---|---|")
for o in sorted({g["opp"] for g in games}):
    gs = [g for g in games if g["opp"] == o]
    print(f"| {o} | {wdl(gs)} | {sum(g['plies'] for g in gs) / len(gs):.0f} | {sum(g['strikes'] for g in gs)} | {sum(g['fallbacks'] for g in gs)} |")

table("Results by seat", lambda g: "first (P1)" if g["seat"] == 1 else "second (P2)")
table("Results by obstacle position", lambda g: cell_class(g["obstacle"]), ["corner", "edge", "centre"])
table("Results by removed size", lambda g: SIZE[g["removed"]], ["S", "M", "L"])

print("\n## Search by phase of the game\n")
print("Every move the student agent searched, over all games. Branching is the number of legal moves at the root;")
print("the effective branching factor is nodes^(1/depth); a full minimax tree of the same depth would have")
print("branching^depth nodes.\n")
print("| plies | moves | legal moves | depth | nodes | effective branching | minimax nodes / searched nodes | table cutoffs per 1000 nodes | cutoffs on first move |")
print("|---|---|---|---|---|---|---|---|---|")
buckets = [(0, 7), (8, 15), (16, 31), (32, 63), (64, 99)]
allm = [m for g in games for m in g["moves"]]
for lo, hi in buckets:
    ms = [m for m in allm if lo <= m["ply"] <= hi and m["depth"] > 0 and m["nodes"] > 2000]
    if not ms:
        continue
    n = sum(m["n"] for m in ms) / len(ms)
    d = sum(m["depth"] for m in ms) / len(ms)
    nodes = sum(m["nodes"] for m in ms) / len(ms)
    ebf = sum(m["nodes"] ** (1 / m["depth"]) for m in ms) / len(ms)
    ratio = sum(m["depth"] * math.log10(m["n"]) - math.log10(m["nodes"]) for m in ms) / len(ms)
    ttc = 1000 * sum(m["stats"][0] for m in ms) / sum(m["nodes"] for m in ms)
    fc = 100 * sum(m["stats"][2] for m in ms) / max(1, sum(m["stats"][1] for m in ms))
    print(f"| {lo}-{hi} | {len(ms)} | {n:.0f} | {d:.1f} | {nodes:,.0f} | {ebf:.1f} | 10^{ratio:.1f} | {ttc:.0f} | {fc:.0f}% |")

print("\n### Depth by obstacle position and removed size\n")
print("| group | moves | legal moves | depth |\n|---|---|---|---|")
for name, key in (("obstacle", lambda g: cell_class(g["obstacle"])), ("removed", lambda g: SIZE[g["removed"]])):
    groups = defaultdict(list)
    for g in games:
        groups[key(g)] += [m for m in g["moves"] if m["nodes"] > 2000]
    for k in sorted(groups):
        ms = groups[k]
        if ms:
            print(f"| {name} {k} | {len(ms)} | {sum(m['n'] for m in ms) / len(ms):.0f} | {sum(m['depth'] for m in ms) / len(ms):.1f} |")

print("\n## Timing\n")
ts = sorted(m["t"] for m in allm)
if ts:
    print(f"- moves timed: {len(ts)}")
    print(f"- mean {sum(ts) / len(ts):.3f} s, median {ts[len(ts) // 2]:.3f} s, 99th percentile {ts[int(len(ts) * 0.99)]:.3f} s, max {ts[-1]:.3f} s")
    print(f"- moves over the 2.0 s limit: {sum(t > 2.0 for t in ts)}")
    short = sum(m["nodes"] <= 2000 for m in allm)
    print(f"- moves answered at once (a forced win or loss was found): {short}")
    print(f"- board moves (as opposed to placements): {100 * sum(m['board_move'] for m in allm) / len(allm):.0f}% of moves")
