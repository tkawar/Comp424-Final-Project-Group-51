"""Play the student agent against an opponent and log every move it makes.

usage: python experiments/collect.py <opponent> <out.jsonl> [--pairs N] [--procs P] [--seed S]
  opponent: "greedy", "random", "self", or the path of an agent file.
Each seed is played twice with the seats swapped. One JSON line per game.
"""
import sys, os, json, time, argparse, importlib.util
from multiprocessing import Pool
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def load(path, tag):
    import store, agents.agent  # noqa: F401
    store.AGENT_REGISTRY.pop("student_agent", None)
    sp = importlib.util.spec_from_file_location("exp_" + tag, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod.StudentAgent


_cache = {}


def make(opp):
    if not _cache:
        _cache["student"] = load(os.path.join(REPO, "agents", "student_agent.py"), "student")
        if opp == "greedy":
            from agents.greedy_agent import GreedyAgent
            _cache["opp"] = GreedyAgent
        elif opp == "random":
            from agents.random_agent import RandomAgent
            _cache["opp"] = RandomAgent
        elif opp == "self":
            _cache["opp"] = _cache["student"]
        else:
            _cache["opp"] = load(os.path.join(REPO, opp) if not os.path.isabs(opp) else opp, "opp")
    return _cache["student"](), _cache["opp"]()


def play(job):
    from helpers import new_game, execute_move, check_endgame, check_move_validity, get_valid_moves
    from constants import STEP_TIME_LIMIT
    opp, seed, seat = job
    a, o = make(opp)
    b, r, obstacle, removed = new_game(seed)
    ags = {seat: a, 3 - seat: o}
    p, ply, moves, strikes = 1, 0, [], 0
    while True:
        over, w, reason = check_endgame(b, r, p, ply)
        if over:
            break
        n = len(get_valid_moves(b, r, p))
        t = time.perf_counter()
        m = ags[p].step(b.copy(), r.copy(), p, 3 - p)
        dt = time.perf_counter() - t
        if not check_move_validity(b, r, p, m):
            raise RuntimeError("illegal move from player %d" % p)
        if p == seat:
            late = dt > STEP_TIME_LIMIT
            strikes += late
            moves.append({"ply": ply, "n": n, "depth": a.last_depth, "nodes": a.last_nodes, "t": round(dt, 4),
                          "stats": list(a.last_stats), "board_move": m.src is not None})
        execute_move(b, r, p, m)
        p = 3 - p
        ply += 1
    return {"opp": opp, "seed": seed, "seat": seat, "obstacle": list(obstacle), "removed": int(removed),
            "result": 1.0 if w == seat else 0.5 if w == 0 else 0.0, "reason": reason, "plies": ply,
            "strikes": int(strikes), "fallbacks": a.fallbacks, "moves": moves}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("opponent"); ap.add_argument("out")
    ap.add_argument("--pairs", type=int, default=20); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    jobs = [(a.opponent, a.seed + i, s) for i in range(a.pairs) for s in (1, 2)]
    with Pool(a.procs) as pool, open(a.out, "w") as f:
        for g in pool.imap_unordered(play, jobs):
            f.write(json.dumps(g) + "\n"); f.flush()
    print("wrote", a.out)
