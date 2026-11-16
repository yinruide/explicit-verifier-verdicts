"""Read-only: detox Exp 2 round-1 in-process scores vs step 6 D_Exp1 scores, bit for bit.

A check of the post-pilot extension (2026-10-31); the result is in
Usage: ``python pipelines/detox_t1_check.py <repo>``.
"""

# ruff: noqa: N806 - the script's original upper-case names, kept inside main()

import json
import struct
import sys
from pathlib import Path


def main(root: Path) -> None:
    ROOT = root
    RUN = ROOT / "runs/exp2_20260928T033025Z"

    def jl(p):
        with open(p, encoding="utf-8") as fh:
            return [json.loads(x) for x in fh if x.strip()]

    def bits(x):
        return struct.pack("<d", float(x)).hex()

    d_exp1 = jl(ROOT / "data/pools/states_d_exp1.jsonl")
    by_x0 = {r["x0_id"]: r for r in d_exp1}
    print("D_Exp1 states", len(d_exp1), "keys", sorted(d_exp1[0].keys()))

    to_score = {r["state_id"]: r["text"] for r in jl(ROOT / "data/step06/to_score.jsonl")}
    x1 = {r["state_id"]: r for r in jl(ROOT / "data/exp2/x1_states.jsonl")}
    print("step06 to_score", len(to_score), "| exp2 x1", len(x1))

    # input text identity
    n_txt_ok = n_txt_bad = 0
    for r in d_exp1:
        sid = r["state_id"]
        a = to_score.get(sid)
        b = x1.get(sid, {}).get("text")
        if a is not None and a == b == r["text"]:
            n_txt_ok += 1
        else:
            n_txt_bad += 1
    print(
        "text identical (states file == step06 input == exp2 x1):",
        n_txt_ok,
        "differ/missing:",
        n_txt_bad,
    )

    # round-1 history text from trajectories (t = 1 entry)
    traj = jl(RUN / "rounds/round_1/trajectories.jsonl")
    n_hist_ok = 0
    for t in traj:
        h1 = [h for h in t["history"] if h["t"] == 1]
        if h1 and h1[0]["x_t"] == by_x0[t["x0_id"]]["text"]:
            n_hist_ok += 1
    print("round_1 trajectories:", len(traj), "t=1 text == D_Exp1 text:", n_hist_ok)

    r1 = jl(RUN / "rounds/round_1/scores.jsonl")
    run_scores = [r for r in jl(RUN / "scores.jsonl") if r["state_id"].endswith("-t1")]
    print("round_1 score rows", len(r1), "| run scores.jsonl t1 rows", len(run_scores))
    same_r1_run = sorted((r["state_id"], r["verifier_id"], bits(r["score"])) for r in r1) == sorted(
        (r["state_id"], r["verifier_id"], bits(r["score"])) for r in run_scores
    )
    print("round_1/scores.jsonl == run scores.jsonl t1 rows (bits):", same_r1_run)

    for cid in ("enc_paradetox", "sg_9b.native"):
        full = {}
        for r in jl(ROOT / f"data/scores/{cid}.jsonl"):
            full[r["state_id"]] = r  # last row per state
        pulled = {r["state_id"]: r for r in jl(ROOT / f"data/step06/pulled/scores/{cid}.jsonl")}
        n = same = same_pulled = 0
        maxdiff = 0.0
        diffs = []
        mv = set()
        ph = set()
        for r in r1:
            if r["verifier_id"] != cid:
                continue
            x0 = r["state_id"].rsplit("-", 2)[0]
            sid = by_x0[x0]["state_id"]
            ref = full[sid]
            n += 1
            mv.add((r["model_version"], ref["model_version"]))
            ph.add((r.get("prompt_hash"), ref.get("prompt_hash")))
            if bits(r["score"]) == bits(ref["score"]):
                same += 1
            else:
                d = abs(r["score"] - ref["score"])
                maxdiff = max(maxdiff, d)
                diffs.append((sid, r["score"], ref["score"]))
            if sid in pulled and bits(pulled[sid]["score"]) == bits(ref["score"]):
                same_pulled += 1
        print(
            f"{cid}: n {n}, bitwise identical {same}, differ {n - same}, max_abs_diff {maxdiff!r};"
            f" step06 table == pulled step06 rows for these states: {same_pulled}/{n}"
        )
        print("   model_version (exp2, step06):", mv)
        print("   prompt_hash   (exp2, step06):", ph)
        for d in diffs[:5]:
            print("   diff:", d)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
