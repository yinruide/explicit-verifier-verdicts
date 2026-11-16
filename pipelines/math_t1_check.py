"""Read-only: math Exp 2 round-1 in-process scores vs step 6 D_Exp1 scores, bit for bit.

Stop point A of the post-pilot extension (2026-10-31), kept as run for the authors, who takes math
over; the result is in docs/progress.md. Usage: ``python pipelines/math_t1_check.py <repo>``.

Also compares the per-item scoring records (input assembly hashes, generated / forced token ids,
verdict logits) and the fixed-shape configuration of the two stages.
"""

# ruff: noqa: N806 - the script's original upper-case names, kept inside main()

import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(root: Path) -> None:
    ROOT = root
    EXP2 = ROOT / "runs/math_formal_n9000/exp2"
    S6 = ROOT / "data/math/formal_pod/remote/d_exp1"
    CIDS = {
        "glmz1_9b": "zai-org/GLM-Z1-9B-0414",
        "r1_llama_8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    }

    def jl(p):
        with open(p, encoding="utf-8") as fh:
            return [json.loads(x) for x in fh if x.strip()]

    def bits(x):
        return struct.pack("<d", float(x)).hex()

    inputs = {r["state_id"]: r for r in jl(ROOT / "data/math/step06/d_exp1_inputs.jsonl")}
    x1 = {r["state_id"]: r for r in jl(ROOT / "data/math/step06/x1_formal.jsonl")}
    print("step06 d_exp1 inputs", len(inputs), "| exp2 x1", len(x1))
    same_text = sum(
        1
        for sid, r in inputs.items()
        if sid in x1 and r["problem"] == x1[sid]["x0"] and r["solution"] == x1[sid]["text"]
    )
    print("problem == x0 and solution == x1 text:", same_text, "/", len(inputs))

    r1 = jl(EXP2 / "rounds/round_1/scores.jsonl")
    remote_r1 = jl(
        ROOT / "data/math/exp2_pod/remote/exp2/exp2_20261016/rounds/round_1/scores.jsonl"
    )
    print(
        "round_1 score rows",
        len(r1),
        "| remote copy identical (bits):",
        sorted((r["state_id"], r["verifier_id"], bits(r["score"])) for r in r1)
        == sorted((r["state_id"], r["verifier_id"], bits(r["score"])) for r in remote_r1),
    )
    run_table = defaultdict(dict)
    for r in jl(ROOT / "runs/math_formal_n9000/scores.jsonl"):
        run_table[r["verifier_id"]][r["state_id"]] = r  # last row per (V, state)

    det2_by_v = defaultdict(list)
    for r in jl(EXP2 / "text_score_details.jsonl"):
        det2_by_v[r["verifier_id"]].append(r)

    for cid, vid in CIDS.items():
        s6 = {r["state_id"]: r for r in jl(S6 / f"scores/{cid}.jsonl")}
        d6 = {r["state_id"]: r for r in jl(S6 / f"details/{cid}.jsonl")}
        n = same = same_run = 0
        maxdiff = 0.0
        flips = 0
        diffs = []
        tau = {"glmz1_9b": 0.23389257597884028, "r1_llama_8b": 0.0012849328596507676}[cid]
        for r in r1:
            if r["verifier_id"] != vid:
                continue
            qid = r["state_id"].rsplit("-", 2)[0]
            sid = qid + "-g1"
            ref = s6[sid]
            n += 1
            if bits(r["score"]) == bits(ref["score"]):
                same += 1
            else:
                maxdiff = max(maxdiff, abs(r["score"] - ref["score"]))
                flips += int((r["score"] >= tau) != (ref["score"] >= tau))
                diffs.append((sid, r["score"], ref["score"]))
            same_run += int(bits(run_table[vid][sid]["score"]) == bits(ref["score"]))
        print(
            f"\n{cid}: round-1 scores n {n}, bitwise identical {same}, differ {n - same}, "
            f"max_abs_diff {maxdiff!r}, flips at tau {flips}; "
            f"run scores.jsonl == step06 table: {same_run}/{n}"
        )
        for d in diffs[:5]:
            print("   diff:", d)

        # per-item records: match Exp 2 details to step 6 details by input_hash
        by_input = defaultdict(list)
        for r in det2_by_v[vid]:
            by_input[r["input_hash"]].append(r)
        fields_item = (
            "serialized_input_hash",
            "token_ids_hash",
            "prompt_sha256",
            "verdict_prompt_sha256",
            "chat_template_hash",
            "input_tokens",
            "generated_token_ids",
            "forced_token_ids",
            "verdict_logits_positive_negative",
            "readouts",
            "stop_reason",
            "truncated",
            "generated_tokens",
        )
        fields_cfg = (
            "fixed_shape",
            "configured_dtype",
            "actual_dtype",
            "attn_implementation",
            "revision",
            "tokenizer_revision",
            "adapter_version",
            "implementation",
            "config_hash",
            "padding_side",
            "pad_token_id",
            "library_versions",
            "score_definition",
            "softmax_arithmetic",
            "verdict_token_ids_positive_negative",
            "execution_mode",
        )
        matched = 0
        multi = 0
        item_diff = Counter()
        cfg_vals = defaultdict(set)
        first_div = []
        for _sid, a in d6.items():
            cands = by_input.get(a["input_hash"], [])
            if not cands:
                continue
            matched += 1
            multi += int(len(cands) > 1)
            b = cands[0]
            for f in fields_item:
                if a.get(f) != b.get(f):
                    item_diff[f] += 1
            if a.get("generated_token_ids") != b.get("generated_token_ids"):
                ga, gb = a["generated_token_ids"], b["generated_token_ids"]
                k = next(
                    (i for i, (x, y) in enumerate(zip(ga, gb, strict=False)) if x != y),
                    min(len(ga), len(gb)),
                )
                first_div.append(k)
            for f in fields_cfg:
                cfg_vals[f].add(
                    (json.dumps(a.get(f), sort_keys=True), json.dumps(b.get(f), sort_keys=True))
                )
        print(
            f"   details: step6 {len(d6)}, matched in exp2 by input_hash {matched} "
            f"(several rows: {multi})"
        )
        print("   per-item fields that differ (count of items):", dict(item_diff) or "none")
        if first_div:
            print(
                "   first diverging generated-token position: min",
                min(first_div),
                "max",
                max(first_div),
            )
        for f in fields_cfg:
            vals = cfg_vals[f]
            status = "same" if all(x == y for x, y in vals) else "DIFFERENT"
            shown = sorted(vals)[:2]
            value = shown if status != "same" or len(vals) > 1 else shown[0][0]
            print(f"   cfg {f}: {status} {value}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
