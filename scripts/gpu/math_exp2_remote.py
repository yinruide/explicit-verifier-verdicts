"""Pod side of math Exp 2: the loop with V1, V2 on the GPU and the OpenAI controller in one
process (math entry). The step 6 smoke and the formal Exp 2 are the same command
(2026-10-13: one structure); only the first states, the arms and the thresholds differ.

    python scripts/gpu/math_exp2_remote.py loop --base B --run-id R --smoke     # 30 pool states, A1
    python scripts/gpu/math_exp2_remote.py loop --base B --run-id R             # D_Exp1, A1 and B

What the pod holds (pushed by the RunPod driver's ``start``, not released): the first states
(``data/x1_smoke.jsonl`` / ``x1_formal.jsonl``: state id, question id, problem, solution -- no
label, no reference answer) and the thresholds (``data/tau.json``: the step 6 session tau for
the smoke, ``data/tau_frozen.json``: step 7's frozen tau for the formal run), each with
its sha256. The loop

* scores with the loop's configuration (``configs/math/exp2_loop.json``, 2026-10-13: L_in
  4,096 and the B this session's loop precheck measured, ``out/loopfit/<id>.json``; every other
  setting as in step 2) and refuses to start unless this session's 203-state check passed for
  both V in that configuration (``out/check203/<id>.loop.json``;: every pod session,
  before any formal score); hands the pod's controller key (environment) straight to the SDK
  client (``controller_client``; never printed or written);
* assigns arms as ``MathPipeline.exp2`` does (balanced, seed ``(master, namespace,
  'assignment')``; the smoke has A1 only);
* runs ``pipelines.math_exp2_loop.run_rounds``: each round, all active trajectories are
  scored together, V1 then V2, each loaded for its batched call and freed after it (the two do
  not fit on the GPU together at their fixed shapes); then the controller decides;
* writes every round atomically (``rounds/round_t/COMPLETE.json``), the scores of every text
  (``text_scores.jsonl``, ``text_score_details.jsonl``), the controller ledger, and at the end
  ``loop_complete.json`` (thresholds, profile hash, the check of this session, versions).

A text longer than the loop's L_in for either V ends its trajectory (2026-10-13,
``v_input_overlength``; ``pipelines/math_exp2_loop.py``). A text a V refuses for any other
reason (control-token text) stops the loop before any decision of that round with exit 17 and
``stopped.json``; the round is not written and ``start`` resumes it. E never runs on the pod
(paper App. E: on the laptop, after the whole loop).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu"))

from pipelines.math_exp2_loop import TextScoreStore, ended, run_rounds  # noqa: E402

import math_step06_remote as s6  # noqa: E402
import math_vprobe2_remote as r2  # noqa: E402
from rafr.contracts.schemas import Arm  # noqa: E402
from rafr.core.seeding import derive_seed  # noqa: E402
from rafr.domains.math.config import MathRunConfig  # noqa: E402
from rafr.domains.math.models import ModelInputError, build_verifier  # noqa: E402
from rafr.domains.math.storage import atomic_json, read_json  # noqa: E402
from rafr.experiments.exp2 import Trajectory, balanced_assignment, complete_rounds  # noqa: E402

EXIT = {"no_check": s6.EXIT["no_check"], "input": 17, "key": 18, "assignment": 19}
K = 4


def mode(smoke: bool) -> dict[str, Any]:
    """The names ``MathPipeline.exp2`` uses for the two runs."""
    return {
        "namespace": "math_exp2_smoke" if smoke else "math_exp2",
        "experiment": "exp2_smoke" if smoke else "exp2",
        "arms": (Arm.A1,) if smoke else (Arm.A1, Arm.B),
        "x1": "x1_smoke.jsonl" if smoke else "x1_formal.jsonl",
        "tau": "tau.json" if smoke else "tau_frozen.json",
        "phase": "smoke" if smoke else "experiment",
    }


def run_dir(base: Path, run_id: str, smoke: bool) -> Path:
    return Path(base) / "out" / mode(smoke)["experiment"] / run_id


def read_checked_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        r2.die(f"{path} not on the pod: `start` pushes it")
    recorded = path.with_name(path.name + ".sha256").read_text(encoding="utf-8").split()[0]
    if r2.sha256_file(path) != recorded:
        r2.die(f"{path} changed since its sha256 was written")
    return json.loads(path.read_text(encoding="utf-8")), recorded


def controller_client(base: Path, out: Path) -> Any:
    """The controller's client on the pod. The key is the pod's ``OPENAI_API_KEY`` environment
    variable (the authors set it in the RunPod UI): its presence is checked and its value goes
    straight into the SDK client, never printed, logged or written -- as detox's pod loop does
    (``rafr.core.runtime.build_pod_client``), since the provider's own default reads ``.env``
    only and the shipped code has none."""
    from rafr.domains.math.generation import frozen_models
    from rafr.llm.cache import LLMCache
    from rafr.llm.client import LLMClient
    from rafr.llm.ledger import Ledger
    from rafr.llm.providers.openai import OpenAIProvider

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set on the pod (the authors set it in the RunPod UI)",
            file=sys.stderr,
        )
        raise SystemExit(EXIT["key"])
    network = frozen_models()["controller"].get("network", {})
    return LLMClient(
        providers={
            "openai": OpenAIProvider(
                api_key=os.environ["OPENAI_API_KEY"],
                max_retries=int(network.get("max_retries", 0)),
                timeout_s=float(network.get("timeout_s", 120)),
            )
        },
        cache=LLMCache(Path(base) / "cache" / "llm"),
        ledger=Ledger(out / "ledger.jsonl"),
        transient_log=out / "transient_retries.jsonl",
    )


def release_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - the pod has torch
        pass


def loop(
    base: Path,
    run_id: str,
    *,
    smoke: bool,
    client: Any = None,
    formal_config: Path | None = None,
    make_verifier: Any = None,
    loop_config: Path | None = None,
) -> dict[str, Any]:
    """``client``, ``make_verifier`` and the configuration paths are injected by the tests."""
    from rafr.domains.math.generation import build_controller_config
    from rafr.formatter.math_feedback import format_observation

    m = mode(smoke)
    paths = s6.layout(base)
    config = MathRunConfig.load(Path(formal_config or s6.FORMAL_CONFIG))
    config.assert_ready(m["phase"])
    candidates = []
    for spec in config.verifiers:
        cid = next(c["candidate_id"] for c in r2.load_cfg()["candidates"] if c["spec"] == spec)
        candidates.append(s6.load_v(cid, formal_config))
    loop_settings = s6.load_loop(loop_config)
    specs = [s6.loop_spec(paths, c, loop_settings) for c in candidates]
    checks, shapes = {}, {}
    for c, spec in zip(candidates, specs, strict=True):
        cid = c["candidate_id"]
        record = s6.require_check(paths, cid, build_verifier(spec).config_hash, loop=True)
        checks[cid] = record["criterion"]
        shapes[cid] = {key: spec.settings[key] for key in s6.LOOP_ADJUSTABLE}
    x1_path = Path(base) / "data" / m["x1"]
    rows = r2.read_rows(x1_path)
    tau_record, tau_sha = read_checked_json(Path(base) / "data" / m["tau"])
    taus = tuple(
        float(tau_record["taus"][tau_record["candidates"].index(c["candidate_id"])])
        for c in candidates
    )
    out = run_dir(base, run_id, smoke)
    out.mkdir(parents=True, exist_ok=True)
    assignment = balanced_assignment(
        [r["x0_id"] for r in rows],
        m["arms"],
        derive_seed(config.seed, m["namespace"], "assignment"),
    )
    assigned = {key: arm.value for key, arm in assignment.items()}
    if (out / "assignment.json").exists():
        if read_json(out / "assignment.json") != assigned:
            r2.die("assignment.json differs from the seeded assignment of these first states")
    else:
        atomic_json(out / "assignment.json", assigned)
    initial = [Trajectory(r["x0_id"], r["x0"], assigned[r["x0_id"]], 1, r["text"]) for r in rows]
    if client is None:
        client = controller_client(base, out)

    def build(i: int) -> Any:
        verifier = (make_verifier or build_verifier)(specs[i])
        verifier.load()
        return verifier

    def batch_note(i: int, n: int, seconds: float) -> None:
        print(f"[exp2] V{i + 1} scored {n} text(s) in {seconds / 60:.1f} min", flush=True)

    store = TextScoreStore(
        out,
        [(spec.model_id, spec.revision) for spec in specs],
        build,
        release=release_gpu,
        on_batch=batch_note,
    )

    def round_note(t: int) -> None:
        print(f"[exp2] round {t} complete", flush=True)

    try:
        trajectories = run_rounds(
            out,
            initial,
            taus=taus,
            verifier_ids=(specs[0].model_id, specs[1].model_id),
            score_texts=store,
            client=client,
            cfg=build_controller_config(config.seed, K),
            feedback=format_observation,
            namespace=m["namespace"],
            experiment=m["experiment"],
            run_id=run_id,
            seed=config.seed,
            smoke=smoke,
            on_round=round_note,
        )
    except ModelInputError as exc:
        atomic_json(
            out / "stopped.json",
            {
                "reason": "a V refused a loop text for another reason than its length "
                "(control-token text); the round was not written",
                "error": str(exc),
                "complete_rounds": complete_rounds(out, K),
                "at": r2.utc_now(),
            },
        )
        print(f"[exp2] stopped: {exc}", file=sys.stderr)
        raise SystemExit(EXIT["input"]) from exc
    record = {
        "k": K,
        "rounds": complete_rounds(out, K),
        "n_trajectories": len(trajectories),
        "SMOKE": smoke,
        "execution_mode": config.execution_mode,
        "thresholds": list(taus),
        "verifiers": [spec.model_id for spec in specs],
        "profile_hash": config.profile_hash(),
        "x1_sha256": r2.sha256_file(x1_path),
        "tau_sha256": tau_sha,
        "check203": checks,
        "loop_configuration": {
            "settings": shapes,
            "config_hashes": {
                c["candidate_id"]: build_verifier(spec).config_hash
                for c, spec in zip(candidates, specs, strict=True)
            },
            "source": "configs/math/exp2_loop.json and this session's loop precheck (the authors "
            "2026-10-13)",
        },
        "n_v_input_overlength": sum(ended(tr) for tr in trajectories),
        "scoring": "each round, all active trajectories together in fixed-shape batches, V1 then "
        "V2, each loaded for its call (II.10 math entry, 2026-10-12)",
        "environment": r2.environment(),
        "finished_at": r2.utc_now(),
    }
    atomic_json(out / "loop_complete.json", record)
    print(
        f"[exp2] loop complete: {len(trajectories)} trajectories, rounds {record['rounds']}",
        flush=True,
    )
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("loop",))
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    loop(args.base, args.run_id, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
