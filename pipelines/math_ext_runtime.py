"""Local extension assembly. Providers are constructed last, never during import/preflight.

Public transport is the only input to a loop. Private references are opened only by
the local post-loop judge/ceiling, with an independent byte binding and question binding.
"""

from __future__ import annotations

import json
from pathlib import Path

from rafr.core import paths

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value)
    return paths.resolve(path)


def checked(path, sha):
    from pipelines.math_ext_run import SHA
    from rafr.domains.math.storage import file_hash

    path = resolve(path)
    if not isinstance(sha, str) or not SHA.fullmatch(sha):
        raise ValueError("an independently supplied SHA256 is required")
    if path.is_symlink() or not path.is_file() or file_hash(path) != sha:
        raise ValueError(f"input hash mismatch/missing: {path}")
    return path


def load_extension(path, source, *, operation, arms=(), live=False):
    from pipelines.math_ext_execution import check_extension_config
    from pipelines.math_ext_run import _config, load_source

    cfg = _config(resolve(path))
    if live:
        check_extension_config(cfg)
        bound = load_source(resolve(cfg["source_manifest"]))
        if bound.provenance() != source.provenance():
            raise ValueError("extension source_manifest differs from CLI source")
        for kind in ("exp1", "exp2"):
            spec = cfg[kind]["e_incorrect"]
            for filename, sha in spec["original_sha256"].items():
                actual = checked(resolve(spec["original_run"]) / filename, sha)
                if actual.resolve() != (source.root / kind / filename).resolve():
                    raise ValueError("original run binding differs from source")
        spec = cfg["exp2"]["e_incorrect"]
        if (
            checked(spec["x1_file"], spec["x1_sha256"]).resolve()
            != (source.root / "states.jsonl").resolve()
        ):
            raise ValueError("x1 source binding differs")
        scores = spec["t1_scores"]
        if (
            checked(scores["source"], scores["source_sha256"]).resolve()
            != (source.root / "exp2/rounds/round_1/scores.jsonl").resolve()
        ):
            raise ValueError("t1 source binding differs")
    return cfg


def cohort_binding(source, ext, *, required, live):
    from pipelines.math_ext_data import same_bound_source
    from rafr.contracts.schemas import MathLabel, Split
    from rafr.core.seeding import component_seed
    from rafr.data.labels import LabelPurpose
    from rafr.experiments.ext_cohort import draw_cohort, read_cohort

    if not required:
        return None, None, []
    spec = ext["cohorts"]["e_correct"]
    path = checked(spec["file"], spec["file_sha256"])
    header, ids = read_cohort(path, spec["file_sha256"])
    if live:
        if header.get("scientifically_frozen") is not True or not same_bound_source(
            header.get("source"), source
        ):
            raise ValueError("correct cohort must be frozen against this source")
        candidates = [
            r.state_id
            for r in source.labels(Split.D_GATE, LabelPurpose.ECORRECT_COHORT)
            if r.label is MathLabel.CORRECT
        ]
        expected = draw_cohort(
            candidates, 300, component_seed(source.master_seed, "ext_ecorrect_cohort")
        )
        if (
            ids != expected
            or header.get("target_n") != 300
            or header.get("seed") != component_seed(source.master_seed, "ext_ecorrect_cohort")
            or header.get("n_candidates") != len(candidates)
            or header.get("domain") != "math"
            or header.get("split") != "d_gate"
        ):
            raise ValueError("correct cohort differs from the frozen seeded draw")
    return path, spec["file_sha256"], ids


def selected(source, ext, *, cohort, smoke, live):
    from rafr.contracts.schemas import MathLabel, Split
    from rafr.core.seeding import component_seed
    from rafr.data.labels import LabelPurpose
    from rafr.experiments.ext_cohort import draw_smoke

    path, sha, formal = cohort_binding(source, ext, required=cohort == "e_correct", live=live)
    split = (
        Split.D_GATE if cohort == "e_correct" else (Split.UNLABELED_POOL if smoke else Split.D_EXP1)
    )
    states = source.states(split)
    if smoke:
        purpose = LabelPurpose.EXT_SMOKE if cohort == "e_correct" else LabelPurpose.DRY_RUN
        wanted = MathLabel.CORRECT if cohort == "e_correct" else MathLabel.INCORRECT
        candidates = [
            r.state_id
            for r in source.labels(split, purpose)
            if r.label is wanted and r.state_id not in formal
        ]
        ids = set(
            draw_smoke(candidates, smoke, component_seed(source.master_seed, "ext_smoke"), cohort)
        )
        states = [s for s in states if s.state_id in ids]
    elif cohort == "e_correct":
        by_id = {s.state_id: s for s in states}
        states = [by_id[sid] for sid in formal]
    if not states or (live and not smoke and cohort == "e_incorrect" and len(states) != 300):
        raise ValueError("formal incorrect cohort requires all 300 original starts")
    return states, path, sha


def calibrator_for(source, ext, arms, *, live):
    from pipelines.math_ext_data import same_bound_source
    from rafr.calibration.apply import Calibrator

    if not any(a.value in {"B_pihat", "B_pihat_mismatched"} for a in arms):
        return None
    spec = ext["calibration"]
    path = checked(spec["frozen_file"], spec["frozen_sha256"])
    cal = Calibrator.load(path, expected_sha256=spec["frozen_sha256"])
    cal.check_verifiers(*source.verifier_ids)
    if cal.record.get("domain") != "math":
        raise ValueError("math-domain calibrator required")
    provenance = cal.record.get("fit_data", {})
    if live and (
        provenance.get("scientifically_frozen") is not True
        or not same_bound_source(provenance, source)
    ):
        raise ValueError("calibrator is not frozen against this source")
    return cal


def client_for(out, *, live, provider_factory=None):
    from rafr.core.runtime import build_client
    from rafr.domains.math.generation import frozen_models

    models = frozen_models()
    if provider_factory is not None:
        provider = provider_factory()
    elif live:
        net = models["controller"]["network"]

        class DeferredOpenAI:
            """Do not read a key/create SDK until the engine finishes its input checks."""

            name = "openai"
            _provider = None

            def complete(self, request):
                if self._provider is None:
                    from rafr.llm.providers.openai import OpenAIProvider

                    self._provider = OpenAIProvider(
                        max_retries=int(net["max_retries"]), timeout_s=float(net["timeout_s"])
                    )
                return self._provider.complete(request)

        provider = DeferredOpenAI()
    else:
        from rafr.llm.providers.fake import FakeProvider

        provider = FakeProvider(lambda req: '{"action":"accept","revised_text":null}')
    return build_client(
        models, run_dir=out, providers={"openai": provider}, cache_dir=out / "cache"
    )


def private_references(path, sha, states):
    """External E-only JSONL, exact schema. Hash supplied independently by operator.

    Each row binds question id, question fingerprint and answer. This never guesses
    joins from answer text, metadata, or source order; public bundle receives none.
    """
    from rafr.domains.math.dataset import content_hash, problem_fingerprint

    rows = [
        json.loads(line)
        for line in checked(path, sha).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = {s.x0_id: s.x0 for s in states}
    refs = {}
    fields = {"question_id", "problem_sha256", "reference_answer", "binding_sha256"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("private reference rows require exactly four binding fields")
        qid = row["question_id"]
        if qid in refs or qid not in wanted:
            raise ValueError("private references contain duplicate/unselected question")
        answer = row["reference_answer"]
        fp = problem_fingerprint(wanted[qid])
        if (
            not isinstance(answer, str)
            or not answer.strip()
            or row["problem_sha256"] != fp
            or row["binding_sha256"] != content_hash([qid, fp, answer])
        ):
            raise ValueError("private reference question/answer binding mismatch")
        refs[qid] = answer
    if set(refs) != set(wanted):
        raise ValueError("private references must exactly cover selected questions")
    return refs


def bundle_references(directory, source, states):
    """Read original E-only bundle without rereading/downloading the upstream dataset."""
    from rafr.domains.math.dataset import PrivateReference, content_hash, problem_fingerprint

    root = resolve(directory)
    expected = source.read_bound_json("snapshot.json")["math_identity"]["bundle"]
    manifest = json.loads(checked(root / "manifest.json", expected).read_text(encoding="utf-8"))
    names = {
        "public_questions.jsonl",
        "references.private.jsonl",
        "exclusions.jsonl",
        "eligibility.jsonl",
    }
    if manifest.get("domain") != "math" or set(manifest.get("files", {})) != names:
        raise ValueError("original private dataset manifest schema differs")
    for name, sha in manifest["files"].items():
        checked(root / name, sha)
    questions = {}
    for line in (root / "public_questions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if set(row) != {"question_id", "problem"} or row["question_id"] in questions:
            raise ValueError("original public question schema/identity mismatch")
        questions[row["question_id"]] = row["problem"]
    refs = {}
    for line in (root / "references.private.jsonl").read_text(encoding="utf-8").splitlines():
        r = PrivateReference(**json.loads(line))
        if r.question_id not in questions or r.question_id in refs:
            raise ValueError("original private reference identity mismatch")
        fp = problem_fingerprint(questions[r.question_id])
        if r.problem_sha256 != fp or r.binding_sha256 != content_hash(
            [r.question_id, fp, r.reference_answer]
        ):
            raise ValueError("original private reference binding mismatch")
        refs[r.question_id] = r.reference_answer
    if set(refs) != set(questions) or any(questions.get(s.x0_id) != s.x0 for s in states):
        raise ValueError("original bundle questions differ from starting states")
    return {s.x0_id: refs[s.x0_id] for s in states}


def bundle_setup(args, source, ext, *, out):
    from pipelines.math_ext_exp2 import setup_from_public_bundle
    from pipelines.math_ext_transport import import_public_bundle
    from rafr.domains.math.storage import digest
    from rafr.formatter.math_extended import math_wording

    bundle = import_public_bundle(
        resolve(args.bundle),
        expected_manifest_sha256=args.bundle_sha256,
        expected_source_identity_sha256=digest(source.provenance()),
        expected_verifier_ids=source.verifier_ids,
        expected_verifier_versions=source.verifier_versions,
        expected_source_mode=source.execution_mode,
    )
    if bundle.manifest["cohort"] != args.cohort or bundle.manifest["smoke"] != bool(args.smoke):
        raise ValueError("bundle cohort/smoke differs from requested run")
    states, _, _ = selected(
        source, ext, cohort=args.cohort, smoke=args.smoke, live=args.stage == "live"
    )
    if [(s.state_id, s.x0_id, s.x0, s.text) for s in states] != [
        (s.state_id, s.x0_id, s.x0, s.text) for s in bundle.states
    ]:
        raise ValueError("public starts differ from selected bound source")
    arms = set(a for values in bundle.plan.values() for a in values)
    cal = calibrator_for(source, ext, arms, live=args.stage == "live")

    def no_scores(*_):
        raise ValueError("local runner cannot invoke verifier; use pod runner")

    setup = setup_from_public_bundle(
        bundle,
        run_id=args.run_id,
        run_dir=out,
        taus=source.taus,
        score_texts=no_scores,
        wording=math_wording(),
        extension_config=ext,
        channel_config=source.read_bound_json("fit.json")["channel"],
        calibrator=cal,
        confirmed=args.confirm,
    )
    return setup, bundle


def execute(args, *, provider_factory=None, grader=None):
    from pipelines.math_ext_data import validate_output
    from pipelines.math_ext_execution import check_mode
    from pipelines.math_ext_run import load_source
    from rafr.contracts.schemas import Arm, Split

    live = args.stage == "live"
    if live and not args.confirm:
        raise ValueError("live is not authorized without explicit --confirm")
    if live and (provider_factory is not None or grader is not None):
        raise ValueError("test boundary injection is forbidden in live execution")
    source = load_source(args.manifest)
    check_mode(args.stage, confirmed=args.confirm, source_mode=source.execution_mode)
    ext = load_extension(args.config, source, operation=args.command, live=live)
    out = validate_output(
        resolve(args.out), source_root=source.root, allow_existing=args.command != "starts"
    )
    if (out / "STOPPED.json").exists():
        raise ValueError("run is STOPPED; preserve evidence and use a new output")
    if args.command != "starts" and (not args.run_id or args.run_id != out.name):
        raise ValueError("run-id must exactly equal output directory name")
    if args.command != "starts":
        family = {
            "exp1": "math_exp1ext",
            "loop": "math_exp2ext",
            "judge": "math_exp2ext",
            "ceiling": "math_ceiling",
        }[args.command]
        expected_prefix = family + ("_smoke_" if args.smoke else "_")
        if not out.name.startswith(expected_prefix) or (
            not args.smoke and out.name.startswith(family + "_smoke_")
        ):
            raise ValueError("output run name must explicitly match the experiment and smoke scope")
    if (out / "ext_meta.json").exists():
        previous = json.loads((out / "ext_meta.json").read_text(encoding="utf-8"))
        if previous.get("run_id") != args.run_id or previous.get("execution_mode") != args.stage:
            raise ValueError("existing run identity differs")
    states, cohort_file, cohort_sha = selected(
        source, ext, cohort=args.cohort, smoke=args.smoke, live=live
    )
    default_arms = (
        ext["exp1"][args.cohort]["arms"]
        if args.command == "exp1"
        else (["no_verifier"] if args.command == "starts" and not args.fill_original else [])
    )
    requested = getattr(args, "arms", None)
    arms = [Arm(a) for a in (requested if requested is not None else default_arms)]
    if len(set(arms)) != len(arms):
        raise ValueError("duplicate arms")
    if (
        args.command == "exp1"
        and live
        and not args.smoke
        and arms != [Arm(a) for a in ext["exp1"][args.cohort]["arms"]]
    ):
        raise ValueError("formal Exp1 requires the complete frozen arm set")
    source.assert_unchanged()
    if args.command == "starts":
        from pipelines.math_ext_transport import export_public_bundle
        from rafr.experiments.exp2_ext import full_pairing_plan

        if args.fill_original and (args.cohort != "e_incorrect" or args.smoke):
            raise ValueError("original fill requires the formal incorrect cohort")
        plan = full_pairing_plan(
            [s.x0_id for s in states],
            new_arms=arms,
            fill_arms=[Arm.A1, Arm.B] if args.fill_original else (),
            original_assignment=source.read_bound_json("exp2/assignment.json")
            if args.fill_original
            else None,
        )
        return export_public_bundle(
            source,
            out,
            cohort=args.cohort,
            plan=plan,
            smoke=bool(args.smoke),
            fixture=not live,
            state_ids=[s.state_id for s in states],
            cohort_file=cohort_file,
            cohort_sha256=cohort_sha,
        )
    if args.command == "exp1":
        from pipelines.math_ext_exp1 import (
            COHORTS,
            reconstruct_original_channel,
            run_math_exp1_extension,
        )
        from rafr.contracts.schemas import Lambda
        from rafr.domains.math.generation import build_controller_config
        from rafr.formatter.math_extended import math_wording

        if set(arms) - set(COHORTS[args.cohort].arms):
            raise ValueError("unsupported arm for this Exp1 cohort")
        math_wording()  # prompt guard before provider construction
        if args.cohort == "e_incorrect" and not args.smoke:
            reconstruct_original_channel(
                source,
                states,
                lambdas=(Lambda.LOW, Lambda.MID, Lambda.HIGH),
                controller_cfg=build_controller_config(source.master_seed, 1),
            )
        cal = calibrator_for(source, ext, arms, live=live)
        return run_math_exp1_extension(
            client_for(out, live=live, provider_factory=provider_factory),
            source,
            run_dir=out,
            run_id=args.run_id,
            cohort=args.cohort,
            state_ids=[s.state_id for s in states],
            cohort_file=cohort_file,
            cohort_sha256=cohort_sha,
            arms=arms,
            calibrator=cal,
            smoke=bool(args.smoke),
            execution_mode=args.stage,
            confirmed=args.confirm,
            extension_config=ext,
        )
    if args.command in {"loop", "judge"}:
        from pipelines.math_ext_exp2 import grade_math_exp2, run_math_exp2
        from rafr.domains.math.generation import build_controller_config
        from rafr.formatter.math_extended import build_no_verifier_config

        setup, bundle = bundle_setup(args, source, ext, out=out)
        if args.command == "loop":
            if any(a is not Arm.NO_VERIFIER for values in setup.plan.values() for a in values):
                raise ValueError("local loop supports only NoVerifier; use pod for verifier arms")
            cfg, nv = (
                build_controller_config(source.master_seed, 4),
                build_no_verifier_config(source.master_seed, 4),
            )
            trajectories = run_math_exp2(
                client_for(out, live=live, provider_factory=provider_factory), cfg, nv, setup
            )
            return {
                "status": "COMPLETE",
                "stage": args.stage,
                "n_trajectories": len(trajectories),
                "run_dir": str(out),
                "judged": False,
            }
        from rafr.experiments.exp2 import complete_rounds, load_round

        if complete_rounds(out, 4) != [1, 2, 3, 4]:
            raise ValueError("judge requires four completed loop rounds")
        trajectories = load_round(out, 4)
        has_verifier = any(
            a is not Arm.NO_VERIFIER for choices in setup.plan.values() for a in choices
        )
        if has_verifier or (out / "pod_identity.json").exists() or args.pod_receipt:
            if args.pod_receipt is None:
                raise ValueError("pod judge requires the independently bound pod receipt")
            from env.math_ext_remote import validate_delivery

            from rafr.domains.math.storage import digest

            receipt = json.loads(
                checked(args.pod_receipt, args.pod_receipt_sha256).read_text(encoding="utf-8")
            )
            if (
                receipt.get("run_id") != args.run_id
                or receipt.get("source_identity_sha256") != digest(source.provenance())
                or receipt.get("source_mode") != source.execution_mode
                or receipt.get("files", {}).get("transport.json") != args.bundle_sha256
            ):
                raise ValueError("pod receipt source/bundle/run binding differs")
            validate_delivery(
                out,
                receipt=receipt,
                receipt_sha256=args.pod_receipt_sha256,
                bundle=bundle,
                channel=source.read_bound_json("fit.json")["channel"],
            )
        from rafr.contracts.schemas import MathLabel
        from rafr.data.labels import LabelPurpose

        split = (
            Split.D_GATE
            if args.cohort == "e_correct"
            else (Split.UNLABELED_POOL if args.smoke else Split.D_EXP1)
        )
        purpose = (
            LabelPurpose.EXT_SMOKE
            if args.cohort == "e_correct" and args.smoke
            else (
                LabelPurpose.ECORRECT_COHORT
                if args.cohort == "e_correct"
                else (LabelPurpose.DRY_RUN if args.smoke else LabelPurpose.EXP2_T1)
            )
        )
        records = {r.state_id: r for r in source.labels(split, purpose)}
        setup.initial_grades = {
            s.x0_id: {
                "status": "correct"
                if records[s.state_id].label is MathLabel.CORRECT
                else "incorrect",
                "task_success": records[s.state_id].label is MathLabel.CORRECT,
            }
            for s in states
        }
        evaluator = make_e(args, source, states, out, grader=grader)
        results = grade_math_exp2(setup, trajectories, evaluator, e_identity=evaluator.identity)
        return {
            "status": "COMPLETE",
            "stage": args.stage,
            "n_results": len(results),
            "run_dir": str(out),
        }
    if args.command == "ceiling":
        from pipelines.math_ext_ceiling import run_ceiling
        from rafr.domains.math.storage import digest

        if args.cohort != "e_incorrect":
            raise ValueError("ceiling requires incorrect starts")
        evaluator = make_e(args, source, states, out, grader=grader)
        initial = {
            "split": "unlabeled_pool" if args.smoke else "d_exp1",
            "sha256": digest(source.provenance()),
            "source_root": str(source.root),
            "source_execution_mode": source.execution_mode,
        }
        results = run_ceiling(
            client_for(out, live=live, provider_factory=provider_factory),
            states,
            run_dir=out,
            run_id=args.run_id,
            seed=source.master_seed,
            initial_identity=initial,
            grade_callback=evaluator,
            e_identity=evaluator.identity,
            smoke=bool(args.smoke),
            execution_mode=args.stage,
            confirmed=args.confirm,
        )
        return {
            "status": "COMPLETE",
            "stage": args.stage,
            "n_results": len(results),
            "run_dir": str(out),
        }
    raise ValueError("unknown operation")


def make_e(args, source, states, out, *, grader=None):
    from pipelines.math_ext_e import ExtensionMathEvaluator

    refs = (
        bundle_references(args.dataset_bundle, source, states)
        if args.dataset_bundle
        else private_references(args.references, args.references_sha256, states)
    )
    if args.stage == "live":
        from rafr.domains.math.grading import MathGrader

        if grader is not None:
            raise ValueError("live E cannot inject a grading backend")
        grader = MathGrader()
        snapshot = source.read_bound_json("snapshot.json")
        if grader.configuration() != snapshot.get("math_identity", {}).get("grader"):
            raise ValueError("local Math-Verify identity differs from the original source grader")
    source.assert_unchanged()
    return ExtensionMathEvaluator(
        out,
        questions={s.x0_id: s.x0 for s in states},
        private_references=refs,
        source_identity=source.provenance(),
        source_root=source.root,
        grader=grader,
    )


# 2026-11-05: a real freeze writes the files configs/math/ext.yaml names (the calibrator is
# committed and tagged; the cohort, like detox's, sits under data/ext/ and is pinned by sha256).
FROZEN_TARGETS = {
    "calibration": "frozen/calibration_math.json",
    "cohort": "data/ext/ecorrect_cohort_math.jsonl",
}


def freeze_target(kind, stage, out, source):
    """Where ``freeze`` writes. ``real_prepare``: the configured path above, once, never
    overwritten (``--out`` is refused). ``offline_stub``: a new directory ``--out`` (fixtures)."""
    import yaml

    from pipelines.math_ext_data import fresh_output

    name = Path(FROZEN_TARGETS[kind]).name
    if stage != "real_prepare":
        if out is None:
            raise ValueError("an offline_stub freeze needs --out")
        return fresh_output(source, resolve(out)) / name
    if out is not None:
        raise ValueError("a real freeze writes the configured file; --out is for offline_stub")
    ext = yaml.safe_load((ROOT / "configs/math/ext.yaml").read_text(encoding="utf-8"))
    configured = (
        ext["calibration"]["frozen_file"]
        if kind == "calibration"
        else ext["cohorts"]["e_correct"]["file"]
    )
    if configured != FROZEN_TARGETS[kind]:
        raise ValueError(
            f"configs/math/ext.yaml names {configured!r}, not {FROZEN_TARGETS[kind]!r}"
        )
    target = paths.resolve(configured)
    if target.exists() or target.is_symlink():
        raise ValueError(f"{configured} exists; a frozen file is written once, never overwritten")
    if any(p.is_symlink() for p in target.parents if p.is_relative_to(ROOT)):
        raise ValueError(f"{configured}: a parent directory is a symlink")
    if target.resolve().is_relative_to(source.root):
        raise ValueError("a frozen file may not be written inside the original math run")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def freeze(args):
    """Explicit M-C decision: write a new frozen file; never edit config or candidate."""
    from pipelines.math_ext_data import same_bound_source
    from pipelines.math_ext_run import load_source
    from rafr.calibration.apply import Calibrator, write_frozen
    from rafr.contracts.schemas import MathLabel, Split
    from rafr.core.seeding import component_seed
    from rafr.data.labels import LabelPurpose
    from rafr.experiments.ext_cohort import draw_cohort, read_cohort, write_cohort

    if not args.confirm:
        raise ValueError(
            "freezing requires explicit --confirm after review of candidate diagnostics"
        )
    source = load_source(args.manifest)
    wanted_mode = "live" if args.stage == "real_prepare" else "offline_stub"
    if source.execution_mode != wanted_mode:
        raise ValueError("freeze stage/source mismatch")
    path = checked(args.candidate, args.candidate_sha256)
    if args.kind == "calibration":
        cal = Calibrator.load(path, expected_sha256=args.candidate_sha256)
        cal.check_verifiers(*source.verifier_ids)
        record = cal.record
        provenance = record.get("fit_data", {})
        if not same_bound_source(provenance, source):
            raise ValueError("calibration candidate source differs")
        if (
            provenance.get("stage")
            != ("offline_candidate" if wanted_mode == "live" else "offline_stub")
            or provenance.get("scientifically_frozen") is not False
        ):
            raise ValueError("expected an unfrozen candidate from the matching stage")
        receipt = json.loads(checked(args.receipt, args.receipt_sha256).read_text(encoding="utf-8"))
        if (
            receipt.get("calibrator_sha256") != args.candidate_sha256
            or not same_bound_source(receipt.get("source"), source)
            or receipt.get("stage") != provenance["stage"]
            or receipt.get("scientifically_frozen") is not False
            or (wanted_mode == "live" and receipt.get("n_boot") != 10000)
            or receipt.get("diagnostics_sha256") != args.diagnostics_sha256
        ):
            raise ValueError(
                "calibration completion receipt does not bind candidate/source/diagnostics"
            )
        checked(args.diagnostics, args.diagnostics_sha256)
        clip = json.loads(
            checked(path.parent / "clip_report.json", receipt.get("clip_report_sha256")).read_text(
                encoding="utf-8"
            )
        )
        cells = clip.get("cells", {})
        required_cells = {
            f"{split}|{v}" for split in ("unlabeled_pool", "d_gate", "d_exp1") for v in ("v1", "v2")
        }
        if (
            clip.get("epsilon") != 1e-7
            or clip.get("max_share") != 0.01
            or set(cells) != required_cells
        ):
            raise ValueError("calibration clipping report contract differs")
        for row in cells.values():
            n, clipped = row.get("n"), row.get("clipped")
            if (
                type(n) is not int
                or n <= 0
                or type(clipped) is not int
                or not 0 <= clipped <= n
                or row.get("share") != clipped / n
                or clipped / n > 0.01
            ):
                raise ValueError("calibration clipping exceeds 1% or is inconsistent")
        provenance.update(
            scientifically_frozen=True,
            freeze_stage=args.stage,
            review_receipt_sha256=args.receipt_sha256,
            diagnostics_sha256=args.diagnostics_sha256,
            clip_report_sha256=receipt["clip_report_sha256"],
        )
        source.assert_unchanged()
        out = freeze_target("calibration", args.stage, args.out, source)
        sha = write_frozen(out, record)
    else:
        header, ids = read_cohort(path, args.candidate_sha256)
        seed = component_seed(source.master_seed, "ext_ecorrect_cohort")
        rows = source.labels(Split.D_GATE, LabelPurpose.ECORRECT_COHORT)
        candidates = [r.state_id for r in rows if r.label is MathLabel.CORRECT]
        target = 300 if wanted_mode == "live" else header.get("target_n")
        if (
            not same_bound_source(header.get("source"), source)
            or header.get("seed") != seed
            or header.get("target_n") != target
            or header.get("n_candidates") != len(candidates)
            or header.get("scientifically_frozen") is not False
            or ids != draw_cohort(candidates, target, seed)
        ):
            raise ValueError("cohort candidate differs from the bound seeded draw")
        header.update(scientifically_frozen=True, freeze_stage=args.stage)
        source.assert_unchanged()
        out = freeze_target("cohort", args.stage, args.out, source)
        sha = write_cohort(out, ids, header)
    return {
        "status": "FROZEN",
        "file": str(out),
        "sha256": sha,
        "stage": args.stage,
        "live_executed": False,
        "config_updated": False,
    }
