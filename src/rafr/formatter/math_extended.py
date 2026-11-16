"""Math extension wording under the 2026-11-01 handoff.

Verdict/pi blocks are referenced from detox, never copied. NoVerifier files only delete
checker-related sentences from the original math controller prompts.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from rafr.contracts.schemas import Arm, Observation, UncertaintyFields
from rafr.controller import decide as ctrl
from rafr.core.config import assert_frozen
from rafr.domains.math import generation
from rafr.formatter import math_feedback
from rafr.formatter import templates as tpl
from rafr.formatter.extended import ExtWording, fill_slots, read_block_template
from rafr.formatter.fields import compute_fields

PROMPT_DIR = Path(__file__).with_name("prompts") / "math_ext"
PROMPT_FILES = ("no_verifier_system.txt", "no_verifier_rewrite.txt")
SHARED_DIR = Path(__file__).with_name("prompts") / "detox"
SHARED_FILES = ("verdict_only.txt", "pihat.txt")


def _read(directory: Path, name: str, *, allow_draft: bool) -> str:
    path = directory / name
    assert_frozen({}, extra_files=[path], only_sections=[], allow_placeholders=allow_draft)
    text = read_block_template(path)
    if not text.strip() or text.strip() == "TBD":
        raise ValueError(f"empty or TBD math extension prompt: {path}")
    return text


def original_feedback(arm: Arm, obs: Observation, donor: Observation | None = None) -> str:
    """Original math feedback, including the complete checked permutation donor."""
    if arm not in (Arm.A1, Arm.B, Arm.PERMUTED_B):
        raise ValueError("original math feedback only accepts A1/B/permuted_B")
    if arm is Arm.PERMUTED_B:
        if donor is None:
            raise ValueError("permuted_B requires the complete donor observation")
        return math_feedback.format_observation(
            arm, obs, permuted_fields=compute_fields(donor), donor_observation=donor
        )
    if donor is not None:
        raise ValueError("only permuted_B may receive a donor")
    return math_feedback.format_observation(arm, obs)


def _verdict_template() -> str:
    """B minus the two distance rows and their two explanatory sentences, exactly."""
    text = tpl.DERIVED_BLOCK
    for line in (
        "{v1_name} distance from its threshold: {m1}\n",
        "{v2_name} distance from its threshold: {m2}\n",
        " The distance is the score minus the threshold, signed: positive means at or above "
        "the threshold, negative means below.",
        " Distances from different checkers are not comparable to each other.",
    ):
        if text.count(line) != 1:
            raise ValueError("original B wording changed; re-review verdict-only deletion")
        text = text.replace(line, "")
    return text


def math_wording(prompt_dir: Path | None = None, *, allow_draft: bool = False) -> ExtWording:
    verdict = _read(SHARED_DIR, "verdict_only.txt", allow_draft=False)
    if verdict != _verdict_template():
        raise ValueError("verdict_only must be exactly original B with only distances removed")
    names = {"v1_name": tpl.V1_NAME, "v2_name": tpl.V2_NAME}
    pihat = fill_slots(_read(SHARED_DIR, "pihat.txt", allow_draft=False), names, complete=False)
    if any(pihat.count("{" + name + "}") != 1 for name in ("pihat1", "pihat2")):
        raise ValueError("pi-hat prompt must have each pi-hat slot exactly once")
    # Validate all template slots now, before any provider can be called.
    fill_slots(pihat, {"pihat1": "0.1", "pihat2": "0.2"})

    def render_verdict(fields: UncertaintyFields) -> str:
        return fill_slots(
            verdict,
            {
                **names,
                "y1": tpl.FLAG_WORDS[fields.y1],
                "y2": tpl.FLAG_WORDS[fields.y2],
                "agreement": tpl.AGREEMENT_WORDS[fields.agreement],
                "conflict_direction": tpl.CONFLICT_WORDS[fields.conflict_direction.value],
            },
        )

    return ExtWording(
        render_a1=math_feedback.render_a1,
        render_b=lambda obs: original_feedback(Arm.B, obs),
        render_verdict=render_verdict,
        pihat_template=pihat,
        separator=math_feedback.SEPARATOR,
    )


def build_no_verifier_config(
    master_seed: int,
    round_budget: int,
    prompt_dir: Path | None = None,
    *,
    allow_draft: bool = False,
) -> ctrl.ControllerConfig:
    """Retain math's exact controller settings and retry policy, remove checker inputs."""
    cfg = generation.build_controller_config(master_seed, round_budget)
    directory = Path(prompt_dir or PROMPT_DIR)
    task_path = generation.PROMPT_DIR / "task.txt"
    assert_frozen({}, extra_files=[task_path], only_sections=[])
    task = task_path.read_text(encoding="utf-8").strip()
    system = _read(directory, "no_verifier_system.txt", allow_draft=allow_draft)
    rewrite = _read(directory, "no_verifier_rewrite.txt", allow_draft=allow_draft)
    # Partial application is intentional; no caller-supplied problem/answer is formatted twice.
    fill_slots(system, {"task": task, "k": str(round_budget)})
    fill_slots(
        rewrite, {name: "" for name in ("task", "x0", "x_t", "x0_section_start", "x0_section_end")}
    )
    if "{x0}" not in rewrite or "{x_t}" not in rewrite or "{k}" not in system:
        raise ValueError("NoVerifier prompts must retain problem, solution and decision budget")
    cfg = replace(
        cfg,
        system_prompt=fill_slots(system, {"task": task}, complete=False),
        rewrite_template=ctrl.resolve_x0_section(
            fill_slots(rewrite, {"task": task}, complete=False), show_x0=True
        ),
    )
    return ctrl.with_round_budget(cfg, round_budget)
