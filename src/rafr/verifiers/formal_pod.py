"""The formal pod's image and GPU (2026-09-24; point 6).

Formal scores (step 6's D_gate / D_Exp1, Exp 2's in-process scoring) come from one fixed pod:
image ``runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`` on an A100 SXM 80GB, written in
``configs/models.yaml: verifiers.formal_pod``. the RunPod driver (not released) and the RunPod
driver (not released) check it over ssh when they bring the pod up and before every job they start,
and the jobs check it again on the pod itself before a model is loaded; any difference stops them.

The image name is not visible from inside a container, so what is checked is what the tag
promises and the container can show: the CUDA toolkit version the image was built with
(``CUDA_VERSION`` in PID 1's environment, set by the image), the torch version and its CUDA,
the Ubuntu release (``/etc/os-release``), and the GPU's name as the driver reports it. The
expected values sit next to the image name in the config (``expect``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

KEYS: tuple[str, ...] = ("cuda_version", "torch", "torch_cuda", "ubuntu", "gpu_name")


class FormalPodError(RuntimeError):
    """This machine is not the fixed formal pod."""


def expected(models_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """``verifiers.formal_pod``: ``image``, ``gpu`` and the ``expect`` values, all present."""
    fp = ((models_cfg.get("verifiers") or {}).get("formal_pod")) or {}
    exp = dict(fp.get("expect") or {})
    missing = [k for k in ("image", "gpu") if not fp.get(k)] + [k for k in KEYS if not exp.get(k)]
    if missing:
        raise FormalPodError(f"models.verifiers.formal_pod lacks {missing}")
    return {
        "image": str(fp["image"]),
        "gpu": str(fp["gpu"]),
        "expect": {k: str(exp[k]) for k in KEYS},
    }


def _normal(key: str, value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if key == "torch":  # "2.8.0+cu128" is torch 2.8.0
        text = text.split("+")[0]
    return text or None


def mismatches(expect: Mapping[str, Any], seen: Mapping[str, Any]) -> list[str]:
    """One line per key whose observed value is not the expected one (missing counts)."""
    out = []
    for k in KEYS:
        want, got = _normal(k, expect.get(k)), _normal(k, seen.get(k))
        if want != got:
            out.append(f"{k}: expected {want!r}, found {got!r}")
    return out


def _os_release(path: Path = Path("/etc/os-release")) -> str | None:
    try:
        rows = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    except OSError:
        return None
    if rows.get("ID", "").strip('"') != "ubuntu":
        return None
    return rows.get("VERSION_ID", "").strip('"') or None


def _image_cuda_version() -> str | None:
    """``CUDA_VERSION`` as the image set it: PID 1's environment (a login shell may not carry
    the image's ENV), falling back to this process's."""
    try:
        for item in Path("/proc/1/environ").read_bytes().split(b"\0"):
            if item.startswith(b"CUDA_VERSION="):
                return item.split(b"=", 1)[1].decode() or None
    except OSError:
        pass
    return os.environ.get("CUDA_VERSION") or None


def observe_here() -> dict[str, Any]:
    """The same facts, read inside this process (on the pod, before a model is loaded)."""
    seen: dict[str, Any] = {"cuda_version": _image_cuda_version(), "ubuntu": _os_release()}
    try:
        import torch
    except ImportError:
        return {**seen, "torch": None, "torch_cuda": None, "gpu_name": None}
    seen["torch"] = torch.__version__
    seen["torch_cuda"] = torch.version.cuda
    seen["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return seen


def check_here(models_cfg: Mapping[str, Any], echo: Any = None) -> dict[str, Any]:
    """Raise :class:`FormalPodError` unless this machine is the configured pod."""
    want = expected(models_cfg)
    seen = observe_here()
    bad = mismatches(want["expect"], seen)
    if bad:
        raise FormalPodError(
            f"this is not the formal pod ({want['image']} on {want['gpu']}): " + "; ".join(bad)
        )
    if echo:
        echo(f"[formal pod] {want['image']} on {want['gpu']}: {seen}")
    return {"expected": want, "observed": seen}
