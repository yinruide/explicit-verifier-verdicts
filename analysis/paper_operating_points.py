"""Paper version of the operating-point figure of analysis/ext_decision_quality.py.

Reads one ext_decision_quality ``report.json`` and draws it. No number is recomputed: every
point and interval is the report's own estimate and 95% interval. The dashed equal-BA lines
are reference geometry only (BA = 1 - (FA + FRR) / 2, so FA + FRR = 2 (1 - BA)). The analysis
outputs are never written to; the figure goes to a new path (vector PDF plus a PNG preview).

The arms carry their intervals; the threshold rules are reference points (their intervals are
in the report's tables). A plotted value or interval end outside the axis ranges is an error,
so nothing is clipped silently.

    python analysis/paper_operating_points.py \
        --report runs/ext_decision_quality_detox_20261101T074059Z/report.json \
        --out figures/detox_operating_points
    python analysis/paper_operating_points.py \
        --report runs/ext_decision_quality_math_20261106T110046Z/report.json \
        --out figures/math_operating_points
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from analysis.ext_decision_quality import ARM_ORDER, MARKERS, SERIES

ARM_NAMES = {
    "no_verifier": "No-Verifier",
    "A1": "A$_1$",
    "verdict_only": "Verdict-only",
    "B": "B",
    "B_pihat": r"B+$\hat{\pi}$",
    "B_pihat_mismatched": r"B+$\hat{\pi}$ (mismatch)",
    "permuted_B": r"B$_{\mathrm{perm}}$",
}
RULE_NAMES = {"V1_only": "V$_1$ only", "V2_only": "V$_2$ only", "OR": "OR", "AND": "AND"}
RULE_LEGEND = "Rule"
TEXT_INDEPENDENT = "verdicts independent of the text"
X_LABEL = "False reject rate (passing states)"
Y_LABEL = "False accept rate (failing states)"
XLIM = (0.0, 0.55)
# 2026-11-06: the false-accept axis runs to 1.0 (math's AND rule sits at 0.987 at lambda
# high), the same for detox and math so the two figures share their axes; it was 0.95 for detox.
YLIM = (0.0, 1.0)
ISO_BA = (0.5, 0.6, 0.7, 0.8)
FIGSIZE = (7.0, 2.6)  # inches: full width of a two-column page
MIN_FONT = 7.0  # points
FONT, FONT_LABEL, FONT_TITLE = 7.0, 8.0, 9.0
INK, INK_2, ISO_LINE, ISO_TEXT, AXIS, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#b4b3ad",
    "#898781",
    "#9f9e96",
    "#ffffff",
)
ARM_MS, RING_MS = 4.4, 6.2  # marker sizes in points
LABEL_GAPS = (2.5, 6.0, 10.0)  # points between a rule's ring and its label
LABEL_DIRS = ((1, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, -1), (-1, 0))
ISO_INSETS = (9.0, 16.0, 24.0, 34.0, 46.0, 60.0)  # points from a line's end to its label
ISO_CLEAR = 3.0  # points kept between a line's label and any marker or ring


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _title(lam: str) -> str:
    return rf"$\lambda_{{\mathrm{{{lam}}}}}$"


def _err(s: Mapping[str, Any]) -> list[list[float]]:
    lo, hi = s["ci"]
    return [[s["estimate"] - lo], [hi - s["estimate"]]]


def check_range(
    report: Mapping[str, Any], xlim: Sequence[float], ylim: Sequence[float]
) -> list[str]:
    """Every value the figure plots that falls outside the axis ranges (empty when all fit)."""
    out = []
    for lam, op in report["operating_points"].items():
        for kind, members in op["points"].items():
            for name, p in members.items():
                for key, (lo, hi) in (("FRR", xlim), ("FA", ylim)):
                    shown = [p[key]["estimate"]]
                    if kind == "arms":
                        shown += list(p[key]["ci"])
                    bad = [v for v in shown if not (lo <= v <= hi)]
                    if bad:
                        out.append(f"{lam} {name} {key} {bad} outside [{lo}, {hi}]")
    return out


def _box(cx: float, cy: float, half_w: float, half_h: float):
    from matplotlib.transforms import Bbox

    return Bbox.from_extents(cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _hits(box, others: Sequence[Any], pad: float = 0.0) -> int:
    grown = box.padded(pad)
    return sum(1 for o in others if grown.overlaps(o))


def _obstacles(ax, op: Mapping[str, Any], px: float) -> tuple[list[Any], list[Any]]:
    """Display-space boxes of the markers and rings (hard) and the interval bars (soft)."""
    hard, soft = [], []
    to = ax.transData.transform
    for kind, members in op["points"].items():
        r = (ARM_MS if kind == "arms" else RING_MS) / 2 * px + 0.5 * px
        for p in members.values():
            x, y = p["FRR"]["estimate"], p["FA"]["estimate"]
            cx, cy = to((x, y))
            hard.append(_box(cx, cy, r, r))
            if kind == "arms":
                (x0, _), (x1, _) = to((p["FRR"]["ci"][0], y)), to((p["FRR"]["ci"][1], y))
                (_, y0), (_, y1) = to((x, p["FA"]["ci"][0])), to((x, p["FA"]["ci"][1]))
                soft.append(_box((x0 + x1) / 2, cy, (x1 - x0) / 2, 0.6 * px))
                soft.append(_box(cx, (y0 + y1) / 2, 0.6 * px, (y1 - y0) / 2))
    return hard, soft


def _iso_segment(c: float, xlim: Sequence[float], ylim: Sequence[float]):
    """The part of FA + FRR = c inside the axes, from its upper-left to its lower-right end."""
    x0, x1 = max(xlim[0], c - ylim[1]), min(xlim[1], c - ylim[0])
    if x0 >= x1:
        return None
    return (x0, c - x0), (x1, c - x1)


def _place_iso_labels(axes, ops, lams, segments, renderer, px, placed) -> None:
    """Each equal-BA line's value goes near one of its ends, at the same spot in every panel,
    the spot nearest an end that clears every marker and interval in all panels."""
    ref = axes[0]
    for ba, seg in segments.items():
        (ux, uy), (lx, ly) = ref.transData.transform(seg[0]), ref.transData.transform(seg[1])
        length = math.hypot(lx - ux, ly - uy)
        dx, dy = (lx - ux) / length, (ly - uy) / length
        angle = math.degrees(math.atan2(dy, dx))
        spots = []
        for inset in ISO_INSETS:
            d = inset * px
            if d < length / 2:
                spots.append((lx - dx * d, ly - dy * d))  # lower-right end first
                spots.append((ux + dx * d, uy + dy * d))
        best, best_score = None, None
        for sx, sy in spots:
            data = ref.transData.inverted().transform((sx, sy))
            score = 0
            for ax, lam in zip(axes, lams, strict=True):
                t = _iso_text(ax, data, ba, angle)
                box = t.get_window_extent(renderer)
                t.remove()
                hard, soft = _obstacles(ax, ops[lam], px)
                score += 100 * _hits(box, hard, ISO_CLEAR * px) + _hits(box, soft, px)
            if best_score is None or score < best_score:
                best, best_score = data, score
            if score == 0:
                break
        for ax, labels in zip(axes, placed, strict=True):
            labels.append(_iso_text(ax, best, ba, angle).get_window_extent(renderer))


def _iso_text(ax, data, ba: float, angle: float):
    return ax.text(
        data[0],
        data[1],
        f"{ba:.1f}",
        rotation=angle,
        rotation_mode="anchor",
        ha="center",
        va="center",
        fontsize=FONT,
        color=ISO_TEXT,
        zorder=1.5,
        bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"},
    )


def _place_rule_labels(ax, op, renderer, px, placed) -> dict[str, tuple[int, ...]]:
    """Each rule's name goes beside its ring, at the nearest spot inside the axes that covers
    no marker, ring or label (and as few interval bars as possible)."""
    hard, soft = _obstacles(ax, op, px)
    frame = ax.get_window_extent(renderer)
    out = {}
    for name, p in op["points"]["rules"].items():
        x, y = p["FRR"]["estimate"], p["FA"]["estimate"]
        best, best_key = None, None
        for gi, gap in enumerate(LABEL_GAPS):
            for di, (ux, uy) in enumerate(LABEL_DIRS):
                norm = math.hypot(ux, uy)
                off = (RING_MS / 2 + gap) / norm
                ha = "left" if ux > 0 else "right" if ux < 0 else "center"
                va = "bottom" if uy > 0 else "top" if uy < 0 else "center"
                t = _rule_text(ax, name, (x, y), (ux * off, uy * off), ha, va)
                box = t.get_window_extent(renderer)
                t.remove()
                inside = frame.x0 <= box.x0 and box.x1 <= frame.x1
                inside = inside and frame.y0 <= box.y0 and box.y1 <= frame.y1
                blocked = _hits(box, hard, 0.5 * px) + _hits(box, placed, 1.0 * px)
                key = (0 if inside else 1, blocked, _hits(box, soft), gi, di)
                if best_key is None or key < best_key:
                    best, best_key = ((ux * off, uy * off), ha, va), key
        off, ha, va = best
        t = _rule_text(ax, name, (x, y), off, ha, va)
        placed.append(t.get_window_extent(renderer))
        out[name] = best_key
    return out


def _rule_text(ax, name, xy, off, ha, va):
    return ax.annotate(
        RULE_NAMES.get(name, name),
        xy,
        xytext=off,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=FONT,
        color=INK_2,
        zorder=2.5,
        bbox={"boxstyle": "square,pad=0.1", "facecolor": SURFACE, "edgecolor": "none"},
    )


def draw(
    report: Mapping[str, Any],
    out_stem: Path,
    *,
    source: str,
    xlim: Sequence[float] = XLIM,
    ylim: Sequence[float] = YLIM,
    iso_ba: Sequence[float] = ISO_BA,
) -> dict[str, Any]:
    """Writes ``<out_stem>.pdf`` and ``<out_stem>.png``; returns the paths and the label
    placement (per panel and rule: outside-the-axes flag, blocked count, bars crossed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from matplotlib.ticker import MultipleLocator

    outside = check_range(report, xlim, ylim)
    if outside:
        raise ValueError("values outside the axis ranges: " + "; ".join(outside))
    ops = report["operating_points"]
    lams = list(ops)
    present = {a for lam in lams for a in ops[lam]["points"]["arms"]}
    arms = [a for a in ARM_ORDER if a in present]
    if set(arms) != present:
        raise ValueError(f"arms without a paper name: {sorted(present - set(arms))}")
    style = {a: (SERIES[i], MARKERS[i]) for i, a in enumerate(arms)}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": FONT,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )
    fig, axes = plt.subplots(1, len(lams), figsize=FIGSIZE, sharex=True, sharey=True)
    axes = list(axes)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.072, right=0.995, bottom=0.215, top=0.865, wspace=0.07)
    px = fig.dpi / 72.0
    segments = {ba: _iso_segment(2 * (1 - ba), xlim, ylim) for ba in iso_ba}
    segments = {ba: s for ba, s in segments.items() if s is not None}
    title_pad = 2.0 + FONT * 1.25 + 1.5
    for ax, lam in zip(axes, lams, strict=True):
        op = ops[lam]
        ax.set_facecolor(SURFACE)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.tick_params(colors=INK_2, labelsize=FONT, pad=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        for (x0, y0), (x1, y1) in segments.values():
            ax.plot([x0, x1], [y0, y1], color=ISO_LINE, lw=0.6, ls=(0, (3, 2)), zorder=1)
        for p in op["points"]["rules"].values():
            ax.plot(
                p["FRR"]["estimate"],
                p["FA"]["estimate"],
                marker="o",
                markersize=RING_MS,
                markerfacecolor="none",
                markeredgecolor=INK_2,
                markeredgewidth=0.9,
                linestyle="none",
                zorder=5,
            )
        for a in arms:
            p = op["points"]["arms"][a]
            color, marker = style[a]
            x, y = p["FRR"]["estimate"], p["FA"]["estimate"]
            ax.errorbar(
                x,
                y,
                xerr=_err(p["FRR"]),
                yerr=_err(p["FA"]),
                fmt="none",
                ecolor=color,
                elinewidth=0.7,
                capsize=1.4,
                capthick=0.7,
                zorder=3,
            )
            ax.plot(
                x,
                y,
                marker=marker,
                markersize=ARM_MS,
                color=color,
                markeredgecolor=SURFACE,
                markeredgewidth=0.5,
                linestyle="none",
                zorder=4,
            )
        # Every value is inside the axis ranges (check_range), but a marker or ring centred near
        # an edge (math's AND at FA 0.987 under the 1.0 limit) would be cut by the axes frame.
        for artist in [*ax.lines, *ax.collections]:
            artist.set_clip_on(False)
        ax.set_title(_title(lam), fontsize=FONT_TITLE, color=INK, pad=title_pad)
        if report["lambda_levels"][lam]["flags_independent_of_text"]:
            ax.annotate(
                TEXT_INDEPENDENT,
                (0.5, 1.0),
                xycoords="axes fraction",
                xytext=(0, 2.0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=FONT,
                color=INK_2,
            )
    axes[0].set_ylabel(Y_LABEL, fontsize=FONT_LABEL, color=INK)
    axes[len(axes) // 2].set_xlabel(X_LABEL, fontsize=FONT_LABEL, color=INK, labelpad=2)
    handles = [
        plt.Line2D(
            [],
            [],
            marker=style[a][1],
            color=style[a][0],
            markeredgecolor=SURFACE,
            markeredgewidth=0.5,
            markersize=ARM_MS + 0.6,
            linestyle="none",
            label=ARM_NAMES[a],
        )
        for a in arms
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="o",
            markersize=RING_MS,
            markerfacecolor="none",
            markeredgecolor=INK_2,
            markeredgewidth=0.9,
            linestyle="none",
            label=RULE_LEGEND,
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5 * (0.072 + 0.995), 0.0),
        ncol=len(handles),
        frameon=False,
        fontsize=FONT,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=1.3,
        borderaxespad=0.2,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = [[] for _ in axes]  # display boxes of the labels already drawn, per panel
    _place_iso_labels(axes, ops, lams, segments, renderer, px, placed)
    placement = {
        lam: _place_rule_labels(ax, ops[lam], renderer, px, labels)
        for ax, lam, labels in zip(axes, lams, placed, strict=True)
    }

    small = sorted(
        {
            t.get_fontsize()
            for t in fig.findobj(Text)
            if t.get_visible() and t.get_text() and t.get_fontsize() < MIN_FONT
        }
    )
    if small:
        raise ValueError(f"text below {MIN_FONT} pt: {small}")
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = out_stem.with_suffix(".pdf"), out_stem.with_suffix(".png")
    meta = {"Title": f"Operating points ({report['domain']})", "Subject": source}
    fig.savefig(pdf, facecolor=SURFACE, metadata={**meta, "CreationDate": None})
    fig.savefig(png, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    return {"pdf": pdf, "png": png, "rule_labels": placement}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", required=True, type=Path, help="ext_decision_quality report.json")
    ap.add_argument("--out", required=True, type=Path, help="output path without extension")
    args = ap.parse_args(argv)
    if not args.report.is_file():
        raise FileNotFoundError(args.report)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    source = f"{args.report.as_posix()} sha256 {_sha256(args.report)}"
    result = draw(report, args.out, source=source)
    print(f"source: {source}")
    print(f"pdf: {result['pdf']}")
    print(f"png: {result['png']}")
    for lam, labels in result["rule_labels"].items():
        for name, (outside, blocked, bars, *_) in labels.items():
            print(f"label {lam} {name}: outside={outside} blocked={blocked} bars={bars}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
