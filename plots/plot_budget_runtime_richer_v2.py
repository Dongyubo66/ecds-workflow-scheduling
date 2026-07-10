#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Fig.7 v2: budget vs. scheduler runtime (3 panels)
- Epigenomics-100k
- Seismology-200p
- Montage-05d

v2 changes:
- MOHEFT uncertainty: box-and-whisker style glyphs (replacing thin error bars)
- ECDS: slightly thicker red dashed baseline (no error band)
- Legend: ECDS red dashed + MOHEFT green line with box-whisker marker
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch


# =========================================================
# Paths
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "results" / "study"
FIG_DIR = ROOT / "results" / "final_submission_snapshot" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BUDGET_SUMMARY = STUDY_DIR / "dynamic_budget_study_summary.csv"

INSTANCE_ORDER = [
    "Epigenomics-100k",
    "Seismology-200p",
    "Montage-05d",
]
BUDGET_ORDER = ["very_low", "low", "medium", "high", "very_high"]
PANEL_LABELS_3 = ("(a)", "(b)", "(c)")

# Deeper journal-friendly colors
COLOR_MOHEFT = "#2E8B57"   # sea green
COLOR_ECDS = "#c62828"     # deeper red


# =========================================================
# Style
# =========================================================
def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 1200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.9,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
    })


def add_panel_labels_below(axes, labels, y=-0.30) -> None:
    for ax, label in zip(axes, labels):
        ax.text(
            0.5, y, label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12.5,
            fontweight="bold",
            clip_on=False,
        )


# =========================================================
# Main plotting
# =========================================================
def main() -> None:
    configure_style()

    if not BUDGET_SUMMARY.exists():
        raise FileNotFoundError(f"Missing: {BUDGET_SUMMARY}")

    df = pd.read_csv(BUDGET_SUMMARY)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8), sharey=False)

    for ax, inst_label in zip(axes, INSTANCE_ORDER):
        sub = df[df["instance_label"] == inst_label].copy()

        # ----- MOHEFT data -----
        mo = sub[sub["scheduler"] == "MOHEFT"].copy()
        mo["__ord__"] = mo["budget_tag"].map(
            {k: i for i, k in enumerate(BUDGET_ORDER)}
        )
        mo = mo.sort_values("__ord__")

        x = mo["decision_budget_ms"].to_numpy(dtype=float)
        y = mo["runtime_mean"].to_numpy(dtype=float)
        yerr = mo["runtime_std"].fillna(0.0).to_numpy(dtype=float)

        # Compute box width proportional to x-range
        x_span = float(x.max() - x.min())
        box_width = max(x_span * 0.055, 2.0)  # ~5.5% of range, min 2.0

        # ----- MOHEFT: line connecting center values -----
        ax.plot(
            x, y,
            color=COLOR_MOHEFT,
            linewidth=2.0,
            linestyle="-",
            zorder=3,
        )

        # ----- MOHEFT: box-and-whisker glyphs at each point -----
        for xi, yi, ei in zip(x, y, yerr):
            if ei <= 0:
                # No uncertainty → just keep the marker
                continue

            # Box (filled rectangle): y ± std
            rect = Rectangle(
                (xi - box_width / 2, yi - ei),   # (left, bottom)
                box_width,                         # width
                2 * ei,                            # height
                facecolor=COLOR_MOHEFT,
                alpha=0.30,                        # semi-transparent fill
                edgecolor=COLOR_MOHEFT,
                linewidth=1.2,
                zorder=2,
                joinstyle="miter",
            )
            ax.add_patch(rect)

            # Whisker caps (horizontal ticks at top and bottom of error range)
            cap_width = box_width * 0.65
            cap_lw = 1.0
            # Bottom whisker
            ax.plot(
                [xi - cap_width / 2, xi + cap_width / 2],
                [yi - ei, yi - ei],
                color=COLOR_MOHEFT,
                linewidth=cap_lw,
                zorder=4,
            )
            # Top whisker
            ax.plot(
                [xi - cap_width / 2, xi + cap_width / 2],
                [yi + ei, yi + ei],
                color=COLOR_MOHEFT,
                linewidth=cap_lw,
                zorder=4,
            )
            # Thin vertical whisker line through the box
            ax.plot(
                [xi, xi],
                [yi - ei, yi + ei],
                color=COLOR_MOHEFT,
                linewidth=0.7,
                alpha=0.5,
                zorder=2,
            )

        # ----- MOHEFT: center triangle markers -----
        ax.scatter(
            x, y,
            marker="^",
            s=42,
            c=COLOR_MOHEFT,
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )

        # ----- ECDS reference: thicker dashed line (no error band) -----
        ref = sub[sub["budget_tag"] == "ecds_ref"]
        if len(ref) > 0:
            ref_mean = float(ref["runtime_mean"].iloc[0])
            ax.axhline(
                ref_mean,
                color=COLOR_ECDS,
                linestyle="--",
                linewidth=2.2,
                zorder=3,
            )

        # ----- Axis labels & formatting -----
        ax.set_title(inst_label, pad=7)
        ax.set_xlabel("Decision budget (ms)")
        ax.set_ylabel("Scheduler runtime (s)")
        ax.grid(True, linestyle="--", alpha=0.18)
        ax.set_axisbelow(True)

        # Hide top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Axis padding
        x_all = x
        y_all_vals = [y - yerr, y + yerr]
        if len(ref) > 0:
            y_all_vals.append(np.array([ref_mean]))
        ylo = float(np.min(np.concatenate(y_all_vals)))
        yhi = float(np.max(np.concatenate(y_all_vals)))
        y_span = max(yhi - ylo, 1e-6)
        ax.set_ylim(ylo - 0.10 * y_span, yhi + 0.12 * y_span)
        ax.set_xlim(x.min() - x_span * 0.06, x.max() + x_span * 0.06)

    # =========================
    # Legend (top center)
    # =========================
    # ECDS: red dashed line
    ecds_handle = Line2D(
        [], [],
        color=COLOR_ECDS,
        linestyle="--",
        linewidth=2.2,
        label="ECDS",
    )

    # MOHEFT: green line + box-whisker glyph
    # We build a composite: line + triangle marker + a small rectangle patch
    moheft_handle = Line2D(
        [], [],
        color=COLOR_MOHEFT,
        linestyle="-",
        linewidth=2.0,
        marker="^",
        markersize=7,
        markerfacecolor=COLOR_MOHEFT,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label="MOHEFT",
    )

    # Add a small box patch to the MOHEFT legend entry to indicate box-whisker style
    # We overlay a small green rectangle for the legend
    from matplotlib.legend_handler import HandlerTuple

    # Simpler approach: just use a thick-stem error-bar-like marker for legend
    # that doesn't look like the old thin error bar
    box_patch = Rectangle(
        (0, 0), 1, 1,
        facecolor=COLOR_MOHEFT,
        alpha=0.30,
        edgecolor=COLOR_MOHEFT,
        linewidth=1.2,
    )

    # Use a combined legend entry: line + box
    # We'll create a proxy artist that shows the green line, triangle, AND a box
    from matplotlib.patches import Patch

    # For the legend, combine a line marker and a patch
    moheft_proxy = (
        Line2D(
            [], [],
            color=COLOR_MOHEFT,
            linestyle="-",
            linewidth=2.0,
            marker="^",
            markersize=7,
            markerfacecolor=COLOR_MOHEFT,
            markeredgecolor="white",
            markeredgewidth=0.5,
        ),
        Patch(
            facecolor=COLOR_MOHEFT,
            alpha=0.30,
            edgecolor=COLOR_MOHEFT,
            linewidth=1.2,
            label="MOHEFT",
        ),
    )

    fig.legend(
        [ecds_handle, moheft_proxy],
        ["ECDS", "MOHEFT"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
    )

    add_panel_labels_below(axes, PANEL_LABELS_3)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    # =========================
    # Output
    # =========================
    pdf_path = FIG_DIR / "fig7_budget_runtime_richerv2.pdf"
    png_path = FIG_DIR / "fig7_budget_runtime_richerv2.png"
    svg_path = FIG_DIR / "fig7_budget_runtime_richerv2.svg"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {svg_path}")


if __name__ == "__main__":
    main()
