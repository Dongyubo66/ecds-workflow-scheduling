#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Fig.6 v6: dynamic Pareto fronts (2 panels only)
- Epigenomics-100k
- Seismology-200p

v6 changes (scatter-only, deeper colors, marker-only legend):
- Remove all connecting lines between Pareto front points
- All methods expressed as scatter points only
- Deeper, more saturated colors for journal print readability
- Legend shows marker-only (no line segments)
- Slightly larger markers with thin edgecolors
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================================================
# Paths
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "results" / "study" / "dynamic_pareto_study_raw.csv"
OUT_DIR = ROOT / "results" / "final_submission_snapshot" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_INSTANCES = [
    "epigenomics-chameleon-hep-1seq-100k-001",
    "seismology-chameleon-200p-001",
]
PANEL_LABELS_2 = ("(a)", "(b)")

SCHED_ORDER = ["HEFT", "GREENHEFT", "MOHEFT", "ECDS"]

# Deeper, low-saturation journal-friendly colors
SCHED_COLORS = {
    "HEFT":       "#1565c0",  # deeper blue
    "GREENHEFT":  "#e65100",  # deeper orange
    "MOHEFT":     "#2e7d32",  # deeper green
    "ECDS":       "#c62828",  # deeper red
}

SCHED_MARKERS = {
    "HEFT":       "o",
    "GREENHEFT":  "s",
    "MOHEFT":     "^",
    "ECDS":       "D",
}

# ---------- visual tuning (v6: scatter-only, deeper, more opaque) ----------
# Raw background point cloud
RAW_SIZE = 36           # slightly larger than before
RAW_ALPHA = 0.30        # more opaque (was 0.20–0.24)
RAW_EDGE_WIDTH = 0.4
RAW_EDGE_COLOR = "white"

# Pareto front scatter points (no lines)
FRONT_SIZE = 72         # larger, more visible
FRONT_ALPHA = 0.95      # nearly opaque
FRONT_EDGE_WIDTH = 0.6
FRONT_EDGE_COLOR = "#333333"  # thin dark border for print clarity

# ECDS slight emphasis (slightly larger)
ECDS_FRONT_SIZE = 80
ECDS_RAW_SIZE = 40


def short_instance_label(instance: str) -> str:
    s = str(instance).lower()
    if "epigenomics" in s and "100k" in s:
        return "Epigenomics-100k"
    if "seismology" in s and "200p" in s:
        return "Seismology-200p"
    return instance


def add_panel_labels_below(axes, labels, y=-0.26):
    for ax, label in zip(axes, labels):
        ax.text(
            0.5, y, label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            clip_on=False,
        )


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.0,
        "grid.linestyle": "--",
        "grid.alpha": 0.18,
    })


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    """Min-min nondominated mask for 2D points."""
    n = points.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (
                points[j, 0] <= points[i, 0]
                and points[j, 1] <= points[i, 1]
                and (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1])
            ):
                mask[i] = False
                break
    return mask


def frontier_points(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    pts = df[[xcol, ycol]].to_numpy(dtype=float)
    if len(pts) == 0:
        return df.iloc[[]].copy()
    mask = nondominated_mask(pts)
    out = df.loc[mask].copy()
    out = out.sort_values(by=[xcol, ycol], ascending=[True, True])
    return out


def main():
    configure_style()

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing raw pareto file: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)

    needed = ["instance", "scheduler", "makespan", "brown_energy"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"dynamic_pareto_study_raw.csv missing columns: {missing}")

    df["scheduler"] = df["scheduler"].astype(str).str.upper()
    df = df[df["scheduler"].isin(SCHED_ORDER)].copy()
    df = df[df["instance"].isin(REPRESENTATIVE_INSTANCES)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.0))
    axes = np.atleast_1d(axes)

    legend_handles = []
    legend_labels = []

    for idx, (ax, inst) in enumerate(zip(axes, REPRESENTATIVE_INSTANCES)):
        sub = df[df["instance"] == inst].copy()
        title = short_instance_label(inst)

        for sched in SCHED_ORDER:
            sdf = sub[sub["scheduler"] == sched].copy()
            if sdf.empty:
                continue

            color = SCHED_COLORS[sched]
            marker = SCHED_MARKERS[sched]

            # -------------------------------------------------
            # 1) Raw cloud: deeper, more opaque, slightly larger
            # -------------------------------------------------
            raw_s = ECDS_RAW_SIZE if sched == "ECDS" else RAW_SIZE

            ax.scatter(
                sdf["makespan"],
                sdf["brown_energy"],
                s=raw_s,
                alpha=RAW_ALPHA,
                color=color,
                edgecolors=RAW_EDGE_COLOR,
                linewidths=RAW_EDGE_WIDTH,
                zorder=1,
            )

            # -------------------------------------------------
            # 2) Pareto front: scatter-only, NO lines
            # -------------------------------------------------
            fdf = frontier_points(sdf, "makespan", "brown_energy")
            if not fdf.empty:
                front_s = ECDS_FRONT_SIZE if sched == "ECDS" else FRONT_SIZE

                ax.scatter(
                    fdf["makespan"],
                    fdf["brown_energy"],
                    s=front_s,
                    marker=marker,
                    c=color,
                    alpha=FRONT_ALPHA,
                    edgecolors=FRONT_EDGE_COLOR,
                    linewidths=FRONT_EDGE_WIDTH,
                    zorder=4 if sched == "ECDS" else 3,
                )

                # Build legend entry (marker-only, no line)
                if idx == 0:
                    legend_handles.append(
                        Line2D(
                            [], [],
                            marker=marker,
                            markersize=7.5,
                            markerfacecolor=color,
                            markeredgecolor=FRONT_EDGE_COLOR,
                            markeredgewidth=0.6,
                            linestyle="none",
                            label=sched,
                        )
                    )
                    legend_labels.append(sched)

        ax.set_title(title, pad=10)
        ax.set_xlabel("Makespan (s)")
        ax.set_ylabel("Brown energy (model units)")
        ax.grid(True, linestyle="--", alpha=0.18)

        # Hide top/right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Natural padding
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xpad = (xmax - xmin) * 0.07 if xmax > xmin else 1.0
        ypad = (ymax - ymin) * 0.08 if ymax > ymin else 1.0
        ax.set_xlim(xmin - 0.15 * xpad, xmax + xpad)
        ax.set_ylim(ymin - 0.12 * ypad, ymax + ypad)

    # =========================
    # Legend: marker-only, no lines
    # =========================
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        handlelength=0.0,
        handletextpad=0.6,
        columnspacing=1.8,
    )

    add_panel_labels_below(axes, PANEL_LABELS_2)
    fig.tight_layout(rect=[0, 0.08, 1, 0.88])

    # =========================
    # Output: high-quality formats
    # =========================
    pdf_path = OUT_DIR / "fig6_dynamic_pareto_projection_richer_v6.pdf"
    png_path = OUT_DIR / "fig6_dynamic_pareto_projection_richer_v6.png"
    svg_path = OUT_DIR / "fig6_dynamic_pareto_projection_richer_v6.svg"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {svg_path}")


if __name__ == "__main__":
    main()
