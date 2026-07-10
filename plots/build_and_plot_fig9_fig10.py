# -*- coding: utf-8 -*-
"""
Build + plot Fig.9 and Fig.10 from:
    results/study/dynamic_budget_study_raw.csv
    results/study/dynamic_pareto_study_raw.csv

Goal:
    - Compute IGD / EPS_ADD for each MOHEFT budget setting
    - Compute ECDS reference IGD / EPS_ADD
    - Plot Fig.9 (IGD) and Fig.10 (EPS_ADD)

Reference front source:
    dynamic_pareto_study_raw.csv

Output:
    results/final_submission_snapshot/figures/fig9_budget_igd.png
    results/final_submission_snapshot/figures/fig10_budget_eps_add.png
    results/final_submission_snapshot/tables/fig9_fig10_budget_quality_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Paths
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
RAW_BUDGET = ROOT / "results" / "study" / "dynamic_budget_study_raw.csv"
RAW_PARETO = ROOT / "results" / "study" / "dynamic_pareto_study_raw.csv"

OUT_DIR_FIG = ROOT / "results" / "final_submission_snapshot" / "figures"
OUT_DIR_TAB = ROOT / "results" / "final_submission_snapshot" / "tables"

OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)
OUT_DIR_TAB.mkdir(parents=True, exist_ok=True)


# =========================================================
# Figure order / labels
# =========================================================
INSTANCE_ORDER = [
    "epigenomics-chameleon-hep-1seq-100k-001",
    "seismology-chameleon-200p-001",
    "montage-chameleon-2mass-05d-001",
]

INSTANCE_LABELS = {
    "epigenomics-chameleon-hep-1seq-100k-001": "Epigenomics-100k",
    "seismology-chameleon-200p-001": "Seismology-200p",
    "montage-chameleon-2mass-05d-001": "Montage-05d",
}

COLOR_MOHEFT = "#1B7F3B"   # green
COLOR_ECDS = "#C62828"     # red
GRID_COLOR = "#D9D9D9"


# =========================================================
# Matplotlib style
# =========================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 160,
    "savefig.dpi": 300,
})


# =========================================================
# Helpers
# =========================================================
def instance_label(instance: str) -> str:
    return INSTANCE_LABELS.get(instance, instance)


def load_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def prettify_axes(ax):
    ax.grid(True, axis="both", linestyle="--", linewidth=0.7, color=GRID_COLOR, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


# =========================================================
# Pareto utilities
# =========================================================
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    # minimization
    return np.all(a <= b) and np.any(a < b)


def nondominated_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    keep = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not keep[i]:
            continue
        for j in range(len(points)):
            if i == j or not keep[j]:
                continue
            if dominates(points[j], points[i]):
                keep[i] = False
                break
    return points[keep]


def normalize_points(points: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    span = np.where((maxs - mins) <= 1e-12, 1.0, (maxs - mins))
    return (points - mins) / span


def igd(front: np.ndarray, ref_front: np.ndarray) -> float:
    """
    Inverted Generational Distance
    """
    if len(front) == 0 or len(ref_front) == 0:
        return np.nan
    dists = []
    for r in ref_front:
        d = np.linalg.norm(front - r, axis=1)
        dists.append(np.min(d))
    return float(np.mean(dists))


def eps_additive(front: np.ndarray, ref_front: np.ndarray) -> float:
    """
    Additive epsilon indicator (minimization)
    """
    if len(front) == 0 or len(ref_front) == 0:
        return np.nan
    eps_vals = []
    for r in ref_front:
        eps_r = np.inf
        for f in front:
            eps_r = min(eps_r, np.max(f - r))
        eps_vals.append(eps_r)
    return float(np.max(eps_vals))


# =========================================================
# Build quality summary from budget raw + pareto raw
# =========================================================
def build_quality_summary_from_raw(
    df_budget_raw: pd.DataFrame,
    df_pareto_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Use dynamic_pareto_study_raw.csv to build reference Pareto front
    for each target instance.

    Then for each:
        instance x scheduler x budget_tag
    in dynamic_budget_study_raw.csv,
    compute:
        IGD, EPS_ADD
    against that reference front.

    Objectives used:
        makespan, total_carbon, brown_energy
    all treated as minimization objectives.
    """
    need_budget = [
        "workflow", "family", "instance", "scheduler", "budget_tag",
        "decision_budget_ms", "makespan", "total_carbon", "brown_energy"
    ]
    miss_budget = [c for c in need_budget if c not in df_budget_raw.columns]
    if miss_budget:
        raise ValueError(f"Budget raw CSV missing columns: {miss_budget}")

    need_pareto = [
        "workflow", "family", "instance", "scheduler",
        "makespan", "total_carbon", "brown_energy"
    ]
    miss_pareto = [c for c in need_pareto if c not in df_pareto_raw.columns]
    if miss_pareto:
        raise ValueError(f"Pareto raw CSV missing columns: {miss_pareto}")

    dfb = df_budget_raw.copy()
    dfp = df_pareto_raw.copy()

    dfb.columns = [str(c).strip() for c in dfb.columns]
    dfp.columns = [str(c).strip() for c in dfp.columns]

    dfb["scheduler"] = dfb["scheduler"].astype(str).str.upper()
    dfp["scheduler"] = dfp["scheduler"].astype(str).str.upper()
    dfb["budget_tag"] = dfb["budget_tag"].astype(str)

    dfb = ensure_numeric(dfb, ["decision_budget_ms", "makespan", "total_carbon", "brown_energy"])
    dfp = ensure_numeric(dfp, ["makespan", "total_carbon", "brown_energy"])

    # only keep three representative instances
    dfb = dfb[dfb["instance"].isin(INSTANCE_ORDER)].copy()
    dfp = dfp[dfp["instance"].isin(INSTANCE_ORDER)].copy()

    dfb["instance_label"] = dfb["instance"].map(instance_label)
    dfp["instance_label"] = dfp["instance"].map(instance_label)

    rows = []

    for inst in INSTANCE_ORDER:
        pb = dfb[dfb["instance"] == inst].copy()
        pp = dfp[dfp["instance"] == inst].copy()

        if pb.empty or pp.empty:
            continue

        # reference front from richer pareto raw
        ref_pts_raw = pp[["makespan", "total_carbon", "brown_energy"]].dropna().to_numpy(dtype=float)
        if len(ref_pts_raw) == 0:
            continue

        # normalization range from union of budget raw + pareto raw
        union_pts = np.vstack([
            pb[["makespan", "total_carbon", "brown_energy"]].dropna().to_numpy(dtype=float),
            ref_pts_raw
        ])
        mins = union_pts.min(axis=0)
        maxs = union_pts.max(axis=0)

        ref_pts = normalize_points(ref_pts_raw, mins, maxs)
        ref_front = nondominated_points(ref_pts)

        for (scheduler, budget_tag), g in pb.groupby(["scheduler", "budget_tag"]):
            pts_raw = g[["makespan", "total_carbon", "brown_energy"]].dropna().to_numpy(dtype=float)
            if len(pts_raw) == 0:
                continue

            pts = normalize_points(pts_raw, mins, maxs)
            front = nondominated_points(pts)

            rows.append({
                "family": g["family"].iloc[0],
                "workflow": g["workflow"].iloc[0],
                "instance": g["instance"].iloc[0],
                "instance_label": g["instance_label"].iloc[0],
                "scheduler": scheduler,
                "budget_tag": budget_tag,
                "decision_budget_ms": g["decision_budget_ms"].iloc[0] if "decision_budget_ms" in g.columns else np.nan,
                "IGD": igd(front, ref_front),
                "EPS_ADD": eps_additive(front, ref_front),
                "n_points": len(pts_raw),
            })

    quality = pd.DataFrame(rows)

    summary = (
        quality.groupby(
            ["family", "workflow", "instance", "instance_label", "scheduler", "budget_tag", "decision_budget_ms"],
            as_index=False
        )
        .agg(
            IGD_mean=("IGD", "mean"),
            IGD_std=("IGD", "std"),
            EPS_ADD_mean=("EPS_ADD", "mean"),
            EPS_ADD_std=("EPS_ADD", "std"),
            n_points_mean=("n_points", "mean"),
            n_runs=("IGD", "count"),
        )
    )

    return summary


def split_curve_and_ref(summary: pd.DataFrame):
    """
    Split summary into:
        df_curve: MOHEFT budgets
        df_ref:   ECDS reference
    """
    s = summary.copy()
    s["scheduler"] = s["scheduler"].astype(str).str.upper()

    df_curve = s[s["scheduler"] == "MOHEFT"].copy()
    df_ref = s[
        (s["scheduler"] == "ECDS") | (s["budget_tag"].astype(str).str.lower() == "ecds_ref")
    ].copy()

    if not df_ref.empty:
        df_ref = df_ref.sort_values(["instance"]).groupby("instance", as_index=False).first()
        df_ref["scheduler"] = "ECDS"
        df_ref["budget_tag"] = "ecds_ref"

    df_curve["instance_order"] = df_curve["instance"].apply(lambda x: INSTANCE_ORDER.index(x) if x in INSTANCE_ORDER else 999)
    df_curve = df_curve.sort_values(["instance_order", "decision_budget_ms"]).drop(columns=["instance_order"])

    if not df_ref.empty:
        df_ref["instance_order"] = df_ref["instance"].apply(lambda x: INSTANCE_ORDER.index(x) if x in INSTANCE_ORDER else 999)
        df_ref = df_ref.sort_values(["instance_order"]).drop(columns=["instance_order"])

    return df_curve, df_ref


# =========================================================
# Plotting
# =========================================================
def add_panel_labels_below(axes, labels, y=-0.30):
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


def plot_budget_metric(
    df_curve: pd.DataFrame,
    df_ref: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    y_label: str,
    out_path: Path,
    panel_labels: tuple[str, ...] | None = None,
):
    n_panels = len(INSTANCE_ORDER)
    fig, axes = plt.subplots(1, n_panels, figsize=(12.8, 4.8))

    legend_handles = []
    legend_labels = []

    for i, (ax, inst) in enumerate(zip(axes, INSTANCE_ORDER)):
        d = df_curve[df_curve["instance"] == inst].copy()
        r = df_ref[df_ref["instance"] == inst].copy()

        if d.empty:
            ax.set_visible(False)
            continue

        d = d.sort_values("decision_budget_ms")
        x = d["decision_budget_ms"].to_numpy(dtype=float)
        y = d[metric_mean].to_numpy(dtype=float)
        yerr = d[metric_std].fillna(0.0).to_numpy(dtype=float) if metric_std in d.columns else None

        eb = ax.errorbar(
            x, y, yerr=yerr,
            fmt="-o",
            color=COLOR_MOHEFT,
            linewidth=2.0,
            markersize=5.8,
            markerfacecolor=COLOR_MOHEFT,
            markeredgecolor="white",
            markeredgewidth=0.8,
            ecolor=COLOR_MOHEFT,
            elinewidth=1.3,
            capsize=4,
            capthick=1.1,
            alpha=0.95,
            zorder=3,
        )

        line = None
        if not r.empty:
            ref_mean = float(r.iloc[0][metric_mean])
            ref_std = float(r.iloc[0][metric_std]) if metric_std in r.columns and pd.notna(r.iloc[0][metric_std]) else 0.0

            line = ax.axhline(
                ref_mean,
                color=COLOR_ECDS,
                linestyle="--",
                linewidth=2.0,
                alpha=0.95,
                zorder=2,
            )

            if ref_std > 0:
                ax.fill_between(
                    [x.min(), x.max()],
                    [ref_mean - ref_std, ref_mean - ref_std],
                    [ref_mean + ref_std, ref_mean + ref_std],
                    color=COLOR_ECDS,
                    alpha=0.10,
                    zorder=1,
                )

        ax.set_title(instance_label(inst), pad=12)
        ax.set_xlabel("Decision budget (ms)")
        ax.set_ylabel(y_label)
        ax.set_xticks(x)

        y_all = y.copy()
        if not r.empty:
            y_all = np.concatenate([y_all, np.array([ref_mean])])

        ymin = np.nanmin(y_all)
        ymax = np.nanmax(y_all)
        span = max(ymax - ymin, 1e-6)
        ax.set_ylim(ymin - 0.10 * span, ymax + 0.12 * span)

        prettify_axes(ax)

        if not legend_handles:
            legend_handles.append(eb[0])
            legend_labels.append("MOHEFT")
            if line is not None:
                legend_handles.append(line)
                legend_labels.append("ECDS ref")

    fig.legend(
        legend_handles, legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02)
    )

    if panel_labels:
        add_panel_labels_below(axes, panel_labels)

    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    if not RAW_BUDGET.exists():
        raise FileNotFoundError(f"Missing raw budget CSV: {RAW_BUDGET}")
    if not RAW_PARETO.exists():
        raise FileNotFoundError(f"Missing raw pareto CSV: {RAW_PARETO}")

    print(f"[INFO] Using raw budget file: {RAW_BUDGET}")
    print(f"[INFO] Using raw pareto file: {RAW_PARETO}")

    df_budget_raw = load_csv_auto(RAW_BUDGET)
    df_pareto_raw = load_csv_auto(RAW_PARETO)

    summary = build_quality_summary_from_raw(df_budget_raw, df_pareto_raw)
    df_curve, df_ref = split_curve_and_ref(summary)

    summary_path = OUT_DIR_TAB / "fig9_fig10_budget_quality_summary.csv"
    curve_used_path = OUT_DIR_TAB / "fig9_fig10_budget_curve_used.csv"
    ref_used_path = OUT_DIR_TAB / "fig9_fig10_ecds_ref_used.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    df_curve.to_csv(curve_used_path, index=False, encoding="utf-8-sig")
    df_ref.to_csv(ref_used_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved quality summary: {summary_path}")
    print(f"[OK] Saved curve table:     {curve_used_path}")
    print(f"[OK] Saved ref table:       {ref_used_path}")

    fig9_path = OUT_DIR_FIG / "fig9_budget_igd.png"
    fig10_path = OUT_DIR_FIG / "fig10_budget_eps_add.png"

    plot_budget_metric(
        df_curve=df_curve,
        df_ref=df_ref,
        metric_mean="IGD_mean",
        metric_std="IGD_std",
        y_label="IGD (normalized)",
        out_path=fig9_path,
        panel_labels=("(a)", "(b)", "(c)"),
    )

    plot_budget_metric(
        df_curve=df_curve,
        df_ref=df_ref,
        metric_mean="EPS_ADD_mean",
        metric_std="EPS_ADD_std",
        y_label="Additive epsilon (normalized)",
        out_path=fig10_path,
        panel_labels=("(a)", "(b)", "(c)"),
    )

    print(f"[OK] Saved Fig.9:  {fig9_path}")
    print(f"[OK] Saved Fig.10: {fig10_path}")


if __name__ == "__main__":
    main()
