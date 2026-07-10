from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# paths
# =========================================================
ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "plots") else Path(__file__).resolve().parent
STUDY_DIR = ROOT / "results" / "study"
FIG_DIR = ROOT / "results" / "final_submission_snapshot" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RAW_BUDGET = STUDY_DIR / "dynamic_budget_study_raw.csv"
OUT_SUMMARY = STUDY_DIR / "dynamic_budget_quality_summary_v2.csv"

# representative instances consistent with earlier discussion
TARGETS = [
    ("montage/chameleon-cloud", "montage-chameleon-2mass-05d-001", "Montage-05d"),
    ("epigenomics/chameleon-cloud", "epigenomics-chameleon-hep-1seq-100k-001", "Epigenomics-100k"),
    ("seismology/chameleon-cloud", "seismology-chameleon-200p-001", "Seismology-200p"),
]

# map budget tags to numeric x-axis
BUDGET_TAG_TO_MS = {
    "very_low": 5.0,
    "low": 20.0,
    "medium": 50.0,
    "high": 100.0,
    "very_high": 200.0,
    # backward compatibility
    "ecds_ref": np.nan,
}

PALETTE = {
    "MOHEFT": "#1B7F3B",   # deep green
    "ECDS": "#D62728",     # red
}


# =========================================================
# small Pareto helpers
# =========================================================
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a <= b) and np.any(a < b)


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if dominates(points[j], points[i]):
                keep[i] = False
                break
    return keep


def hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """
    2D minimization HV. Points assumed normalized to [0, 1]-ish space.
    """
    if len(points) == 0:
        return 0.0

    nd = points[nondominated_mask(points)]
    nd = nd[np.argsort(nd[:, 0])]  # ascending x

    hv = 0.0
    prev_y = ref[1]
    for x, y in nd:
        width = max(ref[0] - x, 0.0)
        height = max(prev_y - y, 0.0)
        hv += width * height
        prev_y = min(prev_y, y)
    return hv


def igd(front: np.ndarray, ref_front: np.ndarray) -> float:
    if len(front) == 0 or len(ref_front) == 0:
        return np.nan
    dists = []
    for r in ref_front:
        d = np.linalg.norm(front - r, axis=1)
        dists.append(np.min(d))
    return float(np.mean(dists))


def eps_additive(front: np.ndarray, ref_front: np.ndarray) -> float:
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
# data prep
# =========================================================
def short_label(workflow: str, instance: str) -> str:
    for wf, inst, lab in TARGETS:
        if workflow == wf and instance == inst:
            return lab
    return instance


def load_raw_budget() -> pd.DataFrame:
    if not RAW_BUDGET.exists():
        raise FileNotFoundError(f"Missing raw budget CSV: {RAW_BUDGET}")
    df = pd.read_csv(RAW_BUDGET)

    need = [
        "scheduler", "family", "workflow", "instance", "seed",
        "budget_tag", "decision_budget_ms",
        "makespan", "total_carbon", "brown_energy"
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"dynamic_budget_study_raw.csv missing columns: {miss}")

    # only keep target instances
    keep = []
    for wf, inst, _ in TARGETS:
        keep.append((wf, inst))
    mask = df.apply(lambda r: (r["workflow"], r["instance"]) in keep, axis=1)
    df = df[mask].copy()

    df["scheduler"] = df["scheduler"].astype(str).str.upper()
    df["instance_label"] = df.apply(lambda r: short_label(r["workflow"], r["instance"]), axis=1)

    # numeric x for plotting
    df["budget_ms"] = df["budget_tag"].map(BUDGET_TAG_TO_MS)
    df.loc[df["budget_tag"] == "ecds_ref", "budget_ms"] = np.nan

    return df


def build_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each instance, normalize objective points using the union of all budget-study points.
    Then compute HV / IGD / EPS for each scheduler x budget_tag group.
    Objectives: makespan, total_carbon, brown_energy.
    """
    rows = []

    for (_, inst_label), inst_df in df.groupby(["workflow", "instance_label"]):
        # normalization baseline from all points of this instance
        raw_pts = inst_df[["makespan", "total_carbon", "brown_energy"]].to_numpy(dtype=float)
        mins = raw_pts.min(axis=0)
        maxs = raw_pts.max(axis=0)
        spans = np.where(maxs - mins <= 1e-12, 1.0, maxs - mins)

        def norm_points(sub: pd.DataFrame) -> np.ndarray:
            pts = sub[["makespan", "total_carbon", "brown_energy"]].to_numpy(dtype=float)
            return (pts - mins) / spans

        # global reference front
        global_front = norm_points(inst_df)
        global_front = global_front[nondominated_mask(global_front)]

        for (sched, budget_tag), g in inst_df.groupby(["scheduler", "budget_tag"]):
            pts3 = norm_points(g)
            front3 = pts3[nondominated_mask(pts3)]

            # 3D metrics could be added; for plotting Fig.9/10 we use a 2D projection
            # choose makespan vs brown_energy to stay aligned with Pareto storyline
            pts2 = pts3[:, [0, 2]]
            front2 = pts2[nondominated_mask(pts2)]

            ref2 = np.array([1.05, 1.05], dtype=float)
            global_front2 = global_front[:, [0, 2]]

            hv_val = hypervolume_2d(front2, ref2)
            igd_val = igd(front2, global_front2)
            eps_val = eps_additive(front2, global_front2)

            rows.append({
                "workflow": g["workflow"].iloc[0],
                "family": g["family"].iloc[0],
                "instance": g["instance"].iloc[0],
                "instance_label": g["instance_label"].iloc[0],
                "scheduler": sched,
                "budget_tag": budget_tag,
                "budget_ms": g["budget_ms"].iloc[0],
                "n_points": len(g),
                "HV": hv_val,
                "IGD": igd_val,
                "EPS_ADD": eps_val,
            })

    out = pd.DataFrame(rows)

    # aggregate across repeated seeds / points already represented inside each group:
    # here each group is already a cloud of repeated runs under same budget.
    # we still summarize by instance/scheduler/budget in case repeated loading structure changes.
    summary = (
        out.groupby(
            ["workflow", "family", "instance", "instance_label", "scheduler", "budget_tag", "budget_ms"],
            as_index=False
        )[["HV", "IGD", "EPS_ADD", "n_points"]]
        .mean()
    )

    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8")
    print(f"[OK] Saved budget quality summary: {OUT_SUMMARY}")
    return summary


# =========================================================
# plotting
# =========================================================
def configure_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.0,
    })


def plot_budget_metric(summary: pd.DataFrame, metric: str, ylabel: str, output_stem: str):
    inst_order = ["Epigenomics-100k", "Seismology-200p", "Montage-05d"]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=False)
    mo_color = PALETTE["MOHEFT"]
    ecds_color = PALETTE["ECDS"]

    for ax, inst_label in zip(axes, inst_order):
        sub = summary[summary["instance_label"] == inst_label].copy()

        mo = sub[sub["scheduler"] == "MOHEFT"].copy()
        mo = mo[mo["budget_tag"] != "ecds_ref"].copy()
        mo = mo.sort_values("budget_ms")

        ecds = sub[sub["scheduler"] == "ECDS"].copy()
        ecds = ecds[ecds["budget_tag"] == "ecds_ref"].copy()

        # MOHEFT line
        ax.plot(
            mo["budget_ms"],
            mo[metric],
            marker="o",
            markersize=6.5,
            linewidth=2.2,
            color=mo_color,
            label="MOHEFT",
            zorder=3,
        )

        # ECDS reference band
        if not ecds.empty:
            ref = float(ecds[metric].iloc[0])
            # use a narrow band for readability; raw-derived ref is stable here
            band = max(0.01 * abs(ref), 0.005)
            ax.axhline(ref, color=ecds_color, linestyle="--", linewidth=2.0, label="ECDS ref", zorder=2)
            ax.fill_between(
                [mo["budget_ms"].min(), mo["budget_ms"].max()],
                [ref - band, ref - band],
                [ref + band, ref + band],
                color=ecds_color,
                alpha=0.10,
                zorder=1
            )

        ax.set_title(inst_label, pad=8)
        ax.set_xlabel("Decision budget (ms)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([5, 20, 50, 100, 200])
        ax.set_xlim(0, 205)
        ax.grid(axis="both", linestyle="--", alpha=0.25)

        # nicer y padding
        y = mo[metric].to_numpy(dtype=float)
        if not ecds.empty:
            y = np.concatenate([y, np.array([ref])])
        ylo = float(np.min(y))
        yhi = float(np.max(y))
        span = max(yhi - ylo, 1e-6)
        ax.set_ylim(ylo - 0.08 * span, yhi + 0.12 * span)

    handles, labels = axes[0].get_legend_handles_labels()
    # unique legend
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)

    fig.legend(uniq_h, uniq_l, ncol=2, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    pdf_path = FIG_DIR / f"{output_stem}.pdf"
    png_path = FIG_DIR / f"{output_stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")


def main():
    configure_style()
    df = load_raw_budget()
    summary = build_quality_summary(df)

    plot_budget_metric(
        summary=summary,
        metric="IGD",
        ylabel="IGD",
        output_stem="fig9_budget_igd_richer_v2",
    )

    plot_budget_metric(
        summary=summary,
        metric="EPS_ADD",
        ylabel="Additive epsilon",
        output_stem="fig10_budget_eps_richer_v2",
    )


if __name__ == "__main__":
    main()