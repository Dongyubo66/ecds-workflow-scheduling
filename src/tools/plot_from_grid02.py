import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

DEFAULT_METRICS = [
    "makespan",
    "total_energy",
    "total_carbon",
    "green_ratio",
    "green_energy",
    "brown_energy",
    "avg_utilization",
    "flowtime_sum",
]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def _to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def aggregate_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["workflow", "instance", "scheduler", "abl", "w1", "w2", "w3", "w4"]
    keep = [m for m in DEFAULT_METRICS if m in df.columns]

    if not keep:
        return pd.DataFrame()

    g = df.groupby(keys, dropna=False)
    mean_df = g[keep].mean(numeric_only=True).reset_index()
    std_df = g[keep].std(numeric_only=True).reset_index().rename(columns={c: f"{c}_std" for c in keep})
    out = mean_df.merge(std_df, on=keys, how="left")

    numeric_cols = ["w1", "w2", "w3", "w4"] + keep + [f"{c}_std" for c in keep]
    out = _to_num(out, numeric_cols)
    return out


def plot_makespan_green_ratio_comparison(agg: pd.DataFrame, instance: str, outdir: Path):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    df = agg[agg["instance"] == instance].copy()
    df = df.sort_values("w4")

    ax1.set_xlabel("w4 (Green energy weight)")
    ax1.set_ylabel("Makespan (s)", color="tab:blue")

    for sched, sub in df.groupby("scheduler"):
        ax1.plot(sub["w4"], sub["makespan"], label=f"{sched} Makespan", marker="o")

    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title(f"{instance} | Makespan vs Green Ratio")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Green Ratio", color="tab:green")

    for sched, sub in df.groupby("scheduler"):
        ax2.plot(sub["w4"], sub["green_ratio"], label=f"{sched} Green Ratio", linestyle="--", marker="x")

    ax2.tick_params(axis="y", labelcolor="tab:green")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    outdir.mkdir(parents=True, exist_ok=True)
    fpath = outdir / f"{instance}__makespan_vs_green_ratio.png"
    fig.tight_layout()
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fpath


def plot_pareto_comparison(agg: pd.DataFrame, instance: str, outdir: Path, x_metric="makespan",
                           y_metric="total_carbon"):
    fig, ax = plt.subplots(figsize=(10, 6))
    df = agg[agg["instance"] == instance].copy()
    if df.empty:
        return None

    for sched, sub in df.groupby("scheduler"):
        ax.scatter(sub[x_metric], sub[y_metric], label=f"{sched}", marker="o")

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"{instance} | Pareto: {x_metric} vs {y_metric}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    outdir.mkdir(parents=True, exist_ok=True)
    fpath = outdir / f"{instance}__pareto__{x_metric}_vs_{y_metric}.png"
    fig.tight_layout()
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fpath


def plot_ablation_comparison(agg: pd.DataFrame, instance: str, outdir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for tag in agg["abl"].unique():
        sub = agg[agg["abl"] == tag]
        ax.plot(sub["w4"], sub["makespan"], marker="o", label=f"{tag} - Makespan")
        ax.plot(sub["w4"], sub["total_energy"], marker="x", label=f"{tag} - Energy")

    ax.set_xlabel("w4")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    outdir.mkdir(parents=True, exist_ok=True)
    fpath = outdir / f"{instance}__ablation_comparison.png"
    fig.tight_layout()
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="results/plotsV2")
    ap.add_argument("--pareto", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = _clean_columns(df)
    df = _to_num(df, ["w1", "w2", "w3", "w4", "seed"] + DEFAULT_METRICS)

    agg = aggregate_mean_std(df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    insts = sorted(agg["instance"].dropna().unique().tolist())
    saved = []

    for inst in insts:
        plot_makespan_green_ratio_comparison(agg, inst, outdir)

        if args.pareto:
            plot_pareto_comparison(agg, inst, outdir, x_metric="makespan", y_metric="total_carbon")

        plot_ablation_comparison(agg, inst, outdir)

        saved.append(f"{inst}__ablation_comparison.png")

    print("-" * 80)
    print("Loaded:", Path(args.csv).resolve())
    print("Saved figures to:", outdir.resolve())
    print("Total figures:", len(saved))
    if saved:
        print("Example:", saved[0])


if __name__ == "__main__":
    main()
