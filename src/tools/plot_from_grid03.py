import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    # 只把数值列转成数值：不要碰 workflow/instance/scheduler/abl
    numeric_cols = ["w1", "w2", "w3", "w4"] + keep + [f"{c}_std" for c in keep]
    out = _to_num(out, numeric_cols)
    return out

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除重复点，确保每个w4对应唯一的Makespan和Green Ratio。
    """
    # Drop duplicates based on 'w4', 'scheduler', 'Makespan', 'green_ratio'
    return df.drop_duplicates(subset=["w4", "scheduler", "makespan", "green_ratio"])

def plot_metric_vs_w4(agg: pd.DataFrame, instance: str, metric: str, outdir: Path) -> Optional[Path]:
    if agg.empty or metric not in agg.columns:
        return None

    sub = agg[agg["instance"] == instance].copy()
    if sub.empty:
        return None

    # 去除重复点
    sub = remove_duplicates(sub)

    sub = sub.sort_values(["scheduler", "w4"])
    fig = plt.figure()
    for sched in sorted(sub["scheduler"].dropna().unique().tolist()):
        ss = sub[sub["scheduler"] == sched].sort_values("w4")
        if ss.empty:
            continue
        x = ss["w4"].astype(float).to_list()
        y = ss[metric].astype(float).to_list()
        plt.plot(x, y, marker="o", label=str(sched))

    plt.xlabel("w4")
    plt.ylabel(metric)
    plt.title(f"{instance} | {metric} vs w4")
    plt.grid(True, alpha=0.3)
    plt.legend()

    outdir.mkdir(parents=True, exist_ok=True)
    fpath = outdir / f"{instance}__{metric}__vs_w4.png"
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fpath

def plot_pareto(agg: pd.DataFrame, instance: str, outdir: Path, x_metric="makespan", y_metric="total_carbon") -> Optional[Path]:
    if agg.empty or x_metric not in agg.columns or y_metric not in agg.columns:
        return None

    sub = agg[agg["instance"] == instance].copy()
    if sub.empty:
        return None

    # 去除重复点
    sub = remove_duplicates(sub)

    fig = plt.figure()
    for sched in sorted(sub["scheduler"].dropna().unique().tolist()):
        ss = sub[sub["scheduler"] == sched]
        plt.scatter(ss[x_metric].astype(float), ss[y_metric].astype(float), label=str(sched))

    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(f"{instance} | Pareto ({x_metric} vs {y_metric})")
    plt.grid(True, alpha=0.3)
    plt.legend()

    outdir.mkdir(parents=True, exist_ok=True)
    fpath = outdir / f"{instance}__pareto__{x_metric}_vs_{y_metric}.png"
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fpath

def plot_pareto_comparisons(agg: pd.DataFrame, instance: str, outdir: Path) -> Optional[List[str]]:
    """
    绘制多种Pareto对比图：
    - Makespan vs Total Carbon
    - Makespan vs Total Energy
    - Makespan vs Green Ratio
    """
    result_paths = []

    # 绘制 Makespan vs Total Carbon 对比图
    p = plot_pareto(agg, instance, outdir, x_metric="makespan", y_metric="total_carbon")
    if p:
        result_paths.append(p)

    # 绘制 Makespan vs Total Energy 对比图
    p = plot_pareto(agg, instance, outdir, x_metric="makespan", y_metric="total_energy")
    if p:
        result_paths.append(p)

    # 绘制 Makespan vs Green Ratio 对比图
    p = plot_pareto(agg, instance, outdir, x_metric="makespan", y_metric="green_ratio")
    if p:
        result_paths.append(p)

    return result_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="results/plots04")
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
        # 绘制多种Pareto对比图
        result_paths = plot_pareto_comparisons(agg, inst, outdir)
        if result_paths:
            saved.extend(result_paths)

    print("-" * 80)
    print("Loaded:", Path(args.csv).resolve())
    print("Saved figures to:", outdir.resolve())
    print("Total figures:", len(saved))
    if saved:
        print("Example:", saved[0].name)


if __name__ == "__main__":
    main()
