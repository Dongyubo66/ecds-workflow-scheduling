from pathlib import Path
import argparse
import pandas as pd


def safe_read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def normalize_success_series(s: pd.Series) -> pd.Series:
    """
    将 success 列统一转成 bool。
    兼容 1/0, True/False, "1"/"0", "true"/"false"。
    """
    if s.dtype == bool:
        return s

    s2 = s.astype(str).str.strip().str.lower()
    return s2.isin(["1", "true", "t", "yes", "y"])


def normalize_scheduler_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def filter_runs(
    df: pd.DataFrame,
    scenario: str = None,
    only_success: bool = True,
):
    out = df.copy()

    if scenario is not None and "scenario" in out.columns:
        out = out[out["scenario"] == scenario].copy()

    if only_success and "success" in out.columns:
        out = out[normalize_success_series(out["success"])].copy()

    return out


def aggregate_summary(df: pd.DataFrame, group_cols):
    """
    对结果表进行统一聚合。
    """
    agg_spec = {}

    numeric_targets = [
        "makespan",
        "total_energy",
        "total_carbon",
        "green_ratio",
        "brown_energy",
        "avg_utilization",
    ]

    for col in numeric_targets:
        if col in df.columns:
            agg_spec[f"{col}_mean"] = (col, "mean")
            agg_spec[f"{col}_std"] = (col, "std")

    # n_runs 优先按 seed 计数；没有 seed 就按行数计数
    if "seed" in df.columns:
        agg_spec["n_runs"] = ("seed", "count")
    else:
        agg_spec["n_runs"] = (group_cols[0], "count")

    agg = df.groupby(group_cols).agg(**agg_spec).reset_index()

    sort_cols = [c for c in ["family", "instance_scale_tag", "instance", "scheduler"] if c in agg.columns]
    if sort_cols:
        agg = agg.sort_values(by=sort_cols)

    return agg


def build_static_runs(static_df: pd.DataFrame, out_dir: Path):
    """
    导出静态原始结果的精简表。
    这是 raw-selected 表，不是聚合 summary。
    """
    static_df = filter_runs(static_df, scenario="static", only_success=True)

    keep_cols = [
        "family",
        "workflow",
        "instance",
        "instance_scale_tag",
        "scheduler",
        "makespan",
        "total_energy",
        "total_carbon",
        "green_ratio",
        "brown_energy",
        "avg_utilization",
        "seed",
        "success",
        "abl",
    ]
    keep_cols = [c for c in keep_cols if c in static_df.columns]

    out = static_df[keep_cols].copy()

    sort_cols = [c for c in ["family", "instance_scale_tag", "instance", "scheduler", "seed"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols)

    out_path = out_dir / "family_static_runs.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved static runs: {out_path}")
    print(f"Static raw-selected rows: {len(out)}")

    return out


def build_static_main_summary(static_df: pd.DataFrame, out_dir: Path, main_schedulers):
    """
    真正的静态主结果聚合表。
    用于论文主文，默认只保留 ECDS / HEFT。
    """
    static_df = filter_runs(static_df, scenario="static", only_success=True)

    if "scheduler" in static_df.columns:
        static_df = static_df[static_df["scheduler"].isin(main_schedulers)].copy()

    group_cols = [c for c in ["family", "workflow", "instance", "instance_scale_tag", "scheduler"] if c in static_df.columns]
    agg = aggregate_summary(static_df, group_cols)

    out_path = out_dir / "static_main_summary.csv"
    agg.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved static main summary: {out_path}")
    print(f"Static grouped rows: {len(agg)}")

    return agg


def build_dynamic_summary(dynamic_df: pd.DataFrame, out_dir: Path):
    """
    动态全 scheduler 聚合表。
    兼容你原来 family_dynamic_summary.csv 的用途。
    """
    dynamic_df = filter_runs(dynamic_df, scenario="dynamic", only_success=True)

    group_cols = [c for c in ["family", "workflow", "instance", "instance_scale_tag", "scheduler"] if c in dynamic_df.columns]
    agg = aggregate_summary(dynamic_df, group_cols)

    out_path = out_dir / "family_dynamic_summary.csv"
    agg.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved dynamic family summary: {out_path}")
    print(f"Dynamic grouped rows: {len(agg)}")

    return agg


def build_dynamic_main_summary(dynamic_df: pd.DataFrame, out_dir: Path, main_schedulers):
    """
    动态主结果聚合表。
    用于论文主文，默认只保留 ECDS / HEFT。
    """
    dynamic_df = filter_runs(dynamic_df, scenario="dynamic", only_success=True)

    if "scheduler" in dynamic_df.columns:
        dynamic_df = dynamic_df[dynamic_df["scheduler"].isin(main_schedulers)].copy()

    group_cols = [c for c in ["family", "workflow", "instance", "instance_scale_tag", "scheduler"] if c in dynamic_df.columns]
    agg = aggregate_summary(dynamic_df, group_cols)

    out_path = out_dir / "dynamic_main_summary.csv"
    agg.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved dynamic main summary: {out_path}")
    print(f"Dynamic main grouped rows: {len(agg)}")

    return agg


def build_delta_summary(dynamic_main_agg: pd.DataFrame, out_dir: Path):
    """
    只对 ECDS 和 HEFT 计算动态差值。
    使用 dynamic_main_summary.csv 对应的聚合结果。
    """
    if "scheduler" not in dynamic_main_agg.columns:
        print("Skip delta summary: no scheduler column.")
        return

    pivot = dynamic_main_agg.pivot_table(
        index=["family", "workflow", "instance", "instance_scale_tag"],
        columns="scheduler",
        values=[
            "brown_energy_mean",
            "green_ratio_mean",
            "makespan_mean",
            "total_carbon_mean",
            "total_energy_mean",
        ]
    )

    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    needed = [
        "brown_energy_mean_ECDS", "brown_energy_mean_HEFT",
        "green_ratio_mean_ECDS", "green_ratio_mean_HEFT",
        "makespan_mean_ECDS", "makespan_mean_HEFT",
        "total_carbon_mean_ECDS", "total_carbon_mean_HEFT",
        "total_energy_mean_ECDS", "total_energy_mean_HEFT",
    ]
    missing = [c for c in needed if c not in pivot.columns]
    if missing:
        print("Skip delta summary: missing columns:")
        for c in missing:
            print("  -", c)
        return

    pivot["delta_makespan"] = pivot["makespan_mean_ECDS"] - pivot["makespan_mean_HEFT"]
    pivot["delta_energy"] = pivot["total_energy_mean_ECDS"] - pivot["total_energy_mean_HEFT"]
    pivot["delta_carbon"] = pivot["total_carbon_mean_ECDS"] - pivot["total_carbon_mean_HEFT"]
    pivot["delta_green_ratio"] = pivot["green_ratio_mean_ECDS"] - pivot["green_ratio_mean_HEFT"]
    pivot["delta_brown"] = pivot["brown_energy_mean_ECDS"] - pivot["brown_energy_mean_HEFT"]

    sort_cols = [c for c in ["family", "instance_scale_tag", "instance"] if c in pivot.columns]
    if sort_cols:
        pivot = pivot.sort_values(by=sort_cols)

    out_path = out_dir / "family_delta_summary.csv"
    pivot.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved delta summary: {out_path}")
    print(f"Delta rows: {len(pivot)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static",
        default="results/runs_static_families/grid_results_static_families.csv",
        help="Path to static family grid csv"
    )
    parser.add_argument(
        "--dynamic",
        default="results/runs_dynamic_families/grid_results_dynamic_families.csv",
        help="Path to dynamic family grid csv"
    )
    parser.add_argument(
        "--outdir",
        default="results/summaries",
        help="Output directory"
    )
    parser.add_argument(
        "--main-schedulers",
        default="ECDS,HEFT",
        help='Comma-separated schedulers used in paper main tables, e.g. "ECDS,HEFT"'
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    static_path = (project_root / args.static).resolve()
    dynamic_path = (project_root / args.dynamic).resolve()
    out_dir = (project_root / args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    main_schedulers = normalize_scheduler_list(args.main_schedulers)

    static_df = safe_read_csv(static_path)
    dynamic_df = safe_read_csv(dynamic_path)

    print("Static path :", static_path)
    print("Dynamic path:", dynamic_path)
    print("Output dir  :", out_dir)
    print("Main schedulers:", main_schedulers)

    if static_df is not None:
        build_static_runs(static_df, out_dir)
        build_static_main_summary(static_df, out_dir, main_schedulers)
    else:
        print("Static csv not found, skip static outputs.")

    if dynamic_df is not None:
        build_dynamic_summary(dynamic_df, out_dir)
        dyn_main_agg = build_dynamic_main_summary(dynamic_df, out_dir, main_schedulers)
        build_delta_summary(dyn_main_agg, out_dir)
    else:
        print("Dynamic csv not found, skip dynamic outputs.")


if __name__ == "__main__":
    main()