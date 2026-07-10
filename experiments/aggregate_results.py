from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "configs").exists() and (cur / "src").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("Cannot find project root")


PROJECT_ROOT = find_project_root(Path(__file__).parent)
OBJ_COLS = ["makespan", "total_energy", "unavailability", "brown_energy"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Directory containing raw run CSVs")
    p.add_argument("--out-dir", required=True, help="Directory to save summary CSVs")
    return p.parse_args()


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b + 1e-12) and np.any(a < b - 1e-12))


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if _dominates(points[j], points[i]):
                keep[i] = False
                break
    return keep


def unique_rows(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    return np.unique(np.round(points, 12), axis=0)


def hv_recursive_min(points: np.ndarray, ref: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    d = points.shape[1]
    if d == 1:
        return float(max(0.0, ref[0] - np.min(points[:, 0])))

    order = np.argsort(points[:, d - 1])
    pts = points[order]
    hvol = 0.0
    prev = float(ref[d - 1])
    for i in range(len(pts) - 1, -1, -1):
        z = float(pts[i, d - 1])
        if z >= prev - 1e-12:
            continue
        slice_pts = pts[: i + 1, : d - 1]
        slice_ref = ref[: d - 1]
        hvol += hv_recursive_min(slice_pts, slice_ref) * (prev - z)
        prev = z
    return float(hvol)


def hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    pts = points[np.all(points <= ref + 1e-12, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = unique_rows(pts)
    pts = pts[nondominated_mask(pts)]
    return hv_recursive_min(pts, ref)


def igd(front: np.ndarray, ref_front: np.ndarray) -> float:
    if len(front) == 0 or len(ref_front) == 0:
        return float("inf")
    dsum = 0.0
    for r in ref_front:
        d = np.linalg.norm(front - r.reshape(1, -1), axis=1)
        dsum += float(np.min(d))
    return dsum / max(1, len(ref_front))


def eps_add(front: np.ndarray, ref_front: np.ndarray) -> float:
    if len(front) == 0 or len(ref_front) == 0:
        return float("inf")
    best = -float("inf")
    for r in ref_front:
        inner = float("inf")
        for a in front:
            inner = min(inner, float(np.max(a - r)))
        best = max(best, inner)
    return float(best)


def _display_scheduler(row: pd.Series) -> str:
    sched = str(row.get("scheduler", ""))
    variant = str(row.get("variant", "full"))
    if sched.upper() == "ECDS" and variant.lower() != "full":
        return f"ECDS-{variant.upper()}"
    return sched


def _format_mean_std(mean_val: float, std_val: float) -> str:
    return f"{mean_val:.3f} ± {std_val:.3f}"


def _group_indicators(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_group_cols = ["scenario", "workflow", "instance", "seed"]
    for group_key, g in df.groupby(base_group_cols, dropna=False):
        g = g.copy()
        g["unavailability"] = 1.0 - g["avg_utilization"].astype(float)
        g["scheduler_display"] = g.apply(_display_scheduler, axis=1)

        all_points = g[["makespan", "total_energy", "unavailability", "brown_energy"]].astype(float).to_numpy()
        all_points = unique_rows(all_points)
        all_points = all_points[nondominated_mask(all_points)]
        if len(all_points) == 0:
            continue

        mins = np.min(all_points, axis=0)
        maxs = np.max(all_points, axis=0)
        span = np.maximum(maxs - mins, 1e-12)
        ref_front = (all_points - mins) / span
        hv_ref = np.ones(ref_front.shape[1], dtype=float) * 1.05

        for sched, sg in g.groupby("scheduler_display"):
            pts = sg[["makespan", "total_energy", "unavailability", "brown_energy"]].astype(float).to_numpy()
            pts = unique_rows(pts)
            pts = pts[nondominated_mask(pts)]
            pts_n = (pts - mins) / span
            rows.append(
                {
                    "scenario": group_key[0],
                    "workflow": group_key[1],
                    "instance": group_key[2],
                    "seed": group_key[3],
                    "scheduler": sched,
                    "n_points": int(len(pts_n)),
                    "HV": float(hypervolume(pts_n, hv_ref)),
                    "IGD": float(igd(pts_n, ref_front)),
                    "EPS_ADD": float(eps_add(pts_n, ref_front)),
                    "wallclock_s": float(sg["scheduler_wallclock_s"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _agg_static(ind: pd.DataFrame) -> pd.DataFrame:
    if ind.empty:
        return ind
    static_df = ind[ind["scenario"] == "static"].copy()
    static_df = static_df.sort_values(["workflow", "instance", "scheduler"])
    return static_df


def _agg_dynamic(ind: pd.DataFrame) -> pd.DataFrame:
    if ind.empty:
        return ind
    dyn = ind[ind["scenario"] == "dynamic"].copy()
    if dyn.empty:
        return dyn
    rows = []
    for key, g in dyn.groupby(["workflow", "instance", "scheduler"], dropna=False):
        rows.append(
            {
                "workflow": key[0],
                "instance": key[1],
                "scheduler": key[2],
                "n_points_mean": float(g["n_points"].mean()),
                "n_points_std": float(g["n_points"].std(ddof=0)),
                "HV_mean": float(g["HV"].mean()),
                "HV_std": float(g["HV"].std(ddof=0)),
                "IGD_mean": float(g["IGD"].mean()),
                "IGD_std": float(g["IGD"].std(ddof=0)),
                "EPS_ADD_mean": float(g["EPS_ADD"].mean()),
                "EPS_ADD_std": float(g["EPS_ADD"].std(ddof=0)),
                "wallclock_mean": float(g["wallclock_s"].mean()),
                "wallclock_std": float(g["wallclock_s"].std(ddof=0)),
                "n_points": _format_mean_std(float(g["n_points"].mean()), float(g["n_points"].std(ddof=0))),
                "HV": _format_mean_std(float(g["HV"].mean()), float(g["HV"].std(ddof=0))),
                "IGD": _format_mean_std(float(g["IGD"].mean()), float(g["IGD"].std(ddof=0))),
                "EPS_ADD": _format_mean_std(float(g["EPS_ADD"].mean()), float(g["EPS_ADD"].std(ddof=0))),
            }
        )
    return pd.DataFrame(rows).sort_values(["workflow", "instance", "scheduler"])


def _runtime_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    grp_cols = ["scenario", "workflow", "instance", "scheduler"]
    for key, g in df.groupby(grp_cols, dropna=False):
        rows.append(
            {
                "scenario": key[0],
                "workflow": key[1],
                "instance": key[2],
                "scheduler": _display_scheduler(g.iloc[0]),
                "wallclock_mean": float(g["scheduler_wallclock_s"].mean()),
                "wallclock_std": float(g["scheduler_wallclock_s"].std(ddof=0)),
                "unfinished_sum": int(g["unfinished_tasks"].sum()),
                "scheduler_calls_mean": float(g["scheduler_calls"].mean()),
                "dispatch_count_mean": float(g["dispatch_count"].mean()),
                "event_count_mean": float(g["event_count"].mean()),
                "recluster_count_mean": float(g["recluster_count"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(grp_cols)


def _ablation_summary(ind: pd.DataFrame) -> pd.DataFrame:
    if ind.empty:
        return ind
    abl = ind[ind["scheduler"].str.startswith("ECDS", na=False)].copy()
    return abl.sort_values(["scenario", "workflow", "instance", "seed", "scheduler"])


def _screen_gate(ind: pd.DataFrame) -> pd.DataFrame:
    if ind.empty:
        return ind
    rows = []
    for key, g in ind.groupby(["scenario", "workflow", "instance", "seed"], dropna=False):
        hv_rank = g.sort_values(["HV", "IGD", "EPS_ADD"], ascending=[False, True, True]).reset_index(drop=True)
        igd_rank = g.sort_values(["IGD", "HV"], ascending=[True, False]).reset_index(drop=True)
        rows.append(
            {
                "scenario": key[0],
                "workflow": key[1],
                "instance": key[2],
                "seed": key[3],
                "best_by_HV": hv_rank.iloc[0]["scheduler"],
                "best_by_IGD": igd_rank.iloc[0]["scheduler"],
                "ecdS_present": int((g["scheduler"] == "ECDS").any()),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = (PROJECT_ROOT / input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {input_dir}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(input_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No csv found under {input_dir}")
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)

    ind = _group_indicators(df)
    static_df = _agg_static(ind)
    dynamic_df = _agg_dynamic(ind)
    runtime_df = _runtime_summary(df)
    ablation_df = _ablation_summary(ind)
    gate_df = _screen_gate(ind)

    static_df.to_csv(out_dir / "static_summary.csv", index=False, encoding="utf-8")
    dynamic_df.to_csv(out_dir / "dynamic_summary.csv", index=False, encoding="utf-8")
    runtime_df.to_csv(out_dir / "runtime_summary.csv", index=False, encoding="utf-8")
    ablation_df.to_csv(out_dir / "ablation_summary.csv", index=False, encoding="utf-8")
    gate_df.to_csv(out_dir / "screen_gate_report.csv", index=False, encoding="utf-8")
    ind.to_csv(out_dir / "indicator_rows.csv", index=False, encoding="utf-8")

    print(f"Saved summaries to: {out_dir}")


if __name__ == "__main__":
    main()
