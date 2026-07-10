import itertools
from pathlib import Path
import yaml
import pandas as pd
import subprocess
import copy
import sys
import time
from typing import Any


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "configs" / "static.yaml").exists() and (cur / "src" / "main.py").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("Cannot find project root")


PROJECT_ROOT = find_project_root(Path(__file__).parent)


def run_once(cfg: dict, out_csv: Path, quiet: bool = True, timeout_s: float | None = 1200.0) -> dict[str, Any] | None:
    tmp_dir = PROJECT_ROOT / "configs" / "_tmp_runs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stamp = int(time.time() * 1000)
    tmp_cfg = tmp_dir / f"cfg_{stamp}.yaml"
    tmp_cfg.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "main.py"),
        "--cfg",
        str(tmp_cfg),
        "--out",
        str(out_csv),
    ]
    if quiet:
        cmd.append("--quiet")

    try:
        if timeout_s is None:
            p = subprocess.run(cmd, capture_output=True, text=True)
        else:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=float(timeout_s))
    except subprocess.TimeoutExpired:
        return None

    if p.returncode != 0:
        return None

    if not out_csv.exists():
        return None

    df = pd.read_csv(out_csv)
    if df.empty:
        return None
    return dict(df.iloc[0].to_dict())


def main():
    base_cfg_path = PROJECT_ROOT / "configs" / "dynamic.yaml"
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))

    keep_single_run_files = False
    quiet = True
    timeout_s = 1200.0

    schedulers = ["HEFT", "ECDS"]

    w123_grid = [
        (0.5, 0.3, 0.2),
    ]
    w4_grid = [0.2]

    instances = [
        ("montage/chameleon-cloud", "montage-chameleon-2mass-01d-001"),
        ("montage/chameleon-cloud", "montage-chameleon-2mass-05d-001"),

        ("epigenomics/chameleon-cloud", "epigenomics-chameleon-hep-1seq-50k-001"),
        ("epigenomics/chameleon-cloud", "epigenomics-chameleon-hep-1seq-100k-001"),

        ("seismology/chameleon-cloud", "seismology-chameleon-100p-001"),
        ("seismology/chameleon-cloud", "seismology-chameleon-200p-001"),
    ]

    seeds = [1, 2, 3]

    ablations = [
        {"tag": "full", "clustering_method": None, "reschedule_enabled": None, "force_w4": None},
    ]

    out_dir = PROJECT_ROOT / "results" / "runs_dynamic_families"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / "grid_results_dynamic_families.csv"

    results: list[dict[str, Any]] = []
    failed = 0

    for (wf, inst), sched, (w1, w2, w3), w4, seed in itertools.product(
        instances, schedulers, w123_grid, w4_grid, seeds
    ):
        abl_list = ablations if sched == "ECDS" else [ablations[0]]

        for abl in abl_list:
            cfg = copy.deepcopy(base_cfg)
            cfg["dataset"]["workflows"] = [wf]
            cfg["dataset"]["instances"] = [inst]
            cfg["dataset"]["arrivals"] = [0]
            cfg["seed"] = int(seed)

            cfg["scheduler"]["name"] = sched
            cfg["objective"]["w1"] = float(w1)
            cfg["objective"]["w2"] = float(w2)
            cfg["objective"]["w3"] = float(w3)
            cfg["objective"]["w4"] = float(w4)

            if sched == "ECDS":
                force_w4 = abl.get("force_w4", None)
                if force_w4 is not None:
                    cfg["objective"]["w4"] = float(force_w4)

            out_csv = out_dir / f"{wf.replace('/','_')}__{inst}__{sched}__seed{seed}.csv"
            row = run_once(cfg, out_csv, quiet=quiet, timeout_s=timeout_s)

            if row is None:
                failed += 1
                continue

            row["workflow"] = wf
            row["instance"] = inst
            row["seed"] = int(seed)
            row["abl"] = abl["tag"]
            row["w1"], row["w2"], row["w3"], row["w4"] = float(w1), float(w2), float(w3), float(cfg["objective"]["w4"])
            results.append(row)

            if not keep_single_run_files:
                try:
                    out_csv.unlink(missing_ok=True)
                except OSError:
                    pass

    pd.DataFrame(results).to_csv(grid_path, index=False, encoding="utf-8")
    print("Saved:", grid_path)
    print("OK rows:", len(results), "FAILED:", failed)


if __name__ == "__main__":
    main()