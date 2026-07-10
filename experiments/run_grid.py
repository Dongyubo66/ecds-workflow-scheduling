from __future__ import annotations

import argparse
import copy
import itertools
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, help="plan yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _guess_scale_value(instance: str) -> float:
    m = re.search(r"(\d+)d", instance.lower())
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)k", instance.lower())
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)p", instance.lower())
    if m:
        return float(m.group(1))
    return 1e18


def _discover_instances(
    base_dir: Path,
    exclude_families: list[str],
    max_families: int,
    per_family: int,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str, float]] = []
    for p in sorted(base_dir.rglob("*.json")):
        rel = p.relative_to(base_dir)
        family = str(rel.parent).replace("\\", "/")
        if family in set(exclude_families):
            continue
        instance = p.stem
        rows.append((family, instance, _guess_scale_value(instance)))

    by_family: dict[str, list[tuple[str, str, float]]] = {}
    for family, instance, scale in rows:
        by_family.setdefault(family, []).append((family, instance, scale))

    selected: list[tuple[str, str]] = []
    for family in sorted(by_family.keys())[: max(0, max_families)]:
        cand = sorted(by_family[family], key=lambda x: (x[2], x[1]))[: max(1, per_family)]
        selected.extend([(x[0], x[1]) for x in cand])
    return selected


def _resolve_instances(plan: dict, base_cfg: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in plan.get("dataset", {}).get("instances", []):
        out.append((str(item["workflow"]), str(item["instance"])))

    ad = plan.get("dataset", {}).get("auto_discover", {})
    if ad.get("enabled", False):
        base_dir = Path(plan.get("dataset", {}).get("base_dir", base_cfg["dataset"]["base_dir"]))
        more = _discover_instances(
            base_dir=base_dir,
            exclude_families=list(ad.get("exclude_families", [])),
            max_families=int(ad.get("max_families", 0)),
            per_family=int(ad.get("per_family", 1)),
        )
        for item in more:
            if item not in out:
                out.append(item)
    return out


def _simplex_weights(step: float) -> list[tuple[float, float, float, float]]:
    n = int(round(1.0 / step))
    vals = [i * step for i in range(n + 1)]
    ans = []
    for a in vals:
        for b in vals:
            for c in vals:
                d = 1.0 - a - b - c
                if d < -1e-12:
                    continue
                d = round(d, 10)
                if d < -1e-12:
                    continue
                ans.append((round(a, 10), round(b, 10), round(c, 10), round(max(0.0, d), 10)))

    uniq = []
    seen = set()
    for w in ans:
        if abs(sum(w) - 1.0) > 1e-9:
            continue
        key = tuple(round(x, 10) for x in w)
        if key not in seen:
            seen.add(key)
            uniq.append(key)
    return uniq


def _resolve_weights(plan: dict) -> list[tuple[float, float, float, float]]:
    mode = str(plan.get("matrix", {}).get("weight_mode", "explicit")).lower()
    if mode == "explicit":
        return [tuple(float(x) for x in row) for row in plan.get("matrix", {}).get("weights", [])]
    if mode == "simplex_0.25":
        return _simplex_weights(0.25)
    if mode == "simplex_0.5":
        return _simplex_weights(0.5)
    raise ValueError(f"Unknown weight_mode: {mode}")


def _get_main_supported_flags() -> set[str]:
    """
    探测 src/main.py 当前支持哪些命令行参数。
    失败时回退到保守集合。
    """
    main_py = PROJECT_ROOT / "src" / "main.py"
    fallback = {"--cfg", "--out", "--seed", "--quiet"}

    try:
        p = subprocess.run(
            [sys.executable, str(main_py), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = (p.stdout or "") + "\n" + (p.stderr or "")
        flags = set(re.findall(r"(--[a-zA-Z0-9_-]+)", text))
        if flags:
            return flags
    except Exception:
        pass

    return fallback


SUPPORTED_MAIN_FLAGS = _get_main_supported_flags()


def run_once(
    cfg: dict,
    out_csv: Path,
    quiet: bool,
    timeout_s: float | None,
    tag: str,
    scenario: str,
    trace: bool,
    trace_path: str | None,
    seed: int,
) -> tuple[dict[str, Any] | None, str]:
    tmp_dir = PROJECT_ROOT / "configs" / "_tmp_runs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stamp = int(time.time() * 1000)
    tmp_cfg = tmp_dir / f"cfg_{stamp}.yaml"
    tmp_cfg.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(PROJECT_ROOT / "src" / "main.py")]

    if "--cfg" in SUPPORTED_MAIN_FLAGS:
        cmd.extend(["--cfg", str(tmp_cfg)])
    else:
        return None, "main.py does not support --cfg"

    if "--out" in SUPPORTED_MAIN_FLAGS:
        cmd.extend(["--out", str(out_csv)])
    else:
        return None, "main.py does not support --out"

    if "--seed" in SUPPORTED_MAIN_FLAGS:
        cmd.extend(["--seed", str(seed)])

    if quiet and "--quiet" in SUPPORTED_MAIN_FLAGS:
        cmd.append("--quiet")

    # 只有支持时才传，避免 unrecognized arguments
    if "--tag" in SUPPORTED_MAIN_FLAGS:
        cmd.extend(["--tag", str(tag)])
    if "--scenario" in SUPPORTED_MAIN_FLAGS:
        cmd.extend(["--scenario", str(scenario)])

    if trace:
        if "--trace" in SUPPORTED_MAIN_FLAGS:
            cmd.append("--trace")
        if "--trace-path" in SUPPORTED_MAIN_FLAGS:
            cmd.extend(["--trace-path", str(trace_path or "")])

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None, "timeout"

    if p.returncode != 0:
        err = (p.stderr or p.stdout or "subprocess failed")[:4000]
        return None, err

    if not out_csv.exists():
        return None, "missing output csv"

    df = pd.read_csv(out_csv)
    if df.empty:
        return None, "empty output csv"

    return dict(df.iloc[0].to_dict()), "ok"


def _postprocess_raw_csv(
    out_csv: Path,
    *,
    variant: str,
    scenario: str,
    tag: str,
    wf: str,
    inst: str,
    seed: int,
    T: int,
    w1: float,
    w2: float,
    w3: float,
    w4: float,
) -> dict[str, Any]:
    """
    强制把本轮任务的关键元数据回写到 raw csv，
    防止 main.py 内部默认值覆盖掉 variant/scenario 等标签。
    """
    raw_df = pd.read_csv(out_csv)
    if raw_df.empty:
        raise ValueError(f"empty output csv after run: {out_csv}")

    raw_df["variant"] = variant
    raw_df["abl"] = variant
    raw_df["scenario"] = scenario
    raw_df["run_tag"] = tag

    raw_df["workflow"] = wf
    raw_df["family"] = wf
    raw_df["instance"] = inst
    raw_df["seed"] = int(seed)

    raw_df["task_release_random"] = bool(scenario == "dynamic")
    raw_df["task_release_T"] = int(T if scenario == "dynamic" else 0)

    raw_df["w1"] = float(w1)
    raw_df["w2"] = float(w2)
    raw_df["w3"] = float(w3)
    raw_df["w4"] = float(w4)

    raw_df.to_csv(out_csv, index=False, encoding="utf-8")
    return dict(raw_df.iloc[0].to_dict())


def main():
    args = parse_args()
    plan = _load_yaml(args.plan)
    base_cfg = _load_yaml(plan.get("base_config", "configs/static.yaml"))

    exp = plan.get("experiment", {})
    scenario = str(exp.get("scenario", "static"))
    tag = str(exp.get("name", "grid"))
    quiet = bool(exp.get("quiet", True))
    timeout_s = exp.get("timeout_s", 1200.0)
    keep_single_run_files = bool(exp.get("keep_single_run_files", True))
    trace = bool(exp.get("trace", False))

    seeds = [int(x) for x in plan.get("matrix", {}).get("seeds", [base_cfg.get("seed", 0)])]
    schedulers = [str(x) for x in plan.get("matrix", {}).get("schedulers", ["ECDS"])]
    weights = _resolve_weights(plan)
    instances = _resolve_instances(plan, base_cfg)
    dynamic_T_list = [
        int(x) for x in plan.get("matrix", {}).get("dynamic_T", [0 if scenario == "static" else 100])
    ]
    ablations = plan.get("matrix", {}).get("ablations", [{"tag": "full"}])

    out_root = Path(exp.get("output_dir", "results/grid"))
    if not out_root.is_absolute():
        out_root = (PROJECT_ROOT / out_root).resolve()

    raw_dir = out_root / "raw"
    trace_dir = out_root / "traces"
    manifest_dir = out_root / "manifests"
    summary_dir = out_root / "summary"

    for d in (raw_dir, trace_dir, manifest_dir, summary_dir):
        d.mkdir(parents=True, exist_ok=True)

    tasks = []
    for (wf, inst), sched, (w1, w2, w3, w4), seed, T in itertools.product(
        instances, schedulers, weights, seeds, dynamic_T_list
    ):
        if sched.upper() != "ECDS":
            abl_list = [{"tag": "full"}]
        else:
            abl_list = ablations

        for abl in abl_list:
            tasks.append(
                {
                    "workflow": wf,
                    "instance": inst,
                    "scheduler": sched,
                    "w1": w1,
                    "w2": w2,
                    "w3": w3,
                    "w4": w4,
                    "seed": seed,
                    "T": T,
                    "abl": abl,
                }
            )

    manifest_rows = []
    failed_rows = []

    print(f"Plan: {tag}")
    print(f"Scenario: {scenario}")
    print(
        f"Instances: {len(instances)} | Schedulers: {len(schedulers)} | "
        f"Weights: {len(weights)} | Seeds: {len(seeds)} | T-list: {len(dynamic_T_list)}"
    )
    print(f"Total tasks: {len(tasks)}")
    print(f"Detected main.py flags: {sorted(SUPPORTED_MAIN_FLAGS)}")

    if args.dry_run:
        return

    done = 0
    t0_all = time.time()

    for task in tasks:
        done += 1
        wf = task["workflow"]
        inst = task["instance"]
        sched = task["scheduler"]
        seed = int(task["seed"])
        T = int(task["T"])
        abl = task["abl"]
        variant = str(abl.get("tag", "full"))
        percent = 100.0 * done / max(1, len(tasks))

        cfg = copy.deepcopy(base_cfg)

        cfg.setdefault("experiment", {})
        cfg["experiment"]["scenario"] = scenario
        cfg["experiment"]["run_tag"] = tag
        cfg["experiment"]["variant"] = variant

        cfg.setdefault("dataset", {})
        cfg["dataset"]["workflows"] = [wf]
        cfg["dataset"]["instances"] = [inst]
        cfg["dataset"]["arrivals"] = [0]

        # 同时兼容 seed / seeds 两种写法
        cfg["seed"] = seed
        cfg["seeds"] = [seed]

        cfg.setdefault("dynamic", {})
        cfg["dynamic"]["task_release_random"] = bool(scenario == "dynamic")
        cfg["dynamic"]["task_release_T"] = int(T if scenario == "dynamic" else 0)

        cfg.setdefault("scheduler", {})
        cfg["scheduler"]["name"] = sched
        cfg["scheduler"].setdefault("params", {})

        cfg.setdefault("objective", {})
        cfg["objective"]["w1"] = float(task["w1"])
        cfg["objective"]["w2"] = float(task["w2"])
        cfg["objective"]["w3"] = float(task["w3"])
        cfg["objective"]["w4"] = float(task["w4"])

        # baseline-specific params
        if sched.upper() in ("MOHEFT", "MOHEFT_BUDGET", "BUDGETED_MOHEFT"):
            mp = plan.get("scheduler_params", {}).get("MOHEFT", {})
            cfg["scheduler"]["params"]["decision_budget_ms"] = float(mp.get("decision_budget_ms", 30.0))
            cfg["scheduler"]["params"]["candidate_cap"] = int(mp.get("candidate_cap", 12))
            cfg["scheduler"]["params"]["frontier_cap"] = int(mp.get("frontier_cap", 24))

        # ablation overrides for ECDS
        if sched.upper() == "ECDS":
            cfg["scheduler"].setdefault("clustering", {})
            cfg["scheduler"].setdefault("reschedule", {})

            if abl.get("clustering_method") is not None:
                cfg["scheduler"]["clustering"]["method"] = abl["clustering_method"]

            if abl.get("reschedule_enabled") is not None:
                cfg["scheduler"]["reschedule"]["enabled"] = bool(abl["reschedule_enabled"])

            if abl.get("force_w4") is not None:
                cfg["objective"]["w4"] = float(abl["force_w4"])

        stem = (
            f"run__{scenario}__{wf.replace('/', '_')}__{inst}__{sched}__{variant}"
            f"__w{cfg['objective']['w1']}-{cfg['objective']['w2']}-{cfg['objective']['w3']}-{cfg['objective']['w4']}"
            f"__T{cfg['dynamic']['task_release_T']}__seed{seed}"
        )
        out_csv = raw_dir / f"{stem}.csv"
        trace_path = trace_dir / f"{stem}.trace.csv"

        if out_csv.exists() and not args.force:
            try:
                old = pd.read_csv(out_csv)
                if not old.empty and int(old.iloc[0].get("success", 1)) == 1:
                    manifest_rows.append(
                        {
                            "workflow": wf,
                            "instance": inst,
                            "scheduler": sched,
                            "variant": variant,
                            "seed": seed,
                            "T": T,
                            "raw_csv": str(out_csv),
                            "status": "skip_exists",
                        }
                    )
                    continue
            except Exception:
                pass

        print(
            f"[{done:>4}/{len(tasks)}] {percent:6.2f}% START "
            f"{wf} | {inst} | {sched} | {variant} | T={T} | seed={seed}"
        )
        t0 = time.time()

        row, status_msg = run_once(
            cfg=cfg,
            out_csv=out_csv,
            quiet=quiet,
            timeout_s=timeout_s,
            tag=tag,
            scenario=scenario,
            trace=trace,
            trace_path=str(trace_path),
            seed=seed,
        )
        dt = time.time() - t0

        if row is not None:
            try:
                row = _postprocess_raw_csv(
                    out_csv,
                    variant=variant,
                    scenario=scenario,
                    tag=tag,
                    wf=wf,
                    inst=inst,
                    seed=seed,
                    T=T,
                    w1=float(cfg["objective"]["w1"]),
                    w2=float(cfg["objective"]["w2"]),
                    w3=float(cfg["objective"]["w3"]),
                    w4=float(cfg["objective"]["w4"]),
                )
            except Exception as e:
                row = None
                status_msg = f"postprocess raw csv failed: {e}"

        if row is None:
            failed_rows.append(
                {
                    "workflow": wf,
                    "instance": inst,
                    "scheduler": sched,
                    "variant": variant,
                    "seed": seed,
                    "T": T,
                    "raw_csv": str(out_csv),
                    "status": status_msg,
                }
            )
            manifest_rows.append(
                {
                    "workflow": wf,
                    "instance": inst,
                    "scheduler": sched,
                    "variant": variant,
                    "seed": seed,
                    "T": T,
                    "raw_csv": str(out_csv),
                    "status": "failed",
                }
            )
            print(f"[{done:>4}/{len(tasks)}] {percent:6.2f}% FAIL  (t={dt:.2f}s) -> {status_msg}")
            continue

        manifest_rows.append(
            {
                "workflow": wf,
                "instance": inst,
                "scheduler": sched,
                "variant": variant,
                "seed": seed,
                "T": T,
                "raw_csv": str(out_csv),
                "status": "ok",
            }
        )

        print(
            f"[{done:>4}/{len(tasks)}] {percent:6.2f}% DONE  (t={dt:.2f}s) "
            f"makespan={row.get('makespan')} energy={row.get('total_energy')} brown={row.get('brown_energy')}"
        )

        if not keep_single_run_files:
            try:
                out_csv.unlink(missing_ok=True)
            except OSError:
                pass

    manifest_path = manifest_dir / f"manifest__{tag}.csv"
    failed_path = manifest_dir / f"failed_runs__{tag}.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8")
    pd.DataFrame(failed_rows).to_csv(failed_path, index=False, encoding="utf-8")

    elapsed = time.time() - t0_all
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved failures: {failed_path}")
    print(f"Elapsed: {elapsed/60:.2f} min")

    if bool(exp.get("aggregate_after_run", False)):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "aggregate_results.py"),
            "--input-dir",
            str(raw_dir),
            "--out-dir",
            str(summary_dir),
        ]
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()