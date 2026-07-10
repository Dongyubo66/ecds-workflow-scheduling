from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from wf.loader import load_wfcommons_instance
from wf.dynamic import assign_task_release_times

from sim.energy import DVFSPowerModel
from sim.resources import Machine, GreenSegment
from sim.simulator_fast import DiscreteEventSimulator

from sim.schedulers.baselines import (
    FCFS,
    LIST,
    HEFT_Simple,
    GREENHEFT,
    MOHEFT,
    NSGA2,
)
from sim.schedulers.ecds import ECDS, ECDSConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/static.yaml", help="配置文件路径")
    parser.add_argument("--out", default="", help="单次运行输出 CSV 路径")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置文件中的 seed")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--study", action="store_true", help="启用批量 study 模式")
    return parser.parse_args()


def _auto_out_path(cfg: dict, scheduler_name: str, seed: int) -> Path:
    wf = cfg["dataset"]["workflows"][0]
    inst = cfg["dataset"]["instances"][0]
    safe_wf = str(wf).replace("/", "_").replace("\\", "_")
    safe_inst = str(inst).replace("/", "_").replace("\\", "_")
    fname = f"{safe_wf}__{safe_inst}__{scheduler_name}__seed{seed}.csv"
    return PROJECT_ROOT / "results" / "runs" / fname


def _parse_profile(profile_list):
    segs = []
    for item in (profile_list or []):
        segs.append(GreenSegment(start=float(item["start"]), green=float(item["green"])))
    segs.sort(key=lambda x: x.start)
    return segs


def _infer_scale_tag(instance: str) -> str:
    s = str(instance).lower()
    if any(k in s for k in ["01d", "50k", "100p", "small"]):
        return "small"
    if any(k in s for k in ["05d", "100k", "200p", "medium"]):
        return "medium"
    if any(k in s for k in ["10d", "300p", "large"]):
        return "large"
    return "unknown"


def _format_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        m = int(sec // 60)
        s = sec - 60 * m
        return f"{m}m{s:04.1f}s"
    h = int(sec // 3600)
    rem = sec - 3600 * h
    m = int(rem // 60)
    s = rem - 60 * m
    return f"{h}h{m:02d}m{s:04.1f}s"


def build_scheduler(cfg: dict):
    name = str(cfg["scheduler"]["name"]).strip().upper()

    ocfg = cfg.get("objective", {})
    scfg = cfg.get("scheduler", {})
    ccfg = scfg.get("clustering", {})
    rcfg = scfg.get("reschedule", {})
    mcfg = scfg.get("moheft", {})

    if name == "FCFS":
        return FCFS()

    if name == "LIST":
        return LIST()

    if name == "HEFT":
        return HEFT_Simple()

    if name == "GREENHEFT":
        return GREENHEFT(
            wf=float(ocfg.get("w1", 0.5)),
            we=float(ocfg.get("w2", 0.3)),
            wg=float(ocfg.get("w3", 0.2)),
        )

    if name == "MOHEFT":
        return MOHEFT(
            w1=float(ocfg.get("w1", 0.5)),
            w2=float(ocfg.get("w2", 0.3)),
            w3=float(ocfg.get("w3", 0.2)),
            w4=float(ocfg.get("w4", 0.0)),
            decision_budget_ms=float(mcfg.get("decision_budget_ms", scfg.get("decision_budget_ms", 0.0))),
            candidate_cap=int(mcfg.get("candidate_cap", scfg.get("candidate_cap", 0))),
            frontier_cap=int(mcfg.get("frontier_cap", scfg.get("frontier_cap", 0))),
            seed=int(cfg.get("seed", 0)),
        )

    if name in ("NSGA-II", "NSGA2"):
        return NSGA2(
            pop_size=int(scfg.get("pop_size", 40)),
            ngen=int(scfg.get("ngen", 15)),
            cxpb=float(scfg.get("cxpb", 0.7)),
            mutpb=float(scfg.get("mutpb", 0.25)),
            seed=int(cfg.get("seed", 0)),
        )

    if name == "ECDS":
        e = ECDSConfig(
            w1=float(ocfg["w1"]),
            w2=float(ocfg["w2"]),
            w3=float(ocfg["w3"]),
            w4=float(ocfg.get("w4", 0.0)),
            clustering_method=str(ccfg.get("method", "auto")),
            k=int(ccfg.get("k", 8)),
            eps=float(ccfg.get("dbscan_eps", 0.8)),
            min_samples=int(ccfg.get("dbscan_min_samples", 3)),
            auto_large_n_threshold=int(ccfg.get("auto_large_n_threshold", 2000)),
            auto_large_n_method=str(ccfg.get("auto_large_n_method", "kmeans")),
            reschedule_enabled=bool(rcfg.get("enabled", True)),
            dyn_threshold=float(rcfg.get("dyn_threshold", 0.2)),
            reschedule_cooldown=float(rcfg.get("cooldown", 30.0)),
            util_window=float(rcfg.get("util_window", 5.0)),
            ca=float(ccfg.get("score_a", 0.5)),
            cb=float(ccfg.get("score_b", 0.3)),
            cc=float(ccfg.get("score_c", 0.2)),
        )
        return ECDS(e)

    raise ValueError(f"Unknown scheduler: {cfg['scheduler']['name']}")


def run_once(cfg: dict, quiet: bool = False) -> dict[str, Any]:
    seed = int(cfg.get("seed", 0))

    base_dir = Path(cfg["dataset"]["base_dir"])
    workflows = cfg["dataset"]["workflows"]
    instances = cfg["dataset"]["instances"]
    arrivals = cfg["dataset"]["arrivals"]

    if not (len(workflows) == len(instances) == len(arrivals)):
        raise ValueError(
            f"Length mismatch: workflows={len(workflows)}, instances={len(instances)}, arrivals={len(arrivals)}"
        )

    sites_cfg = cfg.get("resources", {}).get("sites", {})
    machines = []
    for m in cfg["resources"]["machines"]:
        pm = DVFSPowerModel(a=float(m["a"]), b=float(m["b"]))

        site_id = str(m.get("site", "default"))
        scfg = sites_cfg.get(site_id, {})

        ci_green = float(m.get("ci_green", scfg.get("ci_green", 0.05)))
        ci_brown = float(m.get("ci_brown", scfg.get("ci_brown", 0.55)))

        green_profile = _parse_profile(m.get("green_profile", scfg.get("green_profile", [])))
        green_ratio = float(m.get("green_ratio", scfg.get("green_ratio", 0.0)))

        machines.append(
            Machine(
                name=str(m["name"]),
                speed=float(m["speed"]),
                f=float(m["f"]),
                power=pm,
                site=site_id,
                green_ratio=green_ratio,
                green_profile=green_profile,
                ci_green=ci_green,
                ci_brown=ci_brown,
            )
        )

    scheduler = build_scheduler(cfg)
    sim = DiscreteEventSimulator(machines, scheduler)

    wf_list = []
    total_tasks = 0
    for wf_folder, inst, at in zip(workflows, instances, arrivals):
        json_path = base_dir / wf_folder / f"{inst}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Instance JSON not found: {json_path}")

        dag = load_wfcommons_instance(str(json_path))
        total_tasks += len(dag.nodes)

        assign_task_release_times(
            dag,
            enabled=bool(cfg["dynamic"].get("task_release_random", False)),
            T=int(cfg["dynamic"].get("task_release_T", 100)),
            seed=seed,
        )
        wf_list.append((f"{wf_folder}:{inst}", dag, float(at)))

    t0 = time.perf_counter()
    metrics = sim.run(wf_list)
    total_wall = float(time.perf_counter() - t0)

    scenario = "dynamic" if bool(cfg["dynamic"].get("task_release_random", False)) else "static"
    workflow = str(workflows[0]) if workflows else ""
    instance = str(instances[0]) if instances else ""

    _sched_wall = float(getattr(metrics, "scheduler_wallclock_s", 0.0))
    if _sched_wall <= 0.0:
        _sched_wall = float(total_wall)

    _n_tasks = int(getattr(metrics, "n_tasks", 0))
    if _n_tasks <= 0:
        _n_tasks = int(total_tasks)

    _dispatch_count = int(getattr(metrics, "dispatch_count", 0))
    _sched_calls = int(getattr(scheduler, "scheduler_calls", 0))

    if _dispatch_count <= 0:
        if _sched_calls > 0:
            _dispatch_count = _sched_calls
        else:
            _dispatch_count = _n_tasks if _n_tasks > 0 else int(total_tasks)

    _event_count = int(getattr(metrics, "event_count", 0))
    if _event_count <= 0:
        _event_count = _dispatch_count

    _recluster_count = int(getattr(scheduler, "recluster_count", 0))

    out: dict[str, Any] = {
        "scheduler": getattr(scheduler, "name", str(cfg["scheduler"]["name"])),
        "scheduler_family": getattr(scheduler, "scheduler_family", ""),
        "variant": getattr(scheduler, "variant", "full"),
        "scenario": scenario,
        "run_tag": str(cfg.get("run_tag", "")),

        "makespan": float(getattr(metrics, "makespan", 0.0)),
        "total_energy": float(getattr(metrics, "total_energy", 0.0)),
        "total_carbon": float(getattr(metrics, "total_carbon", 0.0)),
        "green_ratio": float(getattr(metrics, "green_ratio", 0.0)),
        "green_energy": float(getattr(metrics, "green_energy", 0.0)),
        "brown_energy": float(getattr(metrics, "brown_energy", 0.0)),
        "avg_utilization": float(getattr(metrics, "avg_utilization", 0.0)),
        "flowtime_sum": float(getattr(metrics, "flowtime_sum", 0.0)),

        "n_tasks": _n_tasks,
        "unfinished_tasks": int(getattr(metrics, "unfinished_tasks", 0)),

        "scheduler_wallclock_s": _sched_wall,
        "scheduler_calls": _sched_calls,
        "dispatch_count": _dispatch_count,
        "event_count": _event_count,
        "recluster_count": _recluster_count,

        "decision_budget_ms": float(getattr(scheduler, "decision_budget_ms", 0.0)),
        "candidate_cap": int(getattr(scheduler, "candidate_cap", 0)),
        "frontier_cap": int(getattr(scheduler, "frontier_cap", 0)),

        "task_release_random": bool(cfg["dynamic"].get("task_release_random", False)),
        "task_release_T": int(cfg["dynamic"].get("task_release_T", 0)),

        "w1": float(cfg["objective"]["w1"]),
        "w2": float(cfg["objective"]["w2"]),
        "w3": float(cfg["objective"]["w3"]),
        "w4": float(cfg["objective"].get("w4", 0.0)),

        "workflow": workflow,
        "family": workflow,
        "instance": instance,
        "instance_scale_tag": _infer_scale_tag(instance),
        "seed": seed,
        "trace_path": str(cfg.get("trace_path", "")),
        "success": 1,
    }

    if "budget_tag" in cfg:
        out["budget_tag"] = str(cfg["budget_tag"])
    if "weight_tag" in cfg:
        out["weight_tag"] = str(cfg["weight_tag"])
    if "study_group" in cfg:
        out["study_group"] = str(cfg["study_group"])

    if not quiet:
        print(out)

    return out


def _make_cfg(
    base_cfg: dict,
    workflow: str,
    instance: str,
    arrival: float,
    scheduler_name: str,
    seed: int,
    weights: dict | None = None,
    budget: dict | None = None,
    run_tag: str = "",
    study_group: str = "",
) -> dict:
    cfg = copy.deepcopy(base_cfg)

    cfg["dataset"]["workflows"] = [workflow]
    cfg["dataset"]["instances"] = [instance]
    cfg["dataset"]["arrivals"] = [arrival]

    cfg["scheduler"]["name"] = scheduler_name
    cfg["seed"] = int(seed)

    if weights is not None:
        cfg["objective"]["w1"] = float(weights["w1"])
        cfg["objective"]["w2"] = float(weights["w2"])
        cfg["objective"]["w3"] = float(weights["w3"])
        cfg["objective"]["w4"] = float(weights["w4"])
        cfg["weight_tag"] = str(weights.get("tag", ""))

    if budget is not None:
        cfg["budget_tag"] = str(budget["tag"])
        cfg.setdefault("scheduler", {}).setdefault("moheft", {})
        cfg["scheduler"]["moheft"]["decision_budget_ms"] = float(budget["decision_budget_ms"])
        cfg["scheduler"]["moheft"]["candidate_cap"] = int(budget["candidate_cap"])
        cfg["scheduler"]["moheft"]["frontier_cap"] = int(budget["frontier_cap"])

    cfg["run_tag"] = run_tag
    cfg["study_group"] = study_group
    return cfg


def _enumerate_study_jobs(cfg: dict) -> List[dict]:
    study = cfg.get("study", {})
    if not study.get("enabled", False):
        raise ValueError("study.enabled is false, but --study was requested.")

    seeds = cfg.get("seeds", [cfg.get("seed", 0)])
    mode = str(study.get("mode", "all")).lower()

    jobs: List[dict] = []

    # -----------------------------
    # Pareto jobs for Fig.6
    # -----------------------------
    if mode in ("pareto", "all"):
        probe_instances = study.get("pareto_probe_instances", [])
        schedulers = study.get("pareto_schedulers", ["HEFT", "GREENHEFT", "MOHEFT", "ECDS"])
        weight_grid = study.get("objective_grid", [])

        for item in probe_instances:
            workflow = str(item["workflow"])
            instance = str(item["instance"])
            arrival = float(item.get("arrival", 0))

            for seed in seeds:
                for sched in schedulers:
                    if sched in ("HEFT",):
                        weights = {"tag": "anchor", "w1": 0.5, "w2": 0.3, "w3": 0.2, "w4": 0.0}
                        cfg_one = _make_cfg(
                            cfg, workflow, instance, arrival, sched, seed,
                            weights=weights,
                            run_tag="pareto_anchor",
                            study_group="pareto",
                        )
                        jobs.append({
                            "cfg": cfg_one,
                            "group": "pareto",
                            "workflow": workflow,
                            "instance": instance,
                            "scheduler": sched,
                            "seed": seed,
                            "weight_tag": weights["tag"],
                            "budget_tag": "",
                        })
                    else:
                        for w in weight_grid:
                            cfg_one = _make_cfg(
                                cfg, workflow, instance, arrival, sched, seed,
                                weights=w,
                                run_tag="pareto_weight_sweep",
                                study_group="pareto",
                            )
                            jobs.append({
                                "cfg": cfg_one,
                                "group": "pareto",
                                "workflow": workflow,
                                "instance": instance,
                                "scheduler": sched,
                                "seed": seed,
                                "weight_tag": str(w.get("tag", "")),
                                "budget_tag": "",
                            })

    # -----------------------------
    # Budget jobs for Fig.7–8
    # -----------------------------
    if mode in ("budget", "all"):
        probe_instances = study.get("budget_probe_instances", [])
        weight_grid = study.get("objective_grid", [])
        budgets = study.get("moheft_budgets", [])

        if not weight_grid:
            raise ValueError("study.objective_grid is empty.")
        mid_weight = weight_grid[min(2, len(weight_grid) - 1)]

        for item in probe_instances:
            workflow = str(item["workflow"])
            instance = str(item["instance"])
            arrival = float(item.get("arrival", 0))

            for seed in seeds:
                cfg_ecds = _make_cfg(
                    cfg, workflow, instance, arrival, "ECDS", seed,
                    weights=mid_weight,
                    run_tag="ecds_budget_ref",
                    study_group="budget",
                )
                cfg_ecds["budget_tag"] = "ecds_ref"
                jobs.append({
                    "cfg": cfg_ecds,
                    "group": "budget",
                    "workflow": workflow,
                    "instance": instance,
                    "scheduler": "ECDS",
                    "seed": seed,
                    "weight_tag": str(mid_weight.get("tag", "")),
                    "budget_tag": "ecds_ref",
                })

                for b in budgets:
                    cfg_mo = _make_cfg(
                        cfg, workflow, instance, arrival, "MOHEFT", seed,
                        weights=mid_weight,
                        budget=b,
                        run_tag=f"moheft_budget_{b['tag']}",
                        study_group="budget",
                    )
                    jobs.append({
                        "cfg": cfg_mo,
                        "group": "budget",
                        "workflow": workflow,
                        "instance": instance,
                        "scheduler": "MOHEFT",
                        "seed": seed,
                        "weight_tag": str(mid_weight.get("tag", "")),
                        "budget_tag": str(b.get("tag", "")),
                    })

    return jobs


def _print_progress_header(total_jobs: int):
    print("=" * 92)
    print(f"[STUDY] Total scheduled runs: {total_jobs}")
    print("=" * 92)


def _print_progress_line(
    idx: int,
    total: int,
    job: dict,
    elapsed_total: float,
    avg_per_run: float,
    eta: float,
):
    group = job["group"]
    instance = job["instance"]
    scheduler = job["scheduler"]
    seed = job["seed"]
    weight_tag = job.get("weight_tag", "")
    budget_tag = job.get("budget_tag", "")

    tag_str = []
    if weight_tag:
        tag_str.append(f"weight={weight_tag}")
    if budget_tag:
        tag_str.append(f"budget={budget_tag}")
    extra = " | ".join(tag_str) if tag_str else "-"

    print(
        f"[{idx:>4}/{total}] "
        f"{group:<6} | "
        f"{instance:<40} | "
        f"{scheduler:<10} | "
        f"seed={seed:<2} | "
        f"{extra} | "
        f"elapsed={_format_seconds(elapsed_total)} | "
        f"avg={_format_seconds(avg_per_run)} | "
        f"eta={_format_seconds(eta)}"
    )


def _print_progress_done(idx: int, total: int, single_elapsed: float, out_row: dict):
    print(
        f"          -> done {idx}/{total} | "
        f"run_time={_format_seconds(single_elapsed)} | "
        f"makespan={out_row.get('makespan', 0.0):.4f} | "
        f"carbon={out_row.get('total_carbon', 0.0):.4f} | "
        f"brown={out_row.get('brown_energy', 0.0):.4f} | "
        f"green_ratio={out_row.get('green_ratio', 0.0):.4f}"
    )


def run_study(cfg: dict, quiet: bool = False):
    study = cfg.get("study", {})
    if not study.get("enabled", False):
        raise ValueError("study.enabled is false, but --study was requested.")

    output_cfg = study.get("outputs", {})
    jobs = _enumerate_study_jobs(cfg)

    total_jobs = len(jobs)
    if total_jobs == 0:
        raise ValueError("No study jobs were generated. Check your study config.")

    if not quiet:
        _print_progress_header(total_jobs)

    pareto_rows: List[dict[str, Any]] = []
    budget_rows: List[dict[str, Any]] = []

    t_study0 = time.perf_counter()

    for idx, job in enumerate(jobs, start=1):
        t_now = time.perf_counter()
        elapsed_total = t_now - t_study0
        avg_per_run = elapsed_total / max(1, idx - 1) if idx > 1 else 0.0
        remaining = total_jobs - idx + 1
        eta = avg_per_run * remaining if idx > 1 else 0.0

        if not quiet:
            _print_progress_line(
                idx=idx,
                total=total_jobs,
                job=job,
                elapsed_total=elapsed_total,
                avg_per_run=avg_per_run,
                eta=eta,
            )

        t0 = time.perf_counter()
        out_row = run_once(job["cfg"], quiet=True)
        single_elapsed = time.perf_counter() - t0

        if job["group"] == "pareto":
            pareto_rows.append(out_row)
        elif job["group"] == "budget":
            budget_rows.append(out_row)
        else:
            raise ValueError(f"Unknown job group: {job['group']}")

        if not quiet:
            _print_progress_done(idx, total_jobs, single_elapsed, out_row)

    if pareto_rows:
        pareto_out = PROJECT_ROOT / output_cfg.get("pareto_raw_csv", "results/study/dynamic_pareto_study_raw.csv")
        pareto_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pareto_rows).to_csv(pareto_out, index=False, encoding="utf-8")
        print("[OK] Saved pareto study raw CSV:", pareto_out)

    if budget_rows:
        budget_out = PROJECT_ROOT / output_cfg.get("budget_raw_csv", "results/study/dynamic_budget_study_raw.csv")
        budget_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(budget_rows).to_csv(budget_out, index=False, encoding="utf-8")
        print("[OK] Saved budget study raw CSV:", budget_out)

    if not quiet:
        total_elapsed = time.perf_counter() - t_study0
        print("=" * 92)
        print(f"[STUDY] Finished all {total_jobs} runs in {_format_seconds(total_elapsed)}")
        print("=" * 92)


def main():
    args = parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    elif "seed" not in cfg:
        if "seeds" in cfg and cfg["seeds"]:
            cfg["seed"] = int(cfg["seeds"][0])
        else:
            cfg["seed"] = 0

    if args.study or bool(cfg.get("study", {}).get("enabled", False)):
        run_study(cfg, quiet=args.quiet)
        return

    out = run_once(cfg, quiet=args.quiet)

    if str(args.out).strip():
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = (PROJECT_ROOT / out_path).resolve()
    else:
        out_path = _auto_out_path(cfg, out["scheduler"], int(cfg["seed"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out]).to_csv(out_path, index=False, encoding="utf-8")

    if not args.quiet:
        print("Saved:", out_path)
        print(out)


if __name__ == "__main__":
    main()