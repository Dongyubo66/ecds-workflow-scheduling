import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from wf.loader import load_wfcommons_instance
from wf.dynamic import assign_task_release_times

from sim.energy import DVFSPowerModel
from sim.resources import Machine, GreenSegment
from sim.simulator import DiscreteEventSimulator
from sim.schedulers.baselines import FCFS, LIST, HEFT_Simple, GREENHEFT, BudgetedMOHEFT, NSGA2
from sim.schedulers.ecds import ECDS, ECDSConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_scheduler(cfg: dict):
    name = str(cfg["scheduler"]["name"]).strip().upper()
    ocfg = cfg.get("objective", {})
    scfg = cfg.get("scheduler", {})
    bcfg = scfg.get("params", {})

    if name == "FCFS":
        return FCFS()
    if name == "LIST":
        return LIST()
    if name == "HEFT":
        return HEFT_Simple()
    if name == "GREENHEFT":
        return GREENHEFT()
    if name in ("MOHEFT", "MOHEFT_BUDGET", "BUDGETED_MOHEFT"):
        return BudgetedMOHEFT(
            w1=float(ocfg.get("w1", 0.25)),
            w2=float(ocfg.get("w2", 0.25)),
            w3=float(ocfg.get("w3", 0.25)),
            w4=float(ocfg.get("w4", 0.25)),
            decision_budget_ms=float(bcfg.get("decision_budget_ms", 30.0)),
            candidate_cap=int(bcfg.get("candidate_cap", 12)),
            frontier_cap=int(bcfg.get("frontier_cap", 24)),
        )
    if name in ("NSGA-II", "NSGA2"):
        return NSGA2(
            pop_size=int(bcfg.get("pop_size", 40)),
            ngen=int(bcfg.get("ngen", 15)),
            cxpb=float(bcfg.get("cxpb", 0.7)),
            mutpb=float(bcfg.get("mutpb", 0.25)),
            seed=int(cfg.get("seed", 0)),
        )
    if name == "ECDS":
        ccfg = scfg.get("clustering", {})
        rcfg = scfg.get("reschedule", {})
        e = ECDSConfig(
            w1=float(ocfg.get("w1", 0.25)),
            w2=float(ocfg.get("w2", 0.25)),
            w3=float(ocfg.get("w3", 0.25)),
            w4=float(ocfg.get("w4", 0.25)),
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/static.yaml", help="配置文件路径")
    parser.add_argument("--out", default="", help="单次运行输出 CSV 路径")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置文件中的 seed")
    parser.add_argument("--tag", default="", help="运行标签，例如 smoke / screen / full")
    parser.add_argument("--scenario", default="", help="覆盖 experiment.scenario")
    parser.add_argument("--trace", action="store_true", help="是否输出 per-dispatch trace")
    parser.add_argument("--trace-path", default="", help="trace 输出 CSV 路径")
    parser.add_argument("--quiet", action="store_true")
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


def _build_machines(cfg: dict) -> list[Machine]:
    sites_cfg = cfg.get("resources", {}).get("sites", {})
    machines: list[Machine] = []
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
    return machines


def main():
    args = parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("experiment", {})

    if args.scenario:
        cfg["experiment"]["scenario"] = str(args.scenario)
    if args.tag:
        cfg["experiment"]["run_tag"] = str(args.tag)

    if args.seed is not None:
        seed = int(args.seed)
    elif "seed" in cfg:
        seed = int(cfg["seed"])
    elif "seeds" in cfg and cfg["seeds"]:
        seed = int(cfg["seeds"][0])
    else:
        seed = 0
    cfg["seed"] = seed

    base_dir = Path(cfg["dataset"]["base_dir"])
    workflows = cfg["dataset"]["workflows"]
    instances = cfg["dataset"]["instances"]
    arrivals = cfg["dataset"]["arrivals"]
    if not (len(workflows) == len(instances) == len(arrivals)):
        raise ValueError(
            f"Length mismatch: workflows={len(workflows)}, instances={len(instances)}, arrivals={len(arrivals)}"
        )

    machines = _build_machines(cfg)
    scheduler = build_scheduler(cfg)

    wf_list = []
    for wf_folder, inst, at in zip(workflows, instances, arrivals):
        json_path = base_dir / wf_folder / f"{inst}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Instance JSON not found: {json_path}")

        dag = load_wfcommons_instance(str(json_path))
        assign_task_release_times(
            dag,
            enabled=bool(cfg["dynamic"].get("task_release_random", False)),
            T=int(cfg["dynamic"].get("task_release_T", 100)),
            seed=seed,
        )
        wf_list.append((f"{wf_folder}:{inst}", dag, float(at)))

    trace_path = None
    if args.trace:
        if args.trace_path.strip():
            trace_path = args.trace_path
        else:
            auto_name = f"trace__{workflows[0].replace('/', '_')}__{instances[0]}__{scheduler.name}__seed{seed}.csv"
            trace_path = str((PROJECT_ROOT / "results" / "traces" / auto_name).resolve())

    sim = DiscreteEventSimulator(
        machines,
        scheduler,
        trace_enabled=bool(args.trace),
        trace_path=trace_path,
    )

    total_tasks = sum(int(dag.number_of_nodes()) for _, dag, _ in wf_list)
    t0 = time.perf_counter()
    metrics = sim.run(wf_list)
    wallclock_s = float(time.perf_counter() - t0)
    metrics.scheduler_wallclock_s = wallclock_s

    sstats = scheduler.get_stats() if hasattr(scheduler, "get_stats") else {}
    metrics.scheduler_calls = int(sstats.get("scheduler_calls", 0))
    metrics.recluster_count = int(sstats.get("recluster_count", 0))

    scenario = str(cfg.get("experiment", {}).get("scenario", "dynamic" if cfg["dynamic"].get("task_release_random", False) else "static"))
    run_tag = str(cfg.get("experiment", {}).get("run_tag", ""))
    variant = str(cfg.get("experiment", {}).get("variant", "full"))
    scheduler_family = str(getattr(scheduler, "family", "baseline"))
    workflow_name = str(workflows[0]) if workflows else ""
    instance_name = str(instances[0]) if instances else ""
    instance_scale_tag = "unknown"
    for tok in ("01d", "05d", "10d", "20d", "25d", "50d"):
        if tok in instance_name:
            instance_scale_tag = tok
            break

    out = {
        "scheduler": scheduler.name,
        "scheduler_family": scheduler_family,
        "variant": variant,
        "scenario": scenario,
        "run_tag": run_tag,
        "makespan": metrics.makespan,
        "total_energy": metrics.total_energy,
        "total_carbon": metrics.total_carbon,
        "green_ratio": metrics.green_ratio,
        "green_energy": metrics.green_energy,
        "brown_energy": metrics.brown_energy,
        "avg_utilization": metrics.avg_utilization,
        "flowtime_sum": metrics.flowtime_sum,
        "n_tasks": int(total_tasks),
        "unfinished_tasks": int(metrics.unfinished_tasks),
        "scheduler_wallclock_s": wallclock_s,
        "scheduler_calls": int(metrics.scheduler_calls),
        "dispatch_count": int(metrics.dispatch_count),
        "event_count": int(metrics.event_count),
        "recluster_count": int(metrics.recluster_count),
        "decision_budget_ms": float(cfg.get("scheduler", {}).get("params", {}).get("decision_budget_ms", 0.0)),
        "candidate_cap": int(cfg.get("scheduler", {}).get("params", {}).get("candidate_cap", 0)),
        "frontier_cap": int(cfg.get("scheduler", {}).get("params", {}).get("frontier_cap", 0)),
        "task_release_random": bool(cfg["dynamic"].get("task_release_random", False)),
        "task_release_T": int(cfg["dynamic"].get("task_release_T", 0)),
        "w1": float(cfg["objective"].get("w1", 0.25)),
        "w2": float(cfg["objective"].get("w2", 0.25)),
        "w3": float(cfg["objective"].get("w3", 0.25)),
        "w4": float(cfg["objective"].get("w4", 0.25)),
        "workflow": workflow_name,
        "family": workflow_name,
        "instance": instance_name,
        "instance_scale_tag": instance_scale_tag,
        "seed": seed,
        "trace_path": trace_path or "",
        "success": 1 if int(metrics.unfinished_tasks) == 0 else 0,
    }

    if str(args.out).strip():
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = (PROJECT_ROOT / out_path).resolve()
    else:
        out_path = _auto_out_path(cfg, scheduler.name, seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out]).to_csv(out_path, index=False, encoding="utf-8")

    if not args.quiet:
        print("Saved:", out_path)
        print(out)


if __name__ == "__main__":
    main()
