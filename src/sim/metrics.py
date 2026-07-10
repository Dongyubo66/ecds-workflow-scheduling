from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RunMetrics:
    makespan: float = 0.0
    total_energy: float = 0.0

    # ---- green & carbon ----
    total_carbon: float = 0.0
    green_ratio: float = 0.0
    green_energy: float = 0.0
    brown_energy: float = 0.0

    avg_utilization: float = 0.0
    flowtime_sum: float = 0.0

    # ---- audit / runtime ----
    n_tasks: int = 0
    unfinished_tasks: int = 0
    scheduler_wallclock_s: float = 0.0
    scheduler_calls: int = 0
    dispatch_count: int = 0
    event_count: int = 0
    recluster_count: int = 0

    # ---- task-level bookkeeping ----
    task_start: Dict[str, float] = field(default_factory=dict)
    task_finish: Dict[str, float] = field(default_factory=dict)


def compute_utilization(machine_busy_intervals: Dict[str, List[tuple]], makespan: float) -> float:
    if makespan <= 0:
        return 0.0
    total_busy = 0.0
    for intervals in machine_busy_intervals.values():
        for s, e in intervals:
            total_busy += max(0.0, e - s)
    m = max(1, len(machine_busy_intervals))
    return total_busy / (m * makespan)
