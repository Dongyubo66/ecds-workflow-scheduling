from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import networkx as nx

from .base import SchedulerBase, ReadyTask
from ..resources import Machine


class GreenHEFT(SchedulerBase):
    """
    GreenHEFT baseline:
      - Task priority: HEFT-like upward rank (rank_u)
      - Mapping: pick machine with minimum energy for the task (tie-break by finish time)
    """
    name = "GREENHEFT"

    def __init__(self):
        self.rank_u_cache: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _avg_speed(machines: List[Machine]) -> float:
        return sum(m.speed for m in machines) / max(1e-9, len(machines))

    def _compute_rank_u(self, dag: nx.DiGraph, machines: List[Machine]) -> Dict[str, float]:
        avg_speed = self._avg_speed(machines)
        comp = {n: float(dag.nodes[n].get("runtime", 1.0)) / avg_speed for n in dag.nodes}
        rank_u: Dict[str, float] = {n: 0.0 for n in dag.nodes}
        topo_rev = list(nx.topological_sort(dag))[::-1]
        for n in topo_rev:
            succs = list(dag.successors(n))
            if not succs:
                rank_u[n] = comp[n]
            else:
                rank_u[n] = comp[n] + max(rank_u[s] for s in succs)
        return rank_u

    def on_new_workflow(self, wf_id: str, dag: nx.DiGraph) -> None:
        self.rank_u_cache.pop(wf_id, None)

    def select(self, now, ready_tasks, dags, machines, machine_available_time):
        if not ready_tasks:
            return None

        wf_id = min(ready_tasks, key=lambda rt: (rt.ready_time, rt.wf_id)).wf_id
        dag = dags[wf_id]
        ready_in_wf = [rt for rt in ready_tasks if rt.wf_id == wf_id]
        if not ready_in_wf:
            return None

        if wf_id not in self.rank_u_cache or not self.rank_u_cache[wf_id]:
            self.rank_u_cache[wf_id] = self._compute_rank_u(dag, machines)
        rank_u = self.rank_u_cache[wf_id]

        rt = max(ready_in_wf, key=lambda x: (rank_u.get(x.task_id, 0.0), -x.ready_time))

        runtime = float(dag.nodes[rt.task_id].get("runtime", 1.0))
        best = None
        for m in machines:
            start = max(float(now), float(machine_available_time[m.name]), float(rt.ready_time))
            exec_t = runtime / max(1e-9, m.speed)
            finish = start + exec_t
            e = m.power.energy(m.f, exec_t, exec_t)
            key = (e, finish)
            if best is None or key < best[0]:
                best = (key, m.name)

        return rt, best[1]
