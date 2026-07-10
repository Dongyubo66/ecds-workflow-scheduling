from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import networkx as nx
import numpy as np

from .base import SchedulerBase, ReadyTask
from ..resources import Machine
from ..pareto import truncate_by_pareto_and_crowding


@dataclass(frozen=True)
class MOHEFTRHCConfig:
    K: int = 20                # keep K trade-off partial schedules
    horizon_steps: int = 15    # plan for next H dispatches
    replan_every: int = 5      # reuse plan for N real decisions

    # preference for picking one schedule from Pareto set (scalarization)
    pref_time: float = 0.5
    pref_energy: float = 0.5

    # include brown energy in planning objective (optional extension)
    use_brown: bool = False
    pref_brown: float = 0.0

    # --------- NEW: anti-stall / anti-explosion knobs ----------
    # time budget for one planning call (seconds). If exceeded, stop planning & return best-so-far.
    time_budget_sec: float = 0.5

    # limit candidate tasks considered by planner (Top-L ready tasks at the moment of replan)
    # This is the single most effective speed knob on 05d/10d.
    max_ready_candidates: int = 30


@dataclass
class _PlanState:
    machine_avail: Dict[str, float]
    finish_time: Dict[str, float]     # task -> finish time (for deps)
    started: Dict[str, bool]          # task -> started in plan
    finished: Dict[str, bool]         # task -> finished in plan (within horizon)
    obj: np.ndarray                   # accumulated objectives
    actions: List[Tuple[str, str]]    # [(task_id, machine_name), ...]


class MOHEFT_RHC(SchedulerBase):
    """
    MOHEFT-inspired Pareto list scheduling adapted to your online simulator via receding horizon planning.

    IMPORTANT (practical baseline):
      - This implementation enforces (A) time_budget and (B) candidate truncation to ensure it is runnable
        on medium/large instances (05d/10d) in an online setting.
      - In papers, name it "MOHEFT-Budget" or explicitly report these limits.
    """
    name = "MOHEFT"

    def __init__(self, cfg: MOHEFTRHCConfig):
        self.cfg = cfg
        self._rank_u_cache: Dict[str, Dict[str, float]] = {}
        self._plan_cache: Dict[str, List[Tuple[str, str]]] = {}
        self._plan_pos: Dict[str, int] = {}
        self._since_replan: Dict[str, int] = {}

    # ---- HEFT rank_u ----
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
        self._rank_u_cache.pop(wf_id, None)
        self._plan_cache.pop(wf_id, None)
        self._plan_pos[wf_id] = 0
        self._since_replan[wf_id] = 10**9

    # ---- horizon rollout helpers ----
    def _deps_ready_time(
        self,
        dag: nx.DiGraph,
        arrivals: Dict[str, float],
        wf_id: str,
        t: str,
        finish_time: Dict[str, float],
    ) -> float:
        rel = float(dag.nodes[t].get("release_time", 0.0))
        parent_finish = 0.0
        for p in dag.predecessors(t):
            parent_finish = max(parent_finish, float(finish_time.get(p, 0.0)))
        return max(float(arrivals.get(wf_id, 0.0)), rel, parent_finish)

    def _collect_ready_plan(
        self,
        dag: nx.DiGraph,
        arrivals: Dict[str, float],
        wf_id: str,
        now: float,
        started: Dict[str, bool],
        finished: Dict[str, bool],
        finish_time: Dict[str, float],
        candidate_tasks: List[str],
    ) -> List[str]:
        # CRITICAL SPEEDUP: only scan candidate_tasks (Top-L ready at replan time)
        ready: List[str] = []
        for t in candidate_tasks:
            if started.get(t, False) or finished.get(t, False):
                continue

            # deps done if all predecessors have a finish_time assigned (finished in reality or planned)
            deps_ok = True
            for p in dag.predecessors(t):
                if p not in finish_time:
                    deps_ok = False
                    break
            if not deps_ok:
                continue

            rt = self._deps_ready_time(dag, arrivals, wf_id, t, finish_time)
            if rt <= now + 1e-9:
                ready.append(t)
        return ready

    def _next_future_ready_time_plan(
        self,
        dag: nx.DiGraph,
        arrivals: Dict[str, float],
        wf_id: str,
        now: float,
        started: Dict[str, bool],
        finished: Dict[str, bool],
        finish_time: Dict[str, float],
        candidate_tasks: List[str],
    ) -> Optional[float]:
        nxt = None
        for t in candidate_tasks:
            if started.get(t, False) or finished.get(t, False):
                continue
            deps_ok = True
            for p in dag.predecessors(t):
                if p not in finish_time:
                    deps_ok = False
                    break
            if not deps_ok:
                continue
            rt = self._deps_ready_time(dag, arrivals, wf_id, t, finish_time)
            if rt > now + 1e-9:
                if nxt is None or rt < nxt:
                    nxt = rt
        return nxt

    def _fallback_greedy(
        self,
        now: float,
        ready_tasks: List[ReadyTask],
        dag: nx.DiGraph,
        machines: List[Machine],
        machine_available_time: Dict[str, float],
        wf_id: str,
    ) -> Optional[Tuple[ReadyTask, str]]:
        """Always-progress fallback (HEFT-like criticality + earliest available machine)."""
        if wf_id not in self._rank_u_cache or not self._rank_u_cache[wf_id]:
            self._rank_u_cache[wf_id] = self._compute_rank_u(dag, machines)
        rank_u = self._rank_u_cache[wf_id]

        ready_in_wf = [rt for rt in ready_tasks if rt.wf_id == wf_id]
        if not ready_in_wf:
            return None

        rt_best = max(ready_in_wf, key=lambda rt: (rank_u.get(rt.task_id, 0.0), -rt.ready_time))
        m = min(machines, key=lambda mm: float(machine_available_time[mm.name]))
        return rt_best, m.name

    def _plan(
        self,
        wf_id: str,
        dag: nx.DiGraph,
        arrivals: Dict[str, float],
        machines: List[Machine],
        machine_avail: Dict[str, float],
        now: float,
        initial_started: Dict[str, bool],
        initial_finished: Dict[str, bool],
        initial_finish_time: Dict[str, float],
        candidate_tasks: List[str],
        t0: float,
        time_budget_sec: float,
    ) -> List[Tuple[str, str]]:
        # objectives: [makespan, energy] or [makespan, energy, brown]
        obj_dim = 3 if self.cfg.use_brown else 2

        # start with one plan-state; use fast shallow copies for numeric dicts
        S: List[_PlanState] = [
            _PlanState(
                machine_avail=dict(machine_avail),
                finish_time=dict(initial_finish_time),
                started=dict(initial_started),
                finished=dict(initial_finished),
                obj=np.zeros(obj_dim, dtype=float),
                actions=[],
            )
        ]

        if wf_id not in self._rank_u_cache or not self._rank_u_cache[wf_id]:
            self._rank_u_cache[wf_id] = self._compute_rank_u(dag, machines)
        rank_u = self._rank_u_cache[wf_id]

        t_cursor = float(now)

        for _step in range(int(self.cfg.horizon_steps)):
            # A: time budget check
            if time.perf_counter() - t0 > time_budget_sec:
                break

            expanded: List[_PlanState] = []

            for ps in S:
                if time.perf_counter() - t0 > time_budget_sec:
                    break

                ready = self._collect_ready_plan(
                    dag, arrivals, wf_id, t_cursor,
                    ps.started, ps.finished, ps.finish_time,
                    candidate_tasks=candidate_tasks
                )
                if not ready:
                    nt = self._next_future_ready_time_plan(
                        dag, arrivals, wf_id, t_cursor,
                        ps.started, ps.finished, ps.finish_time,
                        candidate_tasks=candidate_tasks
                    )
                    if nt is None:
                        continue
                    ready = self._collect_ready_plan(
                        dag, arrivals, wf_id, nt,
                        ps.started, ps.finished, ps.finish_time,
                        candidate_tasks=candidate_tasks
                    )
                    t_local = nt
                else:
                    t_local = t_cursor

                if not ready:
                    continue

                # choose task by HEFT criticality among currently ready tasks
                task = max(ready, key=lambda x: rank_u.get(x, 0.0))

                runtime = float(dag.nodes[task].get("runtime", 1.0))
                for m in machines:
                    if time.perf_counter() - t0 > time_budget_sec:
                        break

                    start = max(
                        float(t_local),
                        float(ps.machine_avail[m.name]),
                        self._deps_ready_time(dag, arrivals, wf_id, task, ps.finish_time),
                    )
                    exec_t = runtime / max(1e-9, float(m.speed))
                    finish = start + exec_t

                    e = m.power.energy(m.f, exec_t, exec_t)

                    ps2 = _PlanState(
                        machine_avail=dict(ps.machine_avail),
                        finish_time=dict(ps.finish_time),
                        started=dict(ps.started),
                        finished=dict(ps.finished),
                        obj=ps.obj.copy(),
                        actions=list(ps.actions),
                    )

                    ps2.started[task] = True
                    ps2.finished[task] = True
                    ps2.finish_time[task] = float(finish)
                    ps2.machine_avail[m.name] = float(finish)
                    ps2.actions.append((task, m.name))

                    # update objectives
                    makespan = max(ps2.machine_avail.values()) if ps2.machine_avail else float(finish)
                    ps2.obj[0] = float(makespan)
                    ps2.obj[1] = float(ps2.obj[1] + e)

                    if self.cfg.use_brown:
                        gbar = m.avg_green_fraction(start, finish)
                        brown = (1.0 - gbar) * e
                        ps2.obj[2] = float(ps2.obj[2] + brown)

                    expanded.append(ps2)

            if not expanded:
                break

            F = np.array([ps.obj for ps in expanded], dtype=float)
            keep = truncate_by_pareto_and_crowding(F, int(self.cfg.K))
            S = [expanded[i] for i in keep]

        # pick one schedule from S by scalarization preference
        if not S:
            return []

        F = np.array([ps.obj for ps in S], dtype=float)
        mn = F.min(axis=0)
        mx = F.max(axis=0)
        denom = np.maximum(1e-12, mx - mn)
        Fn = (F - mn) / denom

        best_idx = 0
        best_val = None
        for i in range(len(S)):
            val = float(self.cfg.pref_time) * float(Fn[i, 0]) + float(self.cfg.pref_energy) * float(Fn[i, 1])
            if self.cfg.use_brown:
                val += float(self.cfg.pref_brown) * float(Fn[i, 2])
            if best_val is None or val < best_val:
                best_val = val
                best_idx = i

        return S[best_idx].actions

    def select(self, now, ready_tasks, dags, machines, machine_available_time, ctx=None):
        if not ready_tasks:
            return None

        wf_id = min(ready_tasks, key=lambda rt: (rt.ready_time, rt.wf_id)).wf_id
        dag = dags[wf_id]

        # map for correct ready_time return
        rt_map = {(rt.wf_id, rt.task_id): rt for rt in ready_tasks}

        # reuse cached plan
        since = self._since_replan.get(wf_id, 10**9)
        plan = self._plan_cache.get(wf_id)
        pos = self._plan_pos.get(wf_id, 0)

        need_replan = (plan is None) or (since >= int(self.cfg.replan_every)) or (pos >= len(plan))

        if need_replan:
            # --- candidate truncation (B) ---
            # Only consider Top-L ready tasks (within this workflow) at the moment of planning.
            L = int(getattr(self.cfg, "max_ready_candidates", 30))
            ready_in_wf = [rt for rt in ready_tasks if rt.wf_id == wf_id]
            if L > 0 and len(ready_in_wf) > L:
                # prioritize earliest-ready; could also use rank_u here (but rank_u needs cache)
                ready_in_wf = sorted(ready_in_wf, key=lambda rt: (rt.ready_time, rt.task_id))[:L]
            candidate_tasks = [rt.task_id for rt in ready_in_wf]

            # build initial plan-state from ctx if available
            if ctx is not None:
                st_map = ctx.state[wf_id]
                initial_started = {t: st_map[t].started for t in dag.nodes}
                initial_finished = {t: st_map[t].finished for t in dag.nodes}
                initial_finish_time = {}
                for t in dag.nodes:
                    if st_map[t].finished or st_map[t].started:
                        initial_finish_time[t] = float(st_map[t].finish_time)
            else:
                initial_started = {t: False for t in dag.nodes}
                initial_finished = {t: False for t in dag.nodes}
                initial_finish_time = {}

            # A: time budget
            t0 = time.perf_counter()
            budget = float(getattr(self.cfg, "time_budget_sec", 0.5))

            actions = self._plan(
                wf_id=wf_id,
                dag=dag,
                arrivals=(ctx.arrivals if ctx is not None else {wf_id: 0.0}),
                machines=machines,
                machine_avail=machine_available_time,
                now=float(now),
                initial_started=initial_started,
                initial_finished=initial_finished,
                initial_finish_time=initial_finish_time,
                candidate_tasks=candidate_tasks,
                t0=t0,
                time_budget_sec=budget,
            )

            self._plan_cache[wf_id] = actions
            self._plan_pos[wf_id] = 0
            self._since_replan[wf_id] = 0
            plan = actions
            pos = 0
        else:
            self._since_replan[wf_id] = since + 1

        # execute first feasible action in plan
        ready_set = set(rt_map.keys())
        while plan is not None and pos < len(plan):
            task_id, mname = plan[pos]
            pos += 1
            self._plan_pos[wf_id] = pos
            if (wf_id, task_id) in ready_set:
                # return the real ReadyTask with real ready_time
                return rt_map[(wf_id, task_id)], mname

        # If plan fails / empty, fallback (ensures no stalling)
        fb = self._fallback_greedy(float(now), ready_tasks, dag, machines, machine_available_time, wf_id)
        return fb