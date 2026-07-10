from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import heapq
import networkx as nx

from .resources import Machine
from .metrics import RunMetrics, compute_utilization
from .schedulers.base import SchedulerBase, ReadyTask


@dataclass
class TaskState:
    started: bool = False
    finished: bool = False
    start_time: float = 0.0
    finish_time: float = 0.0
    machine: str = ""


@dataclass(frozen=True)
class SimContext:
    now: float
    dags: Dict[str, nx.DiGraph]
    arrivals: Dict[str, float]
    state: Dict[str, Dict[str, TaskState]]
    machine_avail: Dict[str, float]
    machines: List[Machine]


class DiscreteEventSimulator:
    def __init__(
        self,
        machines: List[Machine],
        scheduler: SchedulerBase,
        trace_enabled: bool = False,
        trace_path: str | None = None,
    ):
        self.machines = machines
        self.scheduler = scheduler
        self.trace_enabled = bool(trace_enabled)
        self.trace_path = trace_path

    def _write_trace(self, rows: List[dict]) -> None:
        if not self.trace_enabled or not self.trace_path:
            return
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            rows = [{
                "now": 0.0,
                "wf_id": "",
                "task_id": "",
                "machine": "",
                "ready_len": 0,
                "start": 0.0,
                "finish": 0.0,
                "runtime": 0.0,
                "green_fraction": 0.0,
                "brown_energy": 0.0,
                "total_energy": 0.0,
                "scheduler": getattr(self.scheduler, "name", "UNKNOWN"),
                "decision_index": 0,
            }]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def run(self, workflows: List[Tuple[str, nx.DiGraph, float]]) -> RunMetrics:
        dags: Dict[str, nx.DiGraph] = {wf_id: dag for wf_id, dag, _ in workflows}
        arrivals: Dict[str, float] = {wf_id: at for wf_id, _, at in workflows}

        for wf_id, dag, _ in workflows:
            self.scheduler.on_new_workflow(wf_id, dag)

        state: Dict[str, Dict[str, TaskState]] = {
            wf_id: {t: TaskState() for t in dag.nodes} for wf_id, dag, _ in workflows
        }

        machine_avail = {m.name: 0.0 for m in self.machines}
        busy_intervals = {m.name: [] for m in self.machines}

        evq: List[Tuple[float, str, str, str, str]] = []
        for wf_id, _, at in workflows:
            heapq.heappush(evq, (float(at), "ARRIVAL", wf_id, "", ""))

        now = 0.0
        event_count = 0
        dispatch_count = 0
        trace_rows: List[dict] = []

        def deps_done(wf_id: str, t: str) -> bool:
            dag = dags[wf_id]
            return all(state[wf_id][p].finished for p in dag.predecessors(t))

        def task_ready_time(wf_id: str, t: str) -> float:
            dag = dags[wf_id]
            rel = float(dag.nodes[t].get("release_time", 0.0))
            parent_finish = 0.0
            for p in dag.predecessors(t):
                parent_finish = max(parent_finish, state[wf_id][p].finish_time)
            return max(float(arrivals[wf_id]), rel, parent_finish)

        def collect_ready(now_t: float) -> List[ReadyTask]:
            rts: List[ReadyTask] = []
            for wf_id, dag in dags.items():
                for t in dag.nodes:
                    ts = state[wf_id][t]
                    if ts.started or ts.finished:
                        continue
                    if deps_done(wf_id, t):
                        rt = task_ready_time(wf_id, t)
                        if rt <= now_t + 1e-9:
                            rts.append(ReadyTask(task_id=t, wf_id=wf_id, ready_time=rt))
            return rts

        def all_done() -> bool:
            for wf_id, dag in dags.items():
                for t in dag.nodes:
                    if not state[wf_id][t].finished:
                        return False
            return True

        def next_future_ready_time(now_t: float) -> Optional[float]:
            nxt = None
            for wf_id, dag in dags.items():
                for t in dag.nodes:
                    ts = state[wf_id][t]
                    if ts.started or ts.finished:
                        continue
                    if deps_done(wf_id, t):
                        rt = task_ready_time(wf_id, t)
                        if rt > now_t + 1e-9:
                            if nxt is None or rt < nxt:
                                nxt = rt
            return nxt

        while evq or (not all_done()):
            if evq:
                time, etype, wf_id, task_id, _machine = heapq.heappop(evq)
                event_count += 1
                now = float(time)
                if etype == "FINISH":
                    st = state[wf_id][task_id]
                    st.finished = True
                    st.finish_time = now
            else:
                nt = next_future_ready_time(now)
                if nt is None:
                    break
                now = float(nt)

            while True:
                ready = collect_ready(now)
                if not ready:
                    break

                if hasattr(self.scheduler, "maybe_reschedule"):
                    per_wf_cnt: Dict[str, int] = {}
                    for rt in ready:
                        per_wf_cnt[rt.wf_id] = per_wf_cnt.get(rt.wf_id, 0) + 1
                    for _wf_id, cnt in per_wf_cnt.items():
                        try:
                            self.scheduler.maybe_reschedule(_wf_id, cnt, now)
                        except TypeError:
                            self.scheduler.maybe_reschedule(_wf_id, cnt)

                ctx = SimContext(
                    now=now,
                    dags=dags,
                    arrivals=arrivals,
                    state=state,
                    machine_avail=machine_avail,
                    machines=self.machines,
                )

                try:
                    decision = self.scheduler.select(now, ready, dags, self.machines, machine_avail, ctx)
                except TypeError:
                    decision = self.scheduler.select(now, ready, dags, self.machines, machine_avail)

                if decision is None:
                    break

                rt, mname = decision
                if state[rt.wf_id][rt.task_id].started:
                    break
                if mname not in machine_avail:
                    break

                dag = dags[rt.wf_id]
                runtime = float(dag.nodes[rt.task_id].get("runtime", 1.0))
                m = next(mm for mm in self.machines if mm.name == mname)

                start = max(now, machine_avail[m.name], rt.ready_time)
                exec_t = runtime / max(1e-9, m.speed)
                finish = start + exec_t
                e = m.power.energy(m.f, exec_t, exec_t)
                gbar = m.avg_green_fraction(start, finish)
                be = (1.0 - gbar) * e

                st = state[rt.wf_id][rt.task_id]
                st.started = True
                st.start_time = start
                st.finish_time = finish
                st.machine = m.name

                busy_intervals[m.name].append((start, finish))
                machine_avail[m.name] = finish
                heapq.heappush(evq, (finish, "FINISH", rt.wf_id, rt.task_id, m.name))
                dispatch_count += 1

                if self.trace_enabled:
                    trace_rows.append(
                        {
                            "now": float(now),
                            "wf_id": str(rt.wf_id),
                            "task_id": str(rt.task_id),
                            "machine": str(m.name),
                            "ready_len": int(len(ready)),
                            "start": float(start),
                            "finish": float(finish),
                            "runtime": float(runtime),
                            "green_fraction": float(gbar),
                            "brown_energy": float(be),
                            "total_energy": float(e),
                            "scheduler": getattr(self.scheduler, "name", "UNKNOWN"),
                            "decision_index": int(dispatch_count),
                        }
                    )

            if not evq and (not all_done()):
                nt = next_future_ready_time(now)
                if nt is not None and nt > now + 1e-9:
                    heapq.heappush(evq, (nt, "ARRIVAL", "", "", ""))

        metrics = RunMetrics()
        all_finishes: List[float] = []
        energy_sum = 0.0
        green_sum = 0.0
        brown_sum = 0.0
        carbon_sum = 0.0
        flow_sum = 0.0

        machine_map = {m.name: m for m in self.machines}
        unfinished = 0
        n_tasks = 0

        for wf_id, dag in dags.items():
            for t in dag.nodes:
                n_tasks += 1
                st = state[wf_id][t]
                metrics.task_start[f"{wf_id}:{t}"] = st.start_time
                metrics.task_finish[f"{wf_id}:{t}"] = st.finish_time

                if not st.finished or st.machine == "":
                    unfinished += 1
                    continue

                all_finishes.append(st.finish_time)

                runtime = float(dag.nodes[t].get("runtime", 1.0))
                m = machine_map.get(st.machine)
                if m is None:
                    unfinished += 1
                    continue

                exec_t = runtime / max(1e-9, m.speed)
                e = m.power.energy(m.f, exec_t, exec_t)
                gbar = m.avg_green_fraction(st.start_time, st.finish_time)
                ge = gbar * e
                be = (1.0 - gbar) * e

                energy_sum += e
                green_sum += ge
                brown_sum += be
                carbon_sum += ge * float(m.ci_green) + be * float(m.ci_brown)
                flow_sum += (st.finish_time - float(arrivals[wf_id]))

        if unfinished > 0:
            print(f"[WARN] {unfinished} tasks unfinished or missing machine assignment. Metrics may be incomplete.")

        metrics.makespan = max(all_finishes) if all_finishes else 0.0
        metrics.total_energy = energy_sum
        metrics.green_energy = green_sum
        metrics.brown_energy = brown_sum
        metrics.green_ratio = (green_sum / energy_sum) if energy_sum > 1e-12 else 0.0
        metrics.total_carbon = carbon_sum
        metrics.flowtime_sum = flow_sum
        metrics.avg_utilization = compute_utilization(busy_intervals, metrics.makespan)

        metrics.n_tasks = int(n_tasks)
        metrics.unfinished_tasks = int(unfinished)
        metrics.event_count = int(event_count)
        metrics.dispatch_count = int(dispatch_count)

        self._write_trace(trace_rows)
        return metrics
