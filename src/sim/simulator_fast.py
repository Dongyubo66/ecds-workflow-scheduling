from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
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


class DiscreteEventSimulator:
    """
    Fast event-driven simulator (incremental ready queue):
      - Maintain remaining predecessor counts + max parent finish time
      - Maintain ready heap keyed by ready_time (arrival/release/deps)
      - Avoid full-graph scanning per scheduling decision

    NOTE: This simulator keeps the same "reservation-based" semantics as your original one:
      - Once a task is selected, it is marked started immediately even if start_time > now,
        and the machine availability is pushed to the future.
    """

    def __init__(self, machines: List[Machine], scheduler: SchedulerBase):
        self.machines = machines
        self.scheduler = scheduler

    def run(self, workflows: List[Tuple[str, nx.DiGraph, float]]) -> RunMetrics:
        dags: Dict[str, nx.DiGraph] = {wf_id: dag for wf_id, dag, _ in workflows}
        arrivals: Dict[str, float] = {wf_id: float(at) for wf_id, _, at in workflows}

        # scheduler hook
        for wf_id, dag, _ in workflows:
            self.scheduler.on_new_workflow(wf_id, dag)

        # per-task state
        state: Dict[str, Dict[str, TaskState]] = {
            wf_id: {str(t): TaskState() for t in dag.nodes} for wf_id, dag, _ in workflows
        }

        machine_avail: Dict[str, float] = {m.name: 0.0 for m in self.machines}
        busy_intervals: Dict[str, List[tuple]] = {m.name: [] for m in self.machines}
        machine_map: Dict[str, Machine] = {m.name: m for m in self.machines}

        # --- incremental dep tracking ---
        rem_preds: Dict[str, Dict[str, int]] = {}
        parent_finish_max: Dict[str, Dict[str, float]] = {}

        # heaps
        ready_heap: List[Tuple[float, str, str]] = []      # (ready_time, wf_id, task_id)
        finish_heap: List[Tuple[float, str, str, str]] = []  # (finish_time, wf_id, task_id, machine)

        total_tasks = 0

        # initialize dep counters + initial ready nodes (indegree=0)
        for wf_id, dag in dags.items():
            total_tasks += dag.number_of_nodes()
            rem_preds[wf_id] = {str(t): int(dag.in_degree(t)) for t in dag.nodes}
            parent_finish_max[wf_id] = {str(t): 0.0 for t in dag.nodes}

            for t in dag.nodes:
                tid = str(t)
                if rem_preds[wf_id][tid] == 0:
                    rel = float(dag.nodes[tid].get("release_time", 0.0))
                    rt = max(arrivals[wf_id], rel, 0.0)
                    heapq.heappush(ready_heap, (rt, wf_id, tid))

        finished_cnt = 0
        now = 0.0

        def pop_ready(now_t: float) -> List[ReadyTask]:
            """Pop all tasks whose ready_time <= now."""
            out: List[ReadyTask] = []
            while ready_heap and float(ready_heap[0][0]) <= float(now_t) + 1e-9:
                rt, wf_id, task_id = heapq.heappop(ready_heap)
                ts = state[wf_id][task_id]
                if ts.started or ts.finished:
                    continue
                out.append(ReadyTask(task_id=task_id, wf_id=wf_id, ready_time=float(rt)))
            return out

        def push_back_ready(ready_list: List[ReadyTask]) -> None:
            """IMPORTANT: put unscheduled ready tasks back to heap, otherwise they are lost."""
            if not ready_list:
                return
            for rt in ready_list:
                ts = state[rt.wf_id][rt.task_id]
                if not ts.started and not ts.finished:
                    heapq.heappush(ready_heap, (float(rt.ready_time), rt.wf_id, rt.task_id))
            ready_list.clear()

        while finished_cnt < total_tasks:
            # 1) process all FINISH events at or before now
            while finish_heap and float(finish_heap[0][0]) <= float(now) + 1e-9:
                tfin, wf_id, task_id, _m = heapq.heappop(finish_heap)
                st = state[wf_id][task_id]
                if st.finished:
                    continue
                st.finished = True
                st.finish_time = float(tfin)
                finished_cnt += 1

                dag = dags[wf_id]
                # update successors' dep counts
                for succ in dag.successors(task_id):
                    sid = str(succ)
                    if state[wf_id][sid].finished or state[wf_id][sid].started:
                        continue
                    parent_finish_max[wf_id][sid] = max(parent_finish_max[wf_id][sid], float(tfin))
                    rem_preds[wf_id][sid] -= 1
                    if rem_preds[wf_id][sid] == 0:
                        rel = float(dag.nodes[sid].get("release_time", 0.0))
                        rt = max(arrivals[wf_id], rel, parent_finish_max[wf_id][sid])
                        heapq.heappush(ready_heap, (rt, wf_id, sid))

            # 2) collect ready tasks at current time
            ready = pop_ready(now)

            # 3) if nothing ready, advance time to next event
            if not ready:
                nxt = None
                if finish_heap:
                    nxt = float(finish_heap[0][0])
                if ready_heap:
                    nxt = float(ready_heap[0][0]) if nxt is None else min(nxt, float(ready_heap[0][0]))
                if nxt is None:
                    break
                now = float(nxt)
                continue

            # 4) optional reschedule trigger
            if hasattr(self.scheduler, "maybe_reschedule"):
                per_wf_cnt: Dict[str, int] = {}
                for rt in ready:
                    per_wf_cnt[rt.wf_id] = per_wf_cnt.get(rt.wf_id, 0) + 1
                for _wf_id, cnt in per_wf_cnt.items():
                    try:
                        self.scheduler.maybe_reschedule(_wf_id, cnt, now)
                    except TypeError:
                        self.scheduler.maybe_reschedule(_wf_id, cnt)

            # 5) schedule tasks (reservation-based)
            idle_requested = False
            while ready:
                decision = self.scheduler.select(now, ready, dags, self.machines, machine_avail)
                if decision is None:
                    idle_requested = True
                    break

                rt, mname = decision
                if mname not in machine_avail:
                    # invalid decision -> treat as idle
                    idle_requested = True
                    break
                if state[rt.wf_id][rt.task_id].started or state[rt.wf_id][rt.task_id].finished:
                    idle_requested = True
                    break

                dag = dags[rt.wf_id]
                runtime = float(dag.nodes[rt.task_id].get("runtime", 1.0))
                m = machine_map[mname]

                start = max(float(now), float(machine_avail[m.name]), float(rt.ready_time))
                exec_t = runtime / max(1e-9, float(m.speed))
                finish = start + exec_t

                st = state[rt.wf_id][rt.task_id]
                st.started = True
                st.start_time = start
                st.finish_time = finish
                st.machine = m.name

                busy_intervals[m.name].append((start, finish))
                machine_avail[m.name] = finish
                heapq.heappush(finish_heap, (finish, rt.wf_id, rt.task_id, m.name))

                # remove chosen task from ready list
                removed = False
                for i, x in enumerate(ready):
                    if x.wf_id == rt.wf_id and x.task_id == rt.task_id:
                        ready.pop(i)
                        removed = True
                        break
                if not removed:
                    # decision task not in ready list -> treat as idle to avoid looping
                    idle_requested = True
                    break

            # 6) push back unscheduled ready tasks (CRITICAL FIX)
            push_back_ready(ready)

            # 7) if scheduler chose to idle while ready tasks exist, we must advance time
            #    otherwise we'd keep popping the same ready tasks at the same "now" forever.
            if idle_requested:
                nxt = None
                if finish_heap:
                    nxt = float(finish_heap[0][0])
                # also consider next future ready time strictly > now
                if ready_heap:
                    # ready_heap[0] could be <= now (because we pushed back), skip those
                    # find smallest ready_time > now (heap doesn't support peek-next easily; do a small scan)
                    # but ready_heap can be large; so prefer finish event if exists
                    if nxt is None:
                        # fallback: scan a few top elements by popping temporarily
                        tmp = []
                        while ready_heap and float(ready_heap[0][0]) <= float(now) + 1e-9 and len(tmp) < 50:
                            tmp.append(heapq.heappop(ready_heap))
                        if ready_heap:
                            nxt = float(ready_heap[0][0])
                        for item in tmp:
                            heapq.heappush(ready_heap, item)

                if nxt is None:
                    # nothing to advance to -> deadlock
                    break
                if nxt > float(now) + 1e-9:
                    now = float(nxt)
                else:
                    # as a safeguard, advance to the next finish if possible
                    if finish_heap and float(finish_heap[0][0]) > float(now) + 1e-9:
                        now = float(finish_heap[0][0])

        # ---- collect metrics (same as your original) ----
        metrics = RunMetrics()
        all_finishes: List[float] = []
        energy_sum = 0.0
        green_sum = 0.0
        brown_sum = 0.0
        carbon_sum = 0.0
        flow_sum = 0.0

        unfinished = 0
        for wf_id, dag in dags.items():
            for t in dag.nodes:
                tid = str(t)
                st = state[wf_id][tid]
                metrics.task_start[f"{wf_id}:{tid}"] = st.start_time
                metrics.task_finish[f"{wf_id}:{tid}"] = st.finish_time

                if not st.finished or st.machine == "":
                    unfinished += 1
                    continue

                all_finishes.append(st.finish_time)

                runtime = float(dag.nodes[tid].get("runtime", 1.0))
                m = machine_map.get(st.machine)
                if m is None:
                    unfinished += 1
                    continue

                exec_t = runtime / max(1e-9, float(m.speed))
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
        return metrics