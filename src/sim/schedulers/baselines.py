from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import random
import copy
import time

import networkx as nx
import numpy as np

from .base import SchedulerBase, ReadyTask
from ..resources import Machine

# DEAP (NSGA-II)
from deap import base as deap_base, creator, tools


# -----------------------------
# Helpers
# -----------------------------

def _minmax_norm(vals: List[float]) -> List[float]:
    if not vals:
        return []
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-12:
        return [0.0] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]


def _compute_rank_u(dag: nx.DiGraph, machines: List[Machine]) -> Dict[str, float]:
    """HEFT-style upward rank as a criticality proxy (comm ignored)."""
    avg_speed = sum(m.speed for m in machines) / max(1e-9, len(machines))
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


def _pick_wf_id(ready_tasks: List[ReadyTask]) -> str:
    """Stable policy: earliest-ready workflow first."""
    return min(ready_tasks, key=lambda rt: (rt.ready_time, rt.wf_id)).wf_id


def _estimate_action(
    now: float,
    rt: ReadyTask,
    dag: nx.DiGraph,
    m: Machine,
    machine_available_time: Dict[str, float],
) -> Tuple[float, float, float, float, float]:
    """
    Return:
      finish, energy, green_energy, brown_energy, runtime
    """
    runtime = float(dag.nodes[rt.task_id].get("runtime", 1.0))
    start = max(float(now), float(machine_available_time[m.name]), float(rt.ready_time))
    exec_t = runtime / max(1e-9, m.speed)
    finish = start + exec_t

    E = float(m.power.energy(m.f, exec_t, exec_t))
    gbar = float(m.avg_green_fraction(start, finish))
    gE = gbar * E
    bE = max(0.0, E - gE)

    return float(finish), float(E), float(gE), float(bE), float(runtime)


def _decode_action(
    individual: List[float],
    now: float,
    ready_in_wf: List[ReadyTask],
    dag: nx.DiGraph,
    machines: List[Machine],
    machine_available_time: Dict[str, float],
    rank_u: Dict[str, float],
) -> Tuple[ReadyTask, str, float, float, float, float]:
    """
    Decode an individual into (task, machine) decision.

    Individual layout (6 floats):
      - task weights:     a,b,c
      - machine weights:  d,e,f

    Task score (maximize):
        a*rank_u_n + b*(1-runtime_n) + c*wait_n
    Machine cost (minimize):
        d*finish_n + e*energy_n + f*(1-green_n)

    Objectives (minimize):
      1) makespan proxy: finish time of selected task on selected machine
      2) energy of the selected task
      3) neg_green: -green_energy   (equiv maximize green_energy)
    """
    a, b, c, d, e, f = [float(x) for x in individual[:6]]

    runtimes = [float(dag.nodes[rt.task_id].get("runtime", 1.0)) for rt in ready_in_wf]
    waits = [max(0.0, float(now) - float(rt.ready_time)) for rt in ready_in_wf]
    ranks = [float(rank_u.get(rt.task_id, 0.0)) for rt in ready_in_wf]

    rt_n = _minmax_norm(runtimes)
    wait_n = _minmax_norm(waits)
    rank_n = _minmax_norm(ranks)

    best_i = 0
    best_score = None
    for i in range(len(ready_in_wf)):
        score = a * rank_n[i] + b * (1.0 - rt_n[i]) + c * wait_n[i]
        if best_score is None or score > best_score:
            best_score = score
            best_i = i

    chosen_rt = ready_in_wf[best_i]
    runtime = float(dag.nodes[chosen_rt.task_id].get("runtime", 1.0))

    finishes: List[float] = []
    energies: List[float] = []
    greens: List[float] = []
    names: List[str] = []

    for m in machines:
        start = max(float(now), float(machine_available_time[m.name]), float(chosen_rt.ready_time))
        exec_t = runtime / max(1e-9, m.speed)
        finish = start + exec_t

        E = float(m.power.energy(m.f, exec_t, exec_t))
        gbar = float(m.avg_green_fraction(start, finish))
        gE = gbar * E

        names.append(m.name)
        finishes.append(finish)
        energies.append(E)
        greens.append(gE)

    f_n = _minmax_norm(finishes)
    e_n = _minmax_norm(energies)
    g_n = _minmax_norm(greens)

    best_m = 0
    best_cost = None
    for i in range(len(names)):
        cost = d * f_n[i] + e * e_n[i] + f * (1.0 - g_n[i])
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_m = i

    mname = names[best_m]
    finish = finishes[best_m]
    E = energies[best_m]
    gE = greens[best_m]
    neg_g = -gE
    return chosen_rt, mname, finish, E, gE, neg_g


def _nondominated_mask(F: np.ndarray) -> np.ndarray:
    """
    Minimization-space nondominance mask.
    """
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                keep[i] = False
                break
    return keep


# -----------------------------
# NSGA-II baseline (online, per decision point)
# -----------------------------

class NSGA2(SchedulerBase):
    """
    Online NSGA-II baseline (small-budget optimization per decision point).
    It outputs ONE (task, machine) decision consistent with your simulator.

    Objectives (minimize):
      1) finish time (makespan proxy)
      2) energy
      3) -green_energy
    """
    name = "NSGA-II"
    scheduler_family = "mo_baseline"
    variant = "full"

    def __init__(
        self,
        pop_size: int = 40,
        ngen: int = 15,
        cxpb: float = 0.7,
        mutpb: float = 0.25,
        seed: int = 0,
    ):
        self.pop_size = int(pop_size)
        self.ngen = int(ngen)
        self.cxpb = float(cxpb)
        self.mutpb = float(mutpb)
        self.rng = random.Random(int(seed))

        self.rank_u_cache: Dict[str, Dict[str, float]] = {}
        self.scheduler_calls = 0
        self.recluster_count = 0

        if not hasattr(creator, "FitnessNSGA2Online"):
            creator.create("FitnessNSGA2Online", deap_base.Fitness, weights=(-1.0, -1.0, -1.0))
        if not hasattr(creator, "IndividualNSGA2Online"):
            creator.create("IndividualNSGA2Online", list, fitness=creator.FitnessNSGA2Online)

    def on_new_workflow(self, wf_id: str, dag: nx.DiGraph) -> None:
        self.rank_u_cache.pop(wf_id, None)

    def _make_toolbox(self, eval_fn):
        tb = deap_base.Toolbox()

        def create_individual():
            return [self.rng.random() for _ in range(6)]

        tb.register("individual", tools.initIterate, creator.IndividualNSGA2Online, create_individual)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("clone", copy.deepcopy)
        tb.register("mate", tools.cxTwoPoint)
        tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.25, indpb=0.3)
        tb.register("select", tools.selNSGA2)
        tb.register("evaluate", eval_fn)
        return tb

    @staticmethod
    def _choose_from_front(front: List, w=(1.0, 1.0, 1.0)) -> List[float] | None:
        if not front:
            return None
        F = np.array([ind.fitness.values for ind in front], dtype=float)
        Fn = np.zeros_like(F)
        for j in range(F.shape[1]):
            Fn[:, j] = np.array(_minmax_norm(F[:, j].tolist()), dtype=float)
        ww = np.array(w, dtype=float).reshape(1, -1)
        score = (Fn * ww).sum(axis=1)
        idx = int(np.argmin(score))
        return list(front[idx])

    def select(self, now, ready_tasks, dags, machines, machine_available_time, ctx=None):
        self.scheduler_calls += 1
        if not ready_tasks:
            return None

        wf_id = _pick_wf_id(ready_tasks)
        dag = dags[wf_id]
        ready_in_wf = [rt for rt in ready_tasks if rt.wf_id == wf_id]
        if not ready_in_wf:
            return None

        if wf_id not in self.rank_u_cache or not self.rank_u_cache[wf_id]:
            self.rank_u_cache[wf_id] = _compute_rank_u(dag, machines)
        rank_u = self.rank_u_cache[wf_id]

        def eval_ind(individual):
            _rt, _mname, finish, E, gE, neg_g = _decode_action(
                individual=individual,
                now=float(now),
                ready_in_wf=ready_in_wf,
                dag=dag,
                machines=machines,
                machine_available_time=machine_available_time,
                rank_u=rank_u,
            )
            return float(finish), float(E), float(neg_g)

        tb = self._make_toolbox(eval_ind)
        pop = tb.population(n=self.pop_size)

        invalid = [ind for ind in pop if not ind.fitness.valid]
        fits = list(map(tb.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit

        pop = tb.select(pop, len(pop))

        for _ in range(self.ngen):
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [tb.clone(ind) for ind in offspring]

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if self.rng.random() < self.cxpb:
                    tb.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values

            for mut in offspring:
                if self.rng.random() < self.mutpb:
                    tb.mutate(mut)
                    del mut.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(tb.evaluate, invalid))
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            pop = tb.select(pop + offspring, self.pop_size)

        fronts = tools.sortNondominated(pop, k=len(pop), first_front_only=True)
        front0 = fronts[0] if fronts else pop

        chosen = self._choose_from_front(front0, w=(1.0, 1.0, 1.0))
        if chosen is None:
            return None

        chosen_rt, mname, *_ = _decode_action(
            individual=chosen,
            now=float(now),
            ready_in_wf=ready_in_wf,
            dag=dag,
            machines=machines,
            machine_available_time=machine_available_time,
            rank_u=rank_u,
        )
        return chosen_rt, mname


# -----------------------------
# Single-objective / anchor baselines
# -----------------------------

class FCFS(SchedulerBase):
    name = "FCFS"
    scheduler_family = "anchor"
    variant = "full"

    def __init__(self):
        self.scheduler_calls = 0
        self.recluster_count = 0

    def select(self, now, ready_tasks, dags, machines, machine_available_time, ctx=None):
        self.scheduler_calls += 1
        if not ready_tasks:
            return None
        rt = sorted(ready_tasks, key=lambda x: (x.ready_time, x.wf_id, x.task_id))[0]
        m = min(machines, key=lambda mm: machine_available_time[mm.name])
        return rt, m.name


class LIST(SchedulerBase):
    name = "LIST"
    scheduler_family = "anchor"
    variant = "full"

    def __init__(self):
        self.scheduler_calls = 0
        self.recluster_count = 0

    def select(self, now, ready_tasks, dags, machines, machine_available_time, ctx=None):
        self.scheduler_calls += 1
        if not ready_tasks:
            return None

        def runtime(rt: ReadyTask) -> float:
            return float(dags[rt.wf_id].nodes[rt.task_id].get("runtime", 1.0))

        rt = sorted(ready_tasks, key=lambda x: (x.ready_time, runtime(x)))[0]
        m = min(machines, key=lambda mm: machine_available_time[mm.name])
        return rt, m.name


class HEFT_Simple(SchedulerBase):
    name = "HEFT"
    scheduler_family = "anchor"
    variant = "full"

    def __init__(self):
        self.rank_u: Dict[str, Dict[str, float]] = {}
        self.scheduler_calls = 0
        self.recluster_count = 0

    def on_new_workflow(self, wf_id: str, dag: nx.DiGraph) -> None:
        self.rank_u[wf_id] = {}

    def select(self, now, ready_tasks, dags, machines, machine_available_time, ctx=None):
        self.scheduler_calls += 1
        if not ready_tasks:
            return None

        wf_id = _pick_wf_id(ready_tasks)
        dag = dags[wf_id]
        ready_in_wf = [rt for rt in ready_tasks if rt.wf_id == wf_id]
        if not ready_in_wf:
            return None

        if not self.rank_u.get(wf_id):
            self.rank_u[wf_id] = _compute_rank_u(dag, machines)

        def key(rt: ReadyTask):
            return (-self.rank_u[rt.wf_id][rt.task_id], rt.ready_time)

        rt = sorted(ready_in_wf, key=key)[0]

        runtime = float(dag.nodes[rt.task_id].get("runtime", 1.0))
        best = None
        for m in machines:
            start = max(float(now), float(machine_available_time[m.name]), float(rt.ready_time))
            exec_t = runtime / max(1e-9, m.speed)
            finish = start + exec_t
            if best is None or finish < best[0]:
                best = (finish, m.name)
        return rt, best[1]


class GREENHEFT(SchedulerBase):
    """
    A green-aware HEFT-style single-point anchor:
    - task choice: HEFT upward rank
    - machine choice: normalized weighted sum of finish / energy / green
    """
    name = "GREENHEFT"
    scheduler_family = "green_anchor"
    variant = "full"

    def __init__(self, wf: float = 0.5, we: float = 0.2, wg: float = 0.3):
        self.wf = float(wf)
        self.we = float(we)
        self.wg = float(wg)
        self.rank_u: Dict[str, Dict[str, float]] = {}
        self.scheduler_calls = 0
        self.recluster_count = 0

    def on_new_workflow(self, wf_id: str, dag: nx.DiGraph) -> None:
        self.rank_u[wf_id] = {}

    def select(self, now, ready_tasks, dags, machines, machine_available_time, ctx=None):
        self.scheduler_calls += 1
        if not ready_tasks:
            return None

        wf_id = _pick_wf_id(ready_tasks)
        dag = dags[wf_id]
        ready_in_wf = [rt for rt in ready_tasks if rt.wf_id == wf_id]
        if not ready_in_wf:
            return None

        if not self.rank_u.get(wf_id):
            self.rank_u[wf_id] = _compute_rank_u(dag, machines)

        rt = sorted(
            ready_in_wf,
            key=lambda x: (-self.rank_u[x.wf_id][x.task_id], x.ready_time, x.task_id),
        )[0]

        finishes = []
        energies = []
        greens = []
        names = []

        for m in machines:
            finish, E, gE, _bE, _rt = _estimate_action(
                float(now), rt, dag, m, machine_available_time
            )
            finishes.append(finish)
            energies.append(E)
            greens.append(gE)
            names.append(m.name)

        f_n = _minmax_norm(finishes)
        e_n = _minmax_norm(energies)
        g_n = _minmax_norm(greens)

        best_i = 0
        best_score = None
        for i in range(len(names)):
            score = self.wf * f_n[i] + self.we * e_n[i] + self.wg * (1.0 - g_n[i])
            if best_score is None or score < best_score:
                best_score = score
                best_i = i

        return rt, names[best_i]


class MOHEFT(SchedulerBase):
    """
    Budgeted MOHEFT-style online comparator.

    This is a practical per-decision Pareto baseline:
    - enumerate candidate (task, machine) actions in a priority order
    - stop early under decision_budget_ms or candidate_cap
    - extract nondominated front in the 4D objective space:
        finish, energy, -green_energy, brown_energy
    - truncate front by frontier_cap
    - pick one action by normalized weighted sum

    It is intentionally budget-sensitive, so low / medium / high
    should produce different runtime/quality trade-offs.
    """
    name = "MOHEFT"
    scheduler_family = "mo_baseline"
    variant = "full"

    def __init__(
        self,
        w1: float = 0.5,
        w2: float = 0.3,
        w3: float = 0.2,
        w4: float = 0.2,
        decision_budget_ms: float = 0.0,
        candidate_cap: int = 0,
        frontier_cap: int = 0,
        seed: int = 0,
    ):
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.w4 = float(w4)

        self.decision_budget_ms = float(decision_budget_ms)
        self.candidate_cap = int(candidate_cap)
        self.frontier_cap = int(frontier_cap)

        self.rng = random.Random(int(seed))
        self.rank_u: Dict[str, Dict[str, float]] = {}

        self.scheduler_calls = 0
        self.recluster_count = 0

    def on_new_workflow(self, wf_id: str, dag: nx.DiGraph) -> None:
        self.rank_u[wf_id] = {}

    def _weighted_machine_order(
        self,
        now: float,
        rt: ReadyTask,
        dag: nx.DiGraph,
        machines: List[Machine],
        machine_available_time: Dict[str, float],
    ) -> List[Tuple[Machine, float, float, float, float]]:
        tmp = []
        for m in machines:
            finish, E, gE, bE, _runtime = _estimate_action(
                now, rt, dag, m, machine_available_time
            )
            tmp.append((m, finish, E, gE, bE))

        finishes = [x[1] for x in tmp]
        energies = [x[2] for x in tmp]
        greens = [x[3] for x in tmp]
        browns = [x[4] for x in tmp]

        f_n = _minmax_norm(finishes)
        e_n = _minmax_norm(energies)
        b_n = _minmax_norm(browns)
        g_n = _minmax_norm(greens)

        scored = []
        for i, item in enumerate(tmp):
            score = (
                self.w1 * f_n[i]
                + self.w2 * e_n[i]
                + self.w3 * (1.0 - g_n[i])
                + self.w4 * b_n[i]
            )
            scored.append((score, item))

        scored.sort(key=lambda x: x[0])
        return [it for _, it in scored]

    def select(self, now, ready_tasks, dags, machines, machine_available_time, ctx=None):
        self.scheduler_calls += 1
        if not ready_tasks:
            return None

        wf_id = _pick_wf_id(ready_tasks)
        dag = dags[wf_id]
        ready_in_wf = [rt for rt in ready_tasks if rt.wf_id == wf_id]
        if not ready_in_wf:
            return None

        if not self.rank_u.get(wf_id):
            self.rank_u[wf_id] = _compute_rank_u(dag, machines)

        rank_u = self.rank_u[wf_id]
        ready_sorted = sorted(
            ready_in_wf,
            key=lambda x: (-rank_u.get(x.task_id, 0.0), x.ready_time, x.task_id),
        )

        t0 = time.perf_counter()
        candidates = []

        hard_cap = self.candidate_cap if self.candidate_cap > 0 else len(ready_sorted) * len(machines)

        stop = False
        for rt in ready_sorted:
            ordered = self._weighted_machine_order(
                float(now), rt, dag, machines, machine_available_time
            )

            for m, finish, E, gE, bE in ordered:
                candidates.append(
                    {
                        "rt": rt,
                        "mname": m.name,
                        "finish": float(finish),
                        "energy": float(E),
                        "green_energy": float(gE),
                        "neg_green": float(-gE),
                        "brown": float(bE),
                    }
                )

                if self.candidate_cap > 0 and len(candidates) >= hard_cap:
                    stop = True
                    break

                if self.decision_budget_ms > 0:
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    if elapsed_ms >= self.decision_budget_ms:
                        stop = True
                        break

            if stop:
                break

        if not candidates:
            # last-resort fallback
            rt = ready_sorted[0]
            m = min(machines, key=lambda mm: machine_available_time[mm.name])
            return rt, m.name

        F = np.array(
            [
                [c["finish"], c["energy"], c["neg_green"], c["brown"]]
                for c in candidates
            ],
            dtype=float,
        )
        keep = _nondominated_mask(F)
        front = [candidates[i] for i in range(len(candidates)) if keep[i]]

        if not front:
            front = candidates[:]

        # frontier truncation
        if self.frontier_cap > 0 and len(front) > self.frontier_cap:
            ff = np.array(
                [[c["finish"], c["energy"], c["brown"], -c["neg_green"]] for c in front],
                dtype=float,
            )
            ffn = np.zeros_like(ff)
            for j in range(ff.shape[1]):
                ffn[:, j] = np.array(_minmax_norm(ff[:, j].tolist()), dtype=float)

            w = np.array([self.w1, self.w2, self.w4, self.w3], dtype=float).reshape(1, -1)
            score = (ffn * w).sum(axis=1)
            idx = np.argsort(score)[: self.frontier_cap]
            front = [front[int(i)] for i in idx.tolist()]

        # final pick from front
        ff = np.array(
            [[c["finish"], c["energy"], c["brown"], -c["neg_green"]] for c in front],
            dtype=float,
        )
        ffn = np.zeros_like(ff)
        for j in range(ff.shape[1]):
            ffn[:, j] = np.array(_minmax_norm(ff[:, j].tolist()), dtype=float)

        w = np.array([self.w1, self.w2, self.w4, self.w3], dtype=float).reshape(1, -1)
        score = (ffn * w).sum(axis=1)
        best_idx = int(np.argmin(score))
        chosen = front[best_idx]

        return chosen["rt"], chosen["mname"]