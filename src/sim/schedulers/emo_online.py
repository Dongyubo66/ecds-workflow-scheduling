from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import copy
import random
import numpy as np
import networkx as nx

from deap import base as deap_base, creator, tools

from .base import SchedulerBase, ReadyTask
from ..resources import Machine
from ..pareto import nondominated_indices, crowding_distance
from ..simulator import SimContext


# -------------------------
# Shared policy decoding
# -------------------------

@dataclass(frozen=True)
class PolicyParam:
    # task weights
    w_rank: float
    w_runtime: float
    w_depth: float
    w_outdeg: float
    # machine weights (all minimization)
    w_finish: float
    w_energy: float
    w_brown: float
    w_util: float

    @staticmethod
    def from_vec(x: List[float]) -> "PolicyParam":
        vals = [float(v) for v in x]
        while len(vals) < 8:
            vals.append(0.0)
        return PolicyParam(*vals[:8])


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    mn = arr.min()
    mx = arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _compute_depth(dag: nx.DiGraph) -> Dict[str, int]:
    topo = list(nx.topological_sort(dag))
    depth = {n: 0 for n in topo}
    for n in topo:
        preds = list(dag.predecessors(n))
        if preds:
            depth[n] = 1 + max(depth[p] for p in preds)
    return depth


def _compute_rank_u(dag: nx.DiGraph, machines: List[Machine]) -> Dict[str, float]:
    avg_speed = sum(m.speed for m in machines) / max(1e-9, len(machines))
    comp = {n: float(dag.nodes[n].get("runtime", 1.0)) / avg_speed for n in dag.nodes}
    rank_u = {n: 0.0 for n in dag.nodes}
    topo_rev = list(nx.topological_sort(dag))[::-1]
    for n in topo_rev:
        succs = list(dag.successors(n))
        if not succs:
            rank_u[n] = comp[n]
        else:
            rank_u[n] = comp[n] + max(rank_u[s] for s in succs)
    return rank_u


def _util_penalty(now: float, m: Machine, machine_avail: Dict[str, float], window: float) -> float:
    window = max(1e-9, float(window))
    busy = max(0.0, min(window, float(machine_avail[m.name]) - float(now)))
    return busy / window


def decode_action(now: float, ready_tasks: List[ReadyTask], dags: Dict[str, nx.DiGraph],
                  machines: List[Machine], machine_avail: Dict[str, float],
                  pp: PolicyParam,
                  rank_u_cache: Dict[str, Dict[str, float]],
                  depth_cache: Dict[str, Dict[str, int]],
                  util_window: float = 5.0) -> Optional[Tuple[ReadyTask, str]]:
    if not ready_tasks:
        return None

    wf_id = min(ready_tasks, key=lambda rt: (rt.ready_time, rt.wf_id)).wf_id
    dag = dags[wf_id]
    ready_in_wf = [rt for rt in ready_tasks if rt.wf_id == wf_id]
    if not ready_in_wf:
        return None

    if wf_id not in rank_u_cache or not rank_u_cache[wf_id]:
        rank_u_cache[wf_id] = _compute_rank_u(dag, machines)
    if wf_id not in depth_cache or not depth_cache[wf_id]:
        depth_cache[wf_id] = _compute_depth(dag)

    rank_u = rank_u_cache[wf_id]
    depth = depth_cache[wf_id]

    # ---- task scoring ----
    ranks = np.array([rank_u.get(rt.task_id, 0.0) for rt in ready_in_wf], dtype=float)
    rts = np.array([float(dag.nodes[rt.task_id].get("runtime", 1.0)) for rt in ready_in_wf], dtype=float)
    deps = np.array([float(depth.get(rt.task_id, 0)) for rt in ready_in_wf], dtype=float)
    outs = np.array([float(dag.out_degree(rt.task_id)) for rt in ready_in_wf], dtype=float)

    ranks_n = _minmax_norm(ranks)
    rts_n = _minmax_norm(rts)
    deps_n = _minmax_norm(deps)
    outs_n = _minmax_norm(outs)

    # larger is better
    task_score = (
        pp.w_rank * ranks_n
        - pp.w_runtime * rts_n
        + pp.w_depth * deps_n
        + pp.w_outdeg * outs_n
    )
    idx = int(np.argmax(task_score))
    chosen = ready_in_wf[idx]

    # ---- machine scoring (all minimization) ----
    runtime = float(dag.nodes[chosen.task_id].get("runtime", 1.0))

    finishes = []
    energies = []
    browns = []
    utils = []
    names = []

    for m in machines:
        start = max(float(now), float(machine_avail[m.name]), float(chosen.ready_time))
        exec_t = runtime / max(1e-9, m.speed)
        finish = start + exec_t
        e = m.power.energy(m.f, exec_t, exec_t)
        gbar = m.avg_green_fraction(start, finish)
        brown = (1.0 - gbar) * e
        u = _util_penalty(now, m, machine_avail, util_window)

        names.append(m.name)
        finishes.append(finish)
        energies.append(e)
        browns.append(brown)
        utils.append(u)

    f_n = _minmax_norm(np.array(finishes))
    e_n = _minmax_norm(np.array(energies))
    b_n = _minmax_norm(np.array(browns))
    u_n = _minmax_norm(np.array(utils))

    J = (
        pp.w_finish * f_n
        + pp.w_energy * e_n
        + pp.w_brown * b_n
        + pp.w_util * u_n
    )

    best = int(np.argmin(J))
    return chosen, names[best]


# -------------------------
# Horizon evaluator
# -------------------------

@dataclass(frozen=True)
class RHCConfig:
    horizon_steps: int = 20
    reopt_every: int = 10
    util_window: float = 5.0


def rollout_eval(ctx: SimContext, wf_id: str, pp: PolicyParam, rhc: RHCConfig,
                 rank_u_cache: Dict[str, Dict[str, float]],
                 depth_cache: Dict[str, Dict[str, int]]) -> Tuple[float, float, float, float]:
    """
    Return objective vector (all minimized):
      f1 makespan_est
      f2 energy_est
      f3 brown_est
      f4 unavail_est  (1 - utilization proxied in [0,1])
    """
    dag = ctx.dags[wf_id]
    st_map = ctx.state[wf_id]

    # clone status
    started = {t: st_map[t].started for t in dag.nodes}
    finished = {t: st_map[t].finished for t in dag.nodes}
    finish_time = {}
    for t in dag.nodes:
        if st_map[t].started or st_map[t].finished:
            finish_time[t] = float(st_map[t].finish_time)

    machine_avail = copy.deepcopy(ctx.machine_avail)

    now = float(ctx.now)
    energy_sum = 0.0
    brown_sum = 0.0
    busy_sum = 0.0
    window_end = now + float(rhc.util_window)

    def deps_ok(t: str) -> bool:
        for p in dag.predecessors(t):
            if p not in finish_time:
                return False
        return True

    def ready_time(t: str) -> float:
        rel = float(dag.nodes[t].get("release_time", 0.0))
        parent_finish = 0.0
        for p in dag.predecessors(t):
            parent_finish = max(parent_finish, float(finish_time.get(p, 0.0)))
        return max(float(ctx.arrivals[wf_id]), rel, parent_finish)

    def collect_ready_local(tnow: float) -> List[ReadyTask]:
        rts = []
        for t in dag.nodes:
            if started.get(t, False) or finished.get(t, False):
                continue
            if not deps_ok(t):
                continue
            rt = ready_time(t)
            if rt <= tnow + 1e-9:
                rts.append(ReadyTask(task_id=t, wf_id=wf_id, ready_time=rt))
        return rts

    def next_ready_time(tnow: float) -> Optional[float]:
        nxt = None
        for t in dag.nodes:
            if started.get(t, False) or finished.get(t, False):
                continue
            if not deps_ok(t):
                continue
            rt = ready_time(t)
            if rt > tnow + 1e-9:
                if nxt is None or rt < nxt:
                    nxt = rt
        return nxt

    for _ in range(int(rhc.horizon_steps)):
        ready = collect_ready_local(now)
        if not ready:
            nt = next_ready_time(now)
            if nt is None:
                break
            now = float(nt)
            continue

        decision = decode_action(now, ready, {wf_id: dag}, ctx.machines, machine_avail,
                                pp, rank_u_cache, depth_cache, util_window=rhc.util_window)
        if decision is None:
            break
        rt, mname = decision
        m = next(mm for mm in ctx.machines if mm.name == mname)

        runtime = float(dag.nodes[rt.task_id].get("runtime", 1.0))
        start = max(now, float(machine_avail[m.name]), float(rt.ready_time))
        exec_t = runtime / max(1e-9, m.speed)
        finish = start + exec_t

        # account energy
        e = m.power.energy(m.f, exec_t, exec_t)
        gbar = m.avg_green_fraction(start, finish)
        brown = (1.0 - gbar) * e

        # utilization proxy in a short fixed window
        overlap_s = max(now, start)
        overlap_e = min(window_end, finish)
        if overlap_e > overlap_s:
            busy_sum += (overlap_e - overlap_s)

        energy_sum += e
        brown_sum += brown

        started[rt.task_id] = True
        finished[rt.task_id] = True
        finish_time[rt.task_id] = finish
        machine_avail[m.name] = finish

    makespan_est = max(machine_avail.values()) if machine_avail else float(ctx.now)
    unavail_est = min(1.0, max(0.0, busy_sum / max(1e-9, float(rhc.util_window) * max(1, len(ctx.machines)))))

    return makespan_est, energy_sum, brown_sum, unavail_est


# -------------------------
# NSGA-II RHC Scheduler
# -------------------------

@dataclass(frozen=True)
class NSGA2RHCConfig:
    pop_size: int = 40
    n_gen: int = 15
    cxpb: float = 0.9
    mutpb: float = 0.2
    eta_cx: float = 15.0
    eta_mut: float = 20.0
    n_var: int = 8
    rhc: RHCConfig = RHCConfig()
    # preference weights to pick one point from Pareto set
    pref: Tuple[float, float, float, float] = (0.5, 0.3, 0.2, 0.5)  # (makespan, energy, brown, unavail)


class NSGA2_RHC(SchedulerBase):
    name = "NSGA2-RHC"

    def __init__(self, cfg: NSGA2RHCConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)
        self.rank_u_cache: Dict[str, Dict[str, float]] = {}
        self.depth_cache: Dict[str, Dict[str, int]] = {}
        self._theta_cache: Dict[str, PolicyParam] = {}
        self._since_reopt: Dict[str, int] = {}

        # DEAP creator re-entry protection (names unique per class)
        fit_name = "FitnessNSGA2RHC"
        ind_name = "IndividualNSGA2RHC"
        if not hasattr(creator, fit_name):
            creator.create(fit_name, deap_base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
        if not hasattr(creator, ind_name):
            creator.create(ind_name, list, fitness=getattr(creator, fit_name))

        self.toolbox = deap_base.Toolbox()
        self.toolbox.register("attr_float", random.random)
        self.toolbox.register("individual", tools.initRepeat, getattr(creator, ind_name), self.toolbox.attr_float, n=self.cfg.n_var)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # SBX + polynomial mutation (bounded [0,1])
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=float(self.cfg.eta_cx))
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=float(self.cfg.eta_mut), indpb=1.0 / float(self.cfg.n_var))
        self.toolbox.register("select", tools.selNSGA2)

    def on_new_workflow(self, wf_id: str, dag: nx.DiGraph) -> None:
        self._theta_cache.pop(wf_id, None)
        self._since_reopt[wf_id] = 10**9
        self.rank_u_cache.pop(wf_id, None)
        self.depth_cache.pop(wf_id, None)

    def _pick_workflow(self, ready_tasks: List[ReadyTask]) -> str:
        return min(ready_tasks, key=lambda rt: (rt.ready_time, rt.wf_id)).wf_id

    def _reopt(self, ctx: SimContext, wf_id: str) -> PolicyParam:
        random.seed(self.seed + hash((wf_id, int(ctx.now))) % 10**6)
        np.random.seed(self.seed + hash((wf_id, int(ctx.now))) % 10**6)

        def evaluate(ind):
            pp = PolicyParam.from_vec(ind)
            return rollout_eval(ctx, wf_id, pp, self.cfg.rhc, self.rank_u_cache, self.depth_cache)

        self.toolbox.register("evaluate", evaluate)

        pop = self.toolbox.population(n=int(self.cfg.pop_size))
        # evaluate
        for ind in pop:
            ind.fitness.values = self.toolbox.evaluate(ind)
        pop = self.toolbox.select(pop, len(pop))

        for _gen in range(int(self.cfg.n_gen)):
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= float(self.cfg.cxpb):
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            for mut in offspring:
                if random.random() <= float(self.cfg.mutpb):
                    self.toolbox.mutate(mut)
                    del mut.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)

            pop = self.toolbox.select(pop + offspring, int(self.cfg.pop_size))

        # pick nondominated front
        F = np.array([ind.fitness.values for ind in pop], dtype=float)
        nd_idx = nondominated_indices(F)
        front = [pop[i] for i in nd_idx]
        Fnd = np.array([ind.fitness.values for ind in front], dtype=float)

        # preference scalarization on normalized objectives
        mn = Fnd.min(axis=0)
        mx = Fnd.max(axis=0)
        denom = np.maximum(1e-12, mx - mn)
        Fn = (Fnd - mn) / denom

        w = np.array(self.cfg.pref, dtype=float)
        w = w / max(1e-12, w.sum())

        best = 0
        best_val = None
        for i in range(Fn.shape[0]):
            val = float(np.dot(w, Fn[i]))
            if best_val is None or val < best_val:
                best_val = val
                best = i

        return PolicyParam.from_vec(front[best])

    def select(self, now, ready_tasks, dags, machines, machine_available_time, ctx: Optional[SimContext] = None):
        if not ready_tasks:
            return None
        wf_id = self._pick_workflow(ready_tasks)

        if ctx is None:
            # fallback: behave like a simple param policy with fixed theta
            pp = self._theta_cache.get(wf_id, PolicyParam(1,1,1,1,1,1,1,1))
            return decode_action(now, ready_tasks, dags, machines, machine_available_time, pp,
                                 self.rank_u_cache, self.depth_cache, util_window=self.cfg.rhc.util_window)

        since = self._since_reopt.get(wf_id, 10**9)
        if (wf_id not in self._theta_cache) or (since >= int(self.cfg.rhc.reopt_every)):
            self._theta_cache[wf_id] = self._reopt(ctx, wf_id)
            self._since_reopt[wf_id] = 0
        else:
            self._since_reopt[wf_id] = since + 1

        pp = self._theta_cache[wf_id]
        return decode_action(now, ready_tasks, dags, machines, machine_available_time, pp,
                             self.rank_u_cache, self.depth_cache, util_window=self.cfg.rhc.util_window)


# -------------------------
# MOEA/D RHC Scheduler (lightweight)
# -------------------------

@dataclass(frozen=True)
class MOEADRHCConfig:
    pop_size: int = 40
    n_gen: int = 20
    n_var: int = 8
    n_neighbors: int = 10
    prob_neighbor_mating: float = 0.7
    F: float = 0.5                  # differential evolution factor
    mut_sigma: float = 0.1
    rhc: RHCConfig = RHCConfig()
    pref: Tuple[float, float, float, float] = (0.5, 0.3, 0.2, 0.5)


def _dirichlet_weights(n: int, m: int, rng: np.random.RandomState) -> np.ndarray:
    W = rng.dirichlet(alpha=np.ones(m), size=n)
    return W


def _tchebycheff(F: np.ndarray, w: np.ndarray, z: np.ndarray) -> float:
    return float(np.max(w * np.abs(F - z)))


class MOEAD_RHC(SchedulerBase):
    name = "MOEAD-RHC"

    def __init__(self, cfg: MOEADRHCConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)
        self.rank_u_cache: Dict[str, Dict[str, float]] = {}
        self.depth_cache: Dict[str, Dict[str, int]] = {}
        self._theta_cache: Dict[str, PolicyParam] = {}
        self._since_reopt: Dict[str, int] = {}

    def on_new_workflow(self, wf_id: str, dag: nx.DiGraph) -> None:
        self._theta_cache.pop(wf_id, None)
        self._since_reopt[wf_id] = 10**9
        self.rank_u_cache.pop(wf_id, None)
        self.depth_cache.pop(wf_id, None)

    def _pick_workflow(self, ready_tasks: List[ReadyTask]) -> str:
        return min(ready_tasks, key=lambda rt: (rt.ready_time, rt.wf_id)).wf_id

    def _reopt(self, ctx: SimContext, wf_id: str) -> PolicyParam:
        rng = np.random.RandomState(self.seed + hash((wf_id, int(ctx.now))) % 10**6)

        N = int(self.cfg.pop_size)
        M = 4  # objectives
        W = _dirichlet_weights(N, M, rng)

        # neighbor sets based on weight distance
        dist = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
        B = [list(np.argsort(dist[i])[: int(self.cfg.n_neighbors)]) for i in range(N)]

        # initialize population in [0,1]
        X = rng.rand(N, int(self.cfg.n_var)).astype(float)

        def eval_x(x: np.ndarray) -> np.ndarray:
            pp = PolicyParam.from_vec(list(x))
            f = rollout_eval(ctx, wf_id, pp, self.cfg.rhc, self.rank_u_cache, self.depth_cache)
            return np.array(f, dtype=float)

        Fpop = np.array([eval_x(X[i]) for i in range(N)], dtype=float)
        z = Fpop.min(axis=0)

        for _gen in range(int(self.cfg.n_gen)):
            for i in range(N):
                if rng.rand() < float(self.cfg.prob_neighbor_mating):
                    P = B[i]
                else:
                    P = list(range(N))

                a, b, c = rng.choice(P, size=3, replace=False)
                y = X[a] + float(self.cfg.F) * (X[b] - X[c])
                # gaussian mutation
                y = y + rng.normal(0.0, float(self.cfg.mut_sigma), size=y.shape)
                y = np.clip(y, 0.0, 1.0)

                Fy = eval_x(y)
                z = np.minimum(z, Fy)

                # update neighbors
                for j in B[i]:
                    gj = _tchebycheff(Fpop[j], W[j], z)
                    gy = _tchebycheff(Fy, W[j], z)
                    if gy <= gj:
                        X[j] = y
                        Fpop[j] = Fy

        # extract nondominated from final population
        nd = nondominated_indices(Fpop)
        Xnd = X[nd]
        Fnd = Fpop[nd]

        mn = Fnd.min(axis=0)
        mx = Fnd.max(axis=0)
        denom = np.maximum(1e-12, mx - mn)
        Fn = (Fnd - mn) / denom

        w = np.array(self.cfg.pref, dtype=float)
        w = w / max(1e-12, w.sum())

        best = int(np.argmin(Fn.dot(w)))
        return PolicyParam.from_vec(list(Xnd[best]))

    def select(self, now, ready_tasks, dags, machines, machine_available_time, ctx: Optional[SimContext] = None):
        if not ready_tasks:
            return None

        wf_id = self._pick_workflow(ready_tasks)

        if ctx is None:
            pp = self._theta_cache.get(wf_id, PolicyParam(1,1,1,1,1,1,1,1))
            return decode_action(now, ready_tasks, dags, machines, machine_available_time, pp,
                                 self.rank_u_cache, self.depth_cache, util_window=self.cfg.rhc.util_window)

        since = self._since_reopt.get(wf_id, 10**9)
        if (wf_id not in self._theta_cache) or (since >= int(self.cfg.rhc.reopt_every)):
            self._theta_cache[wf_id] = self._reopt(ctx, wf_id)
            self._since_reopt[wf_id] = 0
        else:
            self._since_reopt[wf_id] = since + 1

        pp = self._theta_cache[wf_id]
        return decode_action(now, ready_tasks, dags, machines, machine_available_time, pp,
                             self.rank_u_cache, self.depth_cache, util_window=self.cfg.rhc.util_window)
