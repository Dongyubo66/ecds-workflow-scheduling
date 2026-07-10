from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set

import networkx as nx

from .base import SchedulerBase, ReadyTask
from ..resources import Machine
from ..clustering import cluster_tasks as do_cluster_tasks, ClusterResult


@dataclass
class ECDSConfig:
    # multi-objective weights
    w1: float
    w2: float
    w3: float
    w4: float  # minimize brown energy
    use_brown_objective: bool = True

    # clustering
    clustering_method: str = "auto"   # auto/kmeans/dbscan/none
    k: int = 8
    eps: float = 0.8
    min_samples: int = 3

    # auto safeguard
    auto_large_n_threshold: int = 2000
    auto_large_n_method: str = "kmeans"  # for very large graphs

    # dynamic recluster
    reschedule_enabled: bool = True
    dyn_threshold: float = 0.2
    reschedule_cooldown: float = 30.0  # seconds in simulation time

    # RU proxy window
    util_window: float = 50.0

    # cluster score weights
    ca: float = 0.5
    cb: float = 0.3
    cc: float = 0.2


class ECDS(SchedulerBase):
    name = "ECDS"
    scheduler_family = "ecds"
    variant = "full"
    family = "ecds"   # 保留，兼容旧代码

    def __init__(self, cfg: ECDSConfig):
        self.cfg = cfg

        self.cluster_cache: Dict[str, ClusterResult] = {}
        self.rank_u_cache: Dict[str, Dict[str, float]] = {}
        self.cluster_score_cache: Dict[str, Dict[int, float]] = {}

        self._last_ready_count: Dict[str, int] = {}
        self._need_recluster: Dict[str, bool] = {}
        self._last_recluster_time: Dict[str, float] = {}

        # 改成显式属性，方便 main.py 直接读取
        self.scheduler_calls: int = 0
        self.recluster_count: int = 0

    def get_stats(self) -> Dict[str, float]:
        return {
            "scheduler_calls": float(self.scheduler_calls),
            "recluster_count": float(self.recluster_count),
        }

    def on_new_workflow(self, wf_id: str, dag: nx.DiGraph) -> None:
        self.cluster_cache.pop(wf_id, None)
        self.rank_u_cache.pop(wf_id, None)
        self.cluster_score_cache.pop(wf_id, None)

        self._last_ready_count[wf_id] = 0
        self._need_recluster[wf_id] = True
        self._last_recluster_time[wf_id] = -1e18

    def maybe_reschedule(self, wf_id: str, ready_count: int, now: float = 0.0) -> None:
        if not self.cfg.reschedule_enabled:
            return

        last = self._last_ready_count.get(wf_id, 0)
        denom = max(1, last)
        dyn = abs(ready_count - last) / denom
        last_t = float(self._last_recluster_time.get(wf_id, -1e18))

        if dyn >= float(self.cfg.dyn_threshold) and (float(now) - last_t) >= float(self.cfg.reschedule_cooldown):
            self._need_recluster[wf_id] = True

        self._last_ready_count[wf_id] = ready_count

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

    @staticmethod
    def _minmax_norm_map(raw: Dict[int, float]) -> Dict[int, float]:
        if not raw:
            return {}

        vals = list(raw.values())
        mn, mx = min(vals), max(vals)

        if mx - mn < 1e-12:
            return {k: 0.0 for k in raw.keys()}

        return {k: (v - mn) / (mx - mn) for k, v in raw.items()}

    def _choose_clustering_method(self, dag: nx.DiGraph) -> str:
        method = str(self.cfg.clustering_method).strip().lower()
        n = dag.number_of_nodes()

        if method in ("kmeans", "dbscan", "none"):
            return method

        if method != "auto":
            raise ValueError(f"Unknown clustering method: {method}")

        if n <= 200:
            resolved = "kmeans"
        elif n <= 800:
            resolved = "dbscan"
        else:
            resolved = "kmeans"

        if n >= int(self.cfg.auto_large_n_threshold):
            resolved = str(self.cfg.auto_large_n_method).strip().lower()

        if resolved not in ("kmeans", "dbscan", "none"):
            raise ValueError(f"Auto resolved to invalid method: {resolved}")

        return resolved

    def _ensure_models(self, wf_id: str, dag: nx.DiGraph, machines: List[Machine], now: float) -> None:
        if wf_id not in self.cluster_cache or self._need_recluster.get(wf_id, False):
            method = self._choose_clustering_method(dag)

            self.cluster_cache[wf_id] = do_cluster_tasks(
                dag,
                method=method,
                k=int(self.cfg.k),
                eps=float(self.cfg.eps),
                min_samples=int(self.cfg.min_samples),
            )

            self._need_recluster[wf_id] = False
            self._last_recluster_time[wf_id] = float(now)
            self.cluster_score_cache.pop(wf_id, None)
            self.recluster_count += 1

        if wf_id not in self.rank_u_cache or not self.rank_u_cache[wf_id]:
            self.rank_u_cache[wf_id] = self._compute_rank_u(dag, machines)
            self.cluster_score_cache.pop(wf_id, None)

        if wf_id not in self.cluster_score_cache:
            self.cluster_score_cache[wf_id] = self._compute_cluster_scores(wf_id, dag, machines)

    def _compute_cluster_scores(self, wf_id: str, dag: nx.DiGraph, machines: List[Machine]) -> Dict[int, float]:
        cr = self.cluster_cache[wf_id]
        rank_u = self.rank_u_cache[wf_id]

        task_avg_e: Dict[str, float] = {}
        m_cnt = max(1, len(machines))

        for t in dag.nodes:
            rt = float(dag.nodes[t].get("runtime", 1.0))
            e_sum = 0.0
            for m in machines:
                exec_t = rt / max(1e-9, m.speed)
                e_sum += m.power.energy(m.f, exec_t, exec_t)
            task_avg_e[t] = e_sum / m_cnt

        raw_cp: Dict[int, float] = {}
        raw_w: Dict[int, float] = {}
        raw_e: Dict[int, float] = {}

        for cid, ts in cr.clusters.items():
            if not ts:
                raw_cp[cid] = raw_w[cid] = raw_e[cid] = 0.0
                continue

            raw_cp[cid] = max(rank_u.get(t, 0.0) for t in ts)
            raw_w[cid] = sum(float(dag.nodes[t].get("runtime", 1.0)) for t in ts)
            raw_e[cid] = sum(task_avg_e.get(t, 0.0) for t in ts)

        cp_n = self._minmax_norm_map(raw_cp)
        w_n = self._minmax_norm_map(raw_w)
        e_n = self._minmax_norm_map(raw_e)

        a, b, c = float(self.cfg.ca), float(self.cfg.cb), float(self.cfg.cc)
        score: Dict[int, float] = {}
        for cid in cr.clusters.keys():
            score[cid] = a * cp_n.get(cid, 0.0) + b * w_n.get(cid, 0.0) + c * e_n.get(cid, 0.0)

        return score

    def _util_penalty(self, now: float, m: Machine, machine_available_time: Dict[str, float], exec_t: float) -> float:
        wait = max(0.0, float(machine_available_time[m.name]) - float(now))
        et = max(1e-9, float(exec_t))
        return wait / (wait + et)

    def _pick_workflow(self, ready_tasks: List[ReadyTask]) -> str:
        return min(ready_tasks, key=lambda rt: (rt.ready_time, rt.wf_id)).wf_id

    def _pick_cluster(self, wf_id: str, ready_in_wf: List[ReadyTask]) -> int:
        cr = self.cluster_cache[wf_id]
        score = self.cluster_score_cache[wf_id]
        ready_cids: Set[int] = {int(cr.task_to_cluster.get(rt.task_id, 0)) for rt in ready_in_wf}
        return max(ready_cids, key=lambda cid: score.get(cid, 0.0))

    def _pick_task_in_cluster(self, wf_id: str, ready_in_wf: List[ReadyTask], cid: int) -> ReadyTask:
        cr = self.cluster_cache[wf_id]
        rank_u = self.rank_u_cache[wf_id]
        cluster_set = set(cr.clusters.get(cid, []))

        cand = [rt for rt in ready_in_wf if rt.task_id in cluster_set]
        if not cand:
            return max(ready_in_wf, key=lambda rt: (rank_u.get(rt.task_id, 0.0), -rt.ready_time))

        return max(cand, key=lambda rt: (rank_u.get(rt.task_id, 0.0), -rt.ready_time))

    def _choose_machine(
        self,
        now: float,
        rt: ReadyTask,
        dag: nx.DiGraph,
        machines: List[Machine],
        machine_available_time: Dict[str, float],
    ) -> str:
        runtime = float(dag.nodes[rt.task_id].get("runtime", 1.0))

        finishes: List[float] = []
        energies: List[float] = []
        utilp: List[float] = []
        browns: List[float] = []
        names: List[str] = []

        for m in machines:
            start = max(float(now), float(machine_available_time[m.name]), float(rt.ready_time))
            exec_t = runtime / max(1e-9, m.speed)
            finish = start + exec_t

            e = m.power.energy(m.f, exec_t, exec_t)
            u = self._util_penalty(now, m, machine_available_time, exec_t)
            gbar = m.avg_green_fraction(start, finish)
            brown = (1.0 - gbar) * e

            names.append(m.name)
            finishes.append(finish)
            energies.append(e)
            utilp.append(u)
            browns.append(brown)

        min_f = min(finishes)
        min_e = min(energies)
        min_u = min(utilp)
        min_b = min(browns)

        f_n = [v / max(1e-12, min_f) - 1.0 for v in finishes]
        e_n = [v / max(1e-12, min_e) - 1.0 for v in energies]
        u_n = [v / max(1e-12, min_u) - 1.0 for v in utilp]
        b_n = [v / max(1e-12, min_b) - 1.0 for v in browns]

        best_key = None
        best_name = names[0]

        for i, mname in enumerate(names):
            brown_term = float(self.cfg.w4) * b_n[i] if bool(self.cfg.use_brown_objective) else 0.0
            J = (
                float(self.cfg.w1) * f_n[i]
                + float(self.cfg.w2) * e_n[i]
                + float(self.cfg.w3) * u_n[i]
                + brown_term
            )
            key = (J, finishes[i])
            if best_key is None or key < best_key:
                best_key = key
                best_name = mname

        return best_name

    def select(self, now, ready_tasks, dags, machines, machine_available_time, ctx=None):
        # 用显式属性记数，main.py 才能正确读到
        self.scheduler_calls += 1

        if not ready_tasks:
            return None

        wf_id = self._pick_workflow(ready_tasks)
        dag = dags[wf_id]

        # 如果 simulator / ctx 提供了更准确的 ready_count，则可以用它触发 maybe_reschedule
        ready_in_wf = [rt for rt in ready_tasks if rt.wf_id == wf_id]
        self.maybe_reschedule(wf_id, len(ready_in_wf), float(now))

        self._ensure_models(wf_id, dag, machines, now)

        if not ready_in_wf:
            return None

        cid = self._pick_cluster(wf_id, ready_in_wf)
        chosen_rt = self._pick_task_in_cluster(wf_id, ready_in_wf, cid)
        mname = self._choose_machine(now, chosen_rt, dag, machines, machine_available_time)

        return chosen_rt, mname