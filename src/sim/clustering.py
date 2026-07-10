from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN


@dataclass
class ClusterResult:
    task_to_cluster: Dict[str, int]
    clusters: Dict[int, List[str]]


def build_simple_features(dag: nx.DiGraph) -> np.ndarray:
    """
    Feature per task: [runtime, indegree, outdegree, depth]
    """
    topo = list(nx.topological_sort(dag))
    depth = {n: 0 for n in topo}
    for n in topo:
        preds = list(dag.predecessors(n))
        if preds:
            depth[n] = 1 + max(depth[p] for p in preds)

    nodes = list(dag.nodes)
    X = []
    for n in nodes:
        rt = float(dag.nodes[n].get("runtime", 1.0))
        X.append([rt, dag.in_degree(n), dag.out_degree(n), depth.get(n, 0)])
    return np.array(X, dtype=float)


def cluster_tasks(
    dag: nx.DiGraph,
    method: str = "auto",
    k: int = 8,
    eps: float = 0.8,
    min_samples: int = 3,
) -> ClusterResult:
    nodes = list(dag.nodes)
    n = len(nodes)
    if n == 0:
        return ClusterResult({}, {})

    X = build_simple_features(dag)
    method = str(method).strip().lower()

    # 不在 clustering.py 内做 auto 决策；auto 必须在上层（ECDS）被解析成具体方法
    if method == "auto":
        raise ValueError("clustering.cluster_tasks: method='auto' must be resolved by the caller (e.g., ECDS).")

    if method == "none":
        labels = np.arange(n)

    elif method == "kmeans":
        kk = min(max(2, int(k)), n)
        km = KMeans(n_clusters=kk, random_state=0, n_init="auto")
        labels = km.fit_predict(X)

    elif method == "dbscan":
        db = DBSCAN(eps=float(eps), min_samples=int(min_samples))
        labels = db.fit_predict(X)

        # noise = -1 -> each noise becomes its own cluster
        next_id = int(labels.max()) + 1
        for i in range(n):
            if int(labels[i]) == -1:
                labels[i] = next_id
                next_id += 1

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    task_to_cluster = {nodes[i]: int(labels[i]) for i in range(n)}
    clusters: Dict[int, List[str]] = {}
    for t, c in task_to_cluster.items():
        clusters.setdefault(c, []).append(t)

    return ClusterResult(task_to_cluster=task_to_cluster, clusters=clusters)