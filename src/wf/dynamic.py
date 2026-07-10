from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import random
import networkx as nx


@dataclass
class WorkflowArrival:
    wf_id: str
    dag: nx.DiGraph
    arrival_time: float


def assign_task_release_times(
    dag: nx.DiGraph,
    enabled: bool,
    T: int,
    seed: int = 0
) -> None:
    """
    Optional: assign a release_time to each task to simulate task-level dynamic visibility.
    """
    if not enabled:
        for n in dag.nodes:
            dag.nodes[n]["release_time"] = 0.0
        return

    rng = random.Random(seed)
    for n in dag.nodes:
        dag.nodes[n]["release_time"] = float(rng.randint(0, T))


def compute_dyn_delta(prev_dag: nx.DiGraph, new_dag: nx.DiGraph) -> float:
    """
    A simple dyn(G) proxy: weighted change of nodes+edges.
    Your report mentions node add/remove rate, edge modification, priority change, etc.
    Here we implement a minimal measurable version for triggering.
    """
    prev_nodes, new_nodes = set(prev_dag.nodes), set(new_dag.nodes)
    prev_edges, new_edges = set(prev_dag.edges), set(new_dag.edges)

    node_change = len(prev_nodes.symmetric_difference(new_nodes)) / max(1, len(prev_nodes))
    edge_change = len(prev_edges.symmetric_difference(new_edges)) / max(1, len(prev_edges))
    return 0.5 * node_change + 0.5 * edge_change


def split_into_subdags_by_level(dag: nx.DiGraph, num_batches: int) -> List[List[str]]:
    """
    Simple sub-DAG batching: group nodes by topological depth into num_batches batches.
    (This approximates "sub-DAG batch arrival".)
    """
    topo = list(nx.topological_sort(dag))
    depth: Dict[str, int] = {n: 0 for n in topo}
    for n in topo:
        preds = list(dag.predecessors(n))
        if preds:
            depth[n] = 1 + max(depth[p] for p in preds)
    max_depth = max(depth.values()) if depth else 0
    if max_depth == 0:
        return [topo]

    batches: List[List[str]] = [[] for _ in range(num_batches)]
    for n in topo:
        idx = int(depth[n] / (max_depth + 1) * num_batches)
        idx = min(num_batches - 1, max(0, idx))
        batches[idx].append(n)
    return [b for b in batches if b]
