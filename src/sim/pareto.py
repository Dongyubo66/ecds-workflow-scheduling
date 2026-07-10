from __future__ import annotations
from typing import List
import numpy as np

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Minimization dominance: a dominates b if a<=b component-wise and a<b in at least one.
    """
    return np.all(a <= b) and np.any(a < b)


def nondominated_indices(F: np.ndarray) -> List[int]:
    n = F.shape[0]
    keep = []
    for i in range(n):
        dom = False
        for j in range(n):
            if i == j:
                continue
            if dominates(F[j], F[i]):
                dom = True
                break
        if not dom:
            keep.append(i)
    return keep


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """
    NSGA-II crowding distance. Larger means more isolated (better diversity).
    Works for minimization; distance itself is scale-normalized per objective.
    """
    n, m = F.shape
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([np.inf])

    dist = np.zeros(n, dtype=float)
    for k in range(m):
        order = np.argsort(F[:, k])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        fmin = F[order[0], k]
        fmax = F[order[-1], k]
        denom = max(1e-12, fmax - fmin)
        for i in range(1, n - 1):
            if np.isinf(dist[order[i]]):
                continue
            prevv = F[order[i - 1], k]
            nextv = F[order[i + 1], k]
            dist[order[i]] += (nextv - prevv) / denom
    return dist


def truncate_by_pareto_and_crowding(F: np.ndarray, K: int) -> List[int]:
    """
    Return indices of up to K points selected by:
      1) take nondominated first
      2) if >K, keep K with largest crowding
      3) if <K, fill with dominated points sorted by crowding in successive layers (simple)
    """
    n = F.shape[0]
    if n <= K:
        return list(range(n))

    selected = []
    remaining = list(range(n))
    while remaining and len(selected) < K:
        Fr = F[remaining]
        nd_local = nondominated_indices(Fr)
        nd = [remaining[i] for i in nd_local]
        if len(selected) + len(nd) <= K:
            selected.extend(nd)
            remaining = [i for i in remaining if i not in nd]
        else:
            Fnd = F[nd]
            cd = crowding_distance(Fnd)
            order = np.argsort(-cd)  # descending
            take = order[: (K - len(selected))]
            selected.extend([nd[i] for i in take])
            break
    return selected
