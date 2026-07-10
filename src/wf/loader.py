import json
from pathlib import Path
import networkx as nx


def load_wfcommons_instance(json_path: str) -> nx.DiGraph:
    p = Path(json_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    wf = data.get("workflow", {})

    # ========= 统一取 tasks（兼容多 schema） =========
    if "tasks" in wf:
        tasks = wf["tasks"]
    elif "specification" in wf and "tasks" in wf["specification"]:
        tasks = wf["specification"]["tasks"]
    else:
        raise ValueError(f"No tasks found in workflow: {json_path}")

    if not tasks:
        raise ValueError(f"Empty tasks list: {json_path}")

    G = nx.DiGraph()

    # ========= 加节点 =========
    for t in tasks:
        tid = str(t.get("id"))

        # runtime 兼容多个字段名
        runtime = (
            t.get("runtimeInSeconds")
            or t.get("runtime")
            or t.get("runtimeSeconds")
            or 1.0
        )

        G.add_node(tid, runtime=float(runtime))

    # ========= 加边 =========
    for t in tasks:
        tid = str(t.get("id"))

        # parents 形式
        if "parents" in t:
            for p in t["parents"]:
                G.add_edge(str(p), tid)

        # children 形式
        if "children" in t:
            for c in t["children"]:
                G.add_edge(tid, str(c))

        if not nx.is_directed_acyclic_graph(G):
            raise ValueError(f"Graph is not DAG: {json_path}")

    return G
