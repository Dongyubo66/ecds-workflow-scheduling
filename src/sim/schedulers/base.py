from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, TYPE_CHECKING
import networkx as nx

if TYPE_CHECKING:
    from src.sim.resources import Machine


@dataclass
class ReadyTask:
    task_id: str
    wf_id: str
    ready_time: float  # earliest time can start (deps+arrival+release)


class SchedulerBase:
    name: str = "BASE"

    def on_new_workflow(self, wf_id: str, dag: nx.DiGraph) -> None:
        return

    def select(
        self,
        now: float,
        ready_tasks: List[ReadyTask],
        dags: Dict[str, nx.DiGraph],
        machines: List["Machine"],
        machine_available_time: Dict[str, float],
    ) -> Optional[tuple]:
        """
        Return (ReadyTask, machine_name) or None if choose to idle.
        """
        raise NotImplementedError
