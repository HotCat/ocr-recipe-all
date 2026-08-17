from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from uuid import uuid4

from openfrp_vision.camera.base import FrameSnapshot
from openfrp_vision.workflow.model import NodeResult, RecipeGraph


@dataclass(frozen=True)
class WorkflowRun:
    run_id: str
    revision: int
    frame_id: int
    results: dict[str, NodeResult]

    @property
    def final_result(self) -> dict:
        if not self.results:
            return {"passed": False, "decision": "FAIL", "checks": []}
        value = next(reversed(self.results.values())).value
        return value if isinstance(value, dict) else {"passed": False, "value": value}


class WorkflowExecutor:
    def __init__(self, max_workers: int = 1) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="workflow")

    def submit(self, graph: RecipeGraph, snapshot: FrameSnapshot) -> Future[WorkflowRun]:
        run_id = uuid4().hex

        def run() -> WorkflowRun:
            for node in graph.nodes.values():
                if node.type_name == "frame_source":
                    node.params["snapshot"] = snapshot
            results = graph.execute()
            return WorkflowRun(run_id=run_id, revision=graph.revision, frame_id=snapshot.frame_id, results=results)

        return self._pool.submit(run)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
