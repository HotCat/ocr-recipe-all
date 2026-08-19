from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from uuid import uuid4

from openfrp_vision.camera.base import FrameSnapshot
from typing import Any

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

    def submit(self, graph: RecipeGraph, snapshot: FrameSnapshot, context: dict[str, Any] | None = None) -> Future[WorkflowRun]:
        run_id = uuid4().hex
        runtime_context = dict(context or {})

        def run() -> WorkflowRun:
            try:
                for node in graph.nodes.values():
                    self._clear_runtime_params(node.params)
                    if node.type_name == "frame_source":
                        node.params["snapshot"] = snapshot
                    if node.type_name == "sqlite_log":
                        node.params.update(
                            {
                                "_node_id": node.node_id,
                                "_run_id": run_id,
                                "_frame_id": snapshot.frame_id,
                                **runtime_context,
                            }
                        )
                    if node.type_name in {"serial_continuity", "serial_generator"}:
                        node.params.update({"_node_id": node.node_id, **runtime_context})
                results = graph.execute()
                return WorkflowRun(run_id=run_id, revision=graph.revision, frame_id=snapshot.frame_id, results=results)
            finally:
                for node in graph.nodes.values():
                    self._clear_runtime_params(node.params)

        return self._pool.submit(run)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _clear_runtime_params(self, params: dict[str, Any]) -> None:
        for key in list(params.keys()):
            if key == "snapshot" or key.startswith("_"):
                params.pop(key, None)
