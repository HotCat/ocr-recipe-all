from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openfrp_vision.camera.base import FrameSnapshot


class OverlayMode(str, Enum):
    HIDDEN = "hidden"
    ROI = "roi"
    GRAPH = "graph"
    EDIT = "edit"


@dataclass(frozen=True)
class ShortcutPressed:
    name: str


@dataclass(frozen=True)
class FrameArrived:
    frame_id: int
    timestamp_s: float
    size: tuple[int, int]


@dataclass(frozen=True)
class WorkflowRequested:
    snapshot: FrameSnapshot
    recipe_revision: int


@dataclass(frozen=True)
class WorkflowFinished:
    run_id: str
    passed: bool
    result: dict[str, Any]


@dataclass
class AppState:
    overlay_mode: OverlayMode = OverlayMode.GRAPH
    active_recipe_revision: int = 0
    latest_frame_id: int | None = None
    workflow_busy: bool = False
    last_result: dict[str, Any] | None = None
    counters: dict[str, int] = field(default_factory=lambda: {"ok": 0, "ng": 0})


Event = ShortcutPressed | FrameArrived | WorkflowRequested | WorkflowFinished


def reduce_state(state: AppState, event: Event) -> AppState:
    if isinstance(event, FrameArrived):
        state.latest_frame_id = event.frame_id
    elif isinstance(event, WorkflowRequested):
        state.workflow_busy = True
    elif isinstance(event, WorkflowFinished):
        state.workflow_busy = False
        state.last_result = event.result
        state.counters["ok" if event.passed else "ng"] += 1
    elif isinstance(event, ShortcutPressed):
        if event.name == "toggle_overlay":
            state.overlay_mode = {
                OverlayMode.HIDDEN: OverlayMode.ROI,
                OverlayMode.ROI: OverlayMode.GRAPH,
                OverlayMode.GRAPH: OverlayMode.EDIT,
                OverlayMode.EDIT: OverlayMode.HIDDEN,
            }[state.overlay_mode]
    return state
