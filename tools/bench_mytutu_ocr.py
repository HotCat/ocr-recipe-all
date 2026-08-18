from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from PySide6.QtCore import QCoreApplication

from openfrp_vision.camera.hikvision import HikvisionCameraAdapter
from openfrp_vision.core.profiles import ProfileStore
from openfrp_vision.workflow.model import RecipeGraph
from openfrp_vision.workflow.nodes import build_definitions, warm_paddle_worker


def _load_graph(profile_name: str) -> RecipeGraph:
    store = ProfileStore.load()
    profile_id = next(
        (item for item in store.profile_ids() if item == profile_name or store.profile_name(item) == profile_name),
        "",
    )
    if not profile_id:
        raise RuntimeError(f"profile not found: {profile_name}")
    data = store.graph_data(profile_id)
    if data is None:
        raise RuntimeError(f"profile has no graph: {profile_name}")
    return RecipeGraph.from_dict(build_definitions(), data)


def _apply_camera_settings(camera: HikvisionCameraAdapter, graph: RecipeGraph) -> None:
    for node in graph.nodes.values():
        if node.type_name == "camera_settings":
            camera.set_property("exposure_us", int(node.params.get("exposure_us", 30000)))
            camera.set_property("gamma", int(node.params.get("gamma", 100)))
            camera.set_property("contrast", int(node.params.get("contrast", 100)))
            camera.set_property("analog_gain", int(node.params.get("analog_gain", 16)))
            camera.set_property("ae_enabled", bool(node.params.get("ae_enabled", False)))
            camera.set_property("reverse_x", bool(node.params.get("reverse_x", False)))
            camera.set_property("reverse_y", bool(node.params.get("reverse_y", False)))
            return


def _wait_frame(app: QCoreApplication, camera: HikvisionCameraAdapter, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        snapshot = camera.latest_snapshot()
        if snapshot is not None:
            return snapshot
        time.sleep(0.02)
    raise RuntimeError("timed out waiting for real camera frame")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="mytutu")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--warm", action="store_true")
    args = parser.parse_args()

    app = QCoreApplication.instance() or QCoreApplication([])
    graph = _load_graph(args.profile)
    camera = HikvisionCameraAdapter()
    started_at = time.perf_counter()
    _apply_camera_settings(camera, graph)
    if not camera.start():
        raise RuntimeError(camera.last_error or camera.last_status)
    try:
        print(f"camera_start_s={time.perf_counter() - started_at:.3f}")
        snapshot = _wait_frame(app, camera, args.timeout)
        print(f"frame={snapshot.frame_id} size={snapshot.size} status={camera.hud_status}")
        Path("debug").mkdir(exist_ok=True)
        cv2.imwrite("debug/mytutu_camera_capture.png", snapshot.image_bgr)
        if args.warm:
            for node in graph.nodes.values():
                if node.enabled and node.type_name == "ocr":
                    warm_started = time.perf_counter()
                    warm_paddle_worker(dict(node.params))
                    print(f"warm_ocr_s={time.perf_counter() - warm_started:.3f}")

        for run_index in range(args.runs):
            for node in graph.nodes.values():
                if node.type_name == "frame_source":
                    node.params["snapshot"] = snapshot
            started = time.perf_counter()
            results = graph.execute()
            elapsed = time.perf_counter() - started
            print(f"run={run_index + 1} graph_s={elapsed:.3f}")
            for node_id, result in results.items():
                if result.elapsed_ms >= 5 or graph.nodes[node_id].type_name in {"ocr", "qr_code_ocr", "barcode_ocr"}:
                    print(
                        f"  {node_id:18s} {graph.nodes[node_id].type_name:14s} "
                        f"{result.elapsed_ms:8.1f} ms {result.summary}"
                    )
    finally:
        camera.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
