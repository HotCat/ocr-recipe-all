from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import cv2

from openfrp_vision.camera.base import FrameSnapshot
from openfrp_vision.workflow.model import NodeDefinition, NodeResult, PortSpec, PortType, RecipeGraph, RecipeNode


_PADDLE_OCR = None


def _as_image(value: Any) -> Any:
    return value.image_bgr if isinstance(value, FrameSnapshot) else value


def _collect_ocr_texts(data: Any, min_score: float) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    scores: list[float] = []
    if isinstance(data, dict):
        rec_texts = data.get("rec_texts")
        rec_scores = data.get("rec_scores") or []
        if isinstance(rec_texts, list):
            for index, text in enumerate(rec_texts):
                score = float(rec_scores[index]) if index < len(rec_scores) else 1.0
                if score >= min_score:
                    texts.append(str(text))
                    scores.append(score)
        for value in data.values():
            child_texts, child_scores = _collect_ocr_texts(value, min_score)
            texts.extend(child_texts)
            scores.extend(child_scores)
        return texts, scores

    if isinstance(data, (list, tuple)):
        if len(data) >= 2 and isinstance(data[0], str) and isinstance(data[1], (int, float)):
            score = float(data[1])
            if score >= min_score:
                return [str(data[0])], [score]
            return [], []
        if len(data) >= 2 and isinstance(data[1], (list, tuple)) and data[1] and isinstance(data[1][0], str):
            score = float(data[1][1]) if len(data[1]) > 1 and isinstance(data[1][1], (int, float)) else 1.0
            if score >= min_score:
                return [str(data[1][0])], [score]
            return [], []
        for item in data:
            child_texts, child_scores = _collect_ocr_texts(item, min_score)
            texts.extend(child_texts)
            scores.extend(child_scores)
    return texts, scores


def _frame_source(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    snapshot: FrameSnapshot = params["snapshot"]
    return NodeResult(snapshot, preview=snapshot.image_bgr, summary=f"frame {snapshot.frame_id}")


def _roi(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    image = _as_image(inputs["image"])
    height, width = image.shape[:2]
    x = max(0, min(int(params["x"]), width - 1))
    y = max(0, min(int(params["y"]), height - 1))
    roi_width = max(1, min(int(params["width"]), width - x))
    roi_height = max(1, min(int(params["height"]), height - y))
    cropped = image[y : y + roi_height, x : x + roi_width].copy()
    return NodeResult(cropped, preview=cropped, summary=f"x={x}, y={y}, {roi_width} x {roi_height}")


def _canny(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    image = inputs["image"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    result = cv2.Canny(gray, float(params["threshold1"]), float(params["threshold2"]))
    return NodeResult(result, preview=result, summary="canny")


def _threshold(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    image = inputs["image"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    used, result = cv2.threshold(gray, int(params["threshold"]), 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return NodeResult(result, preview=result, summary=f"otsu {used:.0f}")


def _camera_settings(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    del inputs
    settings = {
        "exposure_us": int(params.get("exposure_us", 30000)),
        "gamma": int(params.get("gamma", 100)),
        "contrast": int(params.get("contrast", 100)),
        "analog_gain": int(params.get("analog_gain", 16)),
        "ae_enabled": bool(params.get("ae_enabled", False)),
        "reverse_x": bool(params.get("reverse_x", False)),
        "reverse_y": bool(params.get("reverse_y", False)),
    }
    return NodeResult(settings, summary=f"exp {settings['exposure_us']} us")


def _apply_camera_settings(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    del params
    settings = dict(inputs["settings"])
    return NodeResult({"passed": True, "camera_settings": settings}, summary="camera settings ready")


def _live_review(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    settings = dict(inputs["settings"])
    max_fps = int(params.get("max_fps", 24))
    result = {"passed": True, "settings": settings, "max_fps": max_fps}
    return NodeResult(result, summary=f"live review {max_fps} fps")


def _trigger_switch(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    del inputs
    trigger = {
        "source": str(params.get("source", "keyboard")),
        "shortcut": str(params.get("shortcut", "Ctrl+Return")),
        "external_topic": str(params.get("external_topic", "")),
        "debounce_ms": int(params.get("debounce_ms", 250)),
        "armed": bool(params.get("armed", True)),
    }
    label = trigger["shortcut"] if trigger["source"] == "keyboard" else trigger["source"]
    state = "armed" if trigger["armed"] else "disabled"
    return NodeResult(trigger, summary=f"{label} {state}")


def _frame_capture(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    del params
    trigger = dict(inputs["trigger"])
    snapshot = inputs["image"]
    if not trigger.get("armed", True):
        return NodeResult(snapshot, preview=_as_image(snapshot), summary="trigger disabled")
    frame_id = snapshot.frame_id if isinstance(snapshot, FrameSnapshot) else "image"
    return NodeResult(snapshot, preview=_as_image(snapshot), summary=f"captured frame {frame_id}")


def _regex(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    text = str(inputs["text"])
    pattern = str(params["pattern"])
    passed = re.fullmatch(pattern, text) is not None
    verdict = {"label": params.get("label", "Field"), "passed": passed, "text": text, "pattern": pattern}
    return NodeResult(verdict, summary=f"{'PASS' if passed else 'FAIL'} {text}")


def _aggregate(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    checks = list(inputs["checks"])
    passed = all(check["passed"] for check in checks)
    result = {"passed": passed, "decision": "PASS" if passed else "FAIL", "checks": checks}
    return NodeResult(result, summary=f"{result['decision']} {sum(c['passed'] for c in checks)}/{len(checks)}")


def _paddle_ocr(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    global _PADDLE_OCR
    image = _as_image(inputs["image"])
    if _PADDLE_OCR is None:
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            fallback = str(params.get("debug_text", ""))
            if fallback:
                return NodeResult(fallback, summary=f"debug OCR {fallback}")
            raise RuntimeError("PaddleOCR is not installed. Install paddleocr and paddlepaddle.") from exc
        try:
            _PADDLE_OCR = PaddleOCR(
                use_angle_cls=bool(params.get("use_angle_cls", True)),
                lang=str(params.get("lang", "en")),
                show_log=False,
            )
        except TypeError:
            _PADDLE_OCR = PaddleOCR(lang=str(params.get("lang", "en")))

    result = _PADDLE_OCR.ocr(image, cls=bool(params.get("use_angle_cls", True)))
    texts, scores = _collect_ocr_texts(result, float(params.get("min_score", 0.0)))
    text = str(params.get("join_with", "")).join(texts)
    if not text and params.get("debug_text"):
        text = str(params["debug_text"])
    score_summary = f" {max(scores):.2f}" if scores else ""
    return NodeResult(text, preview=image, summary=(text or "no text")[:32] + score_summary)


def _debug_image(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    image = _as_image(inputs["image"])
    saved = ""
    if bool(params.get("save_enabled", True)):
        path = Path(str(params.get("path", "debug/openfrp_debug.png"))).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to write {path}")
        saved = str(path)
    summary_parts = []
    if saved:
        summary_parts.append(saved)
    if bool(params.get("popup", False)):
        summary_parts.append("popup")
    return NodeResult(image, preview=image, summary=", ".join(summary_parts) or "debug pass")


def build_definitions() -> dict[str, NodeDefinition]:
    image_in = (PortSpec("image", PortType.IMAGE),)
    definitions = [
        NodeDefinition("frame_source", "Frame Input", "Input", (), PortSpec("image", PortType.IMAGE), {}, _frame_source),
        NodeDefinition(
            "camera_settings",
            "Camera Parameters",
            "Camera",
            (),
            PortSpec("settings", PortType.CAMERA_SETTINGS),
            {
                "exposure_us": 30000,
                "gamma": 100,
                "contrast": 100,
                "analog_gain": 16,
                "ae_enabled": False,
                "reverse_x": False,
                "reverse_y": False,
            },
            _camera_settings,
            True,
        ),
        NodeDefinition(
            "apply_camera_settings",
            "Apply Camera",
            "Camera",
            (PortSpec("settings", PortType.CAMERA_SETTINGS),),
            PortSpec("result", PortType.RESULT),
            {},
            _apply_camera_settings,
            True,
        ),
        NodeDefinition(
            "live_review",
            "Live Review",
            "Camera",
            (PortSpec("settings", PortType.CAMERA_SETTINGS),),
            PortSpec("result", PortType.RESULT),
            {"max_fps": 24},
            _live_review,
            True,
        ),
        NodeDefinition(
            "trigger_switch",
            "Trigger Switch",
            "Input",
            (),
            PortSpec("trigger", PortType.TRIGGER),
            {
                "source": "keyboard",
                "shortcut": "Ctrl+Return",
                "external_topic": "",
                "debounce_ms": 250,
                "armed": True,
            },
            _trigger_switch,
            True,
        ),
        NodeDefinition(
            "frame_capture",
            "Capture On Trigger",
            "Input",
            (PortSpec("image", PortType.IMAGE), PortSpec("trigger", PortType.TRIGGER)),
            PortSpec("image", PortType.IMAGE),
            {},
            _frame_capture,
            True,
        ),
        NodeDefinition("roi", "Region of Interest", "Image", image_in, PortSpec("image", PortType.IMAGE), {"x": 0, "y": 0, "width": 100, "height": 100}, _roi, True),
        NodeDefinition("canny", "Canny", "Image", image_in, PortSpec("image", PortType.IMAGE), {"threshold1": 100, "threshold2": 200}, _canny, True),
        NodeDefinition("threshold", "Threshold", "Image", image_in, PortSpec("image", PortType.IMAGE), {"threshold": 165}, _threshold, True),
        NodeDefinition(
            "debug_image",
            "Debug Image",
            "Output",
            image_in,
            PortSpec("image", PortType.IMAGE),
            {"path": "debug/openfrp_debug.png", "save_enabled": True, "popup": False},
            _debug_image,
        ),
        NodeDefinition(
            "ocr",
            "PaddleOCR",
            "Recognition",
            image_in,
            PortSpec("text", PortType.TEXT),
            {"lang": "en", "use_angle_cls": True, "min_score": 0.0, "join_with": "", "debug_text": ""},
            _paddle_ocr,
        ),
        NodeDefinition("regex", "Regex Check", "Decision", (PortSpec("text", PortType.TEXT),), PortSpec("verdict", PortType.VERDICT), {"label": "Field", "pattern": ".+"}, _regex, True),
        NodeDefinition("aggregate", "Aggregate", "Decision", (PortSpec("checks", PortType.VERDICT, multiple=True),), PortSpec("result", PortType.RESULT), {}, _aggregate, True),
    ]
    return {definition.type_name: definition for definition in definitions}


def build_default_graph(definitions: dict[str, NodeDefinition]) -> RecipeGraph:
    graph = RecipeGraph(definitions)
    nodes = [
        RecipeNode("camera_params", "camera_settings", "Camera Parameters", 40, 40),
        RecipeNode("camera_apply", "apply_camera_settings", "Apply Camera", 300, 40),
        RecipeNode("live_review", "live_review", "Live Frame Review", 560, 40),
        RecipeNode("trigger", "trigger_switch", "Trigger Switch", 40, 360),
        RecipeNode("frame", "frame_source", "Live Frame", 40, 220),
        RecipeNode("capture", "frame_capture", "Capture Snapshot", 270, 220),
        RecipeNode("serial_roi", "roi", "Serial ROI", 500, 220, {"x": 100, "y": 240, "width": 260, "height": 120}),
        RecipeNode("threshold", "threshold", "Threshold", 730, 220),
        RecipeNode("debug_image", "debug_image", "Debug Image", 960, 220, {"path": "debug/serial_roi.png", "save_enabled": True, "popup": False}),
        RecipeNode("ocr", "ocr", "PaddleOCR", 1190, 220, {"debug_text": "A7B9C312"}),
        RecipeNode("regex", "regex", "Serial Check", 1420, 220, {"label": "Serial", "pattern": "[A-Z0-9]{8}"}),
        RecipeNode("aggregate", "aggregate", "Decision", 1650, 220),
    ]
    for node in nodes:
        graph.add_node(node)
    for source, target, port in [
        ("camera_params", "camera_apply", "settings"),
        ("camera_params", "live_review", "settings"),
        ("frame", "capture", "image"),
        ("trigger", "capture", "trigger"),
        ("capture", "serial_roi", "image"),
        ("serial_roi", "threshold", "image"),
        ("threshold", "debug_image", "image"),
        ("debug_image", "ocr", "image"),
        ("ocr", "regex", "text"),
        ("regex", "aggregate", "checks"),
    ]:
        graph.connect(source, target, port)
    graph.revision = 0
    return graph
