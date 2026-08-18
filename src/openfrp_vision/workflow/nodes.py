from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import cv2
import numpy as np

from openfrp_vision.camera.base import FrameSnapshot
from openfrp_vision.workflow.model import NodeDefinition, NodeResult, PortSpec, PortType, RecipeGraph, RecipeNode


_PADDLE_OCR = None
_PADDLE_OCR_USE_CLS_ARG = True
_PADDLE_WORKER_PREFIX = "__OPENFRP_OCR_RESULT__"
_PADDLE_WORKER_CLIENT = None


def _as_image(value: Any) -> Any:
    return value.image_bgr if isinstance(value, FrameSnapshot) else value


def _preview_bgr(image: Any) -> Any:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _point_sets(points: Any) -> list[Any]:
    if points is None:
        return []
    array = cv2.UMat.get(points) if isinstance(points, cv2.UMat) else points
    try:
        point_array = np.asarray(array, dtype=float)
    except Exception:
        return []
    if point_array.size == 0:
        return []
    if point_array.ndim == 2 and point_array.shape[1] == 2:
        return [point_array]
    if point_array.ndim == 3 and point_array.shape[-1] == 2:
        return [point_array[index] for index in range(point_array.shape[0])]
    if point_array.ndim == 4 and point_array.shape[-1] == 2:
        return [point_array[index, 0] for index in range(point_array.shape[0])]
    return []


def _draw_detected_regions(preview: Any, points: Any, labels: list[str]) -> None:
    for index, point_set in enumerate(_point_sets(points)):
        pts = point_set.astype("int32").reshape((-1, 1, 2))
        cv2.polylines(preview, [pts], True, (0, 220, 255), 2)
        if pts.size:
            x = int(pts[:, 0, 0].min())
            y = int(pts[:, 0, 1].min())
            label = labels[index] if index < len(labels) else "code"
            cv2.putText(
                preview,
                label[:32],
                (max(0, x), max(14, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )


def _decode_qr_image(detector: Any, image: Any) -> tuple[list[str], Any]:
    try:
        ok, decoded_info, points, _straight = detector.detectAndDecodeMulti(image)
    except cv2.error:
        ok = False
        decoded_info = ()
        points = None
    if ok:
        decoded = [str(text) for text in decoded_info if str(text)]
        if decoded:
            return decoded, points

    try:
        text, points, _straight = detector.detectAndDecode(image)
    except cv2.error:
        return [], None
    return ([str(text)] if text else []), points


def _ordered_quad(points: Any) -> Any | None:
    point_sets = _point_sets(points)
    if not point_sets:
        return None
    pts = np.asarray(point_sets[0], dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        return None
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(sums)], pts[np.argmin(diffs)], pts[np.argmax(sums)], pts[np.argmax(diffs)]],
        dtype=np.float32,
    )


def _scale_points(points: Any, factor: float) -> Any:
    if points is None or factor == 1.0:
        return points
    return np.asarray(points, dtype=np.float32) / factor


def _qr_preprocess_variants(image: Any, points: Any, params: dict[str, Any]) -> list[tuple[str, Any, float, Any]]:
    variants: list[tuple[str, Any, float, Any]] = [("raw", image, 1.0, points)]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    variants.append(("gray", gray, 1.0, points))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(("clahe", clahe, 1.0, points))

    sharp = cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32))
    variants.append(("sharpen", sharp, 1.0, points))

    for block_size in (21, 31, 45):
        for constant in (2, 5):
            adaptive = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                constant,
            )
            variants.append((f"adaptive_gaussian_{block_size}_{constant}", adaptive, 1.0, points))
            if bool(params.get("try_inverted", True)):
                variants.append((f"adaptive_gaussian_inv_{block_size}_{constant}", 255 - adaptive, 1.0, points))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    variants.append(("otsu", otsu, 1.0, points))
    if bool(params.get("try_inverted", True)):
        variants.append(("otsu_inv", 255 - otsu, 1.0, points))

    max_scale = max(1, int(params.get("max_scale", 4)))
    for scale in range(2, max_scale + 1):
        up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append((f"upscale_{scale}", up, float(scale), points))
        adaptive_up = cv2.adaptiveThreshold(
            up,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            3,
        )
        variants.append((f"upscale_{scale}_adaptive", adaptive_up, float(scale), points))

    quad = _ordered_quad(points)
    if quad is not None:
        for size in (280, 420, 560):
            dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(quad, dst)
            warped = cv2.warpPerspective(image, matrix, (size, size), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
            bordered = cv2.copyMakeBorder(warped, size // 8, size // 8, size // 8, size // 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            variants.append((f"rectified_{size}", bordered, 1.0, points))
            warped_gray = cv2.cvtColor(bordered, cv2.COLOR_BGR2GRAY)
            warped_adaptive = cv2.adaptiveThreshold(
                warped_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                3,
            )
            variants.append((f"rectified_{size}_adaptive", warped_adaptive, 1.0, points))

    return variants


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


def _paddle_ocr_in_process(image: Any, params: dict[str, Any]) -> NodeResult:
    global _PADDLE_OCR, _PADDLE_OCR_USE_CLS_ARG
    if _PADDLE_OCR is None:
        try:
            import paddleocr as paddleocr_module
            from paddleocr import PaddleOCR
        except Exception as exc:
            fallback = str(params.get("debug_text", ""))
            if fallback:
                return NodeResult(fallback, summary=f"debug OCR {fallback}")
            raise RuntimeError("PaddleOCR is not installed. Install paddleocr and paddlepaddle.") from exc
        major_version = int(str(getattr(paddleocr_module, "__version__", "2")).split(".", 1)[0])
        try:
            if major_version >= 3:
                _PADDLE_OCR = PaddleOCR(
                    lang=str(params.get("lang", "en")),
                    ocr_version=str(params.get("ocr_version", "PP-OCRv3")),
                    use_textline_orientation=bool(params.get("use_angle_cls", True)),
                )
                _PADDLE_OCR_USE_CLS_ARG = False
            else:
                _PADDLE_OCR = PaddleOCR(
                    use_angle_cls=bool(params.get("use_angle_cls", True)),
                    lang=str(params.get("lang", "en")),
                    ocr_version=str(params.get("ocr_version", "PP-OCRv3")),
                    show_log=False,
                    use_gpu=False,
                    enable_mkldnn=bool(params.get("enable_mkldnn", False)),
                    ir_optim=bool(params.get("ir_optim", False)),
                )
                _PADDLE_OCR_USE_CLS_ARG = True
        except (TypeError, ValueError):
            _PADDLE_OCR = PaddleOCR(
                lang=str(params.get("lang", "en")),
                ocr_version=str(params.get("ocr_version", "PP-OCRv3")),
                use_textline_orientation=bool(params.get("use_angle_cls", True)),
            )
            _PADDLE_OCR_USE_CLS_ARG = False

    if _PADDLE_OCR_USE_CLS_ARG:
        result = _PADDLE_OCR.ocr(image, cls=bool(params.get("use_angle_cls", True)))
    else:
        result = _PADDLE_OCR.ocr(image)
    texts, scores = _collect_ocr_texts(result, float(params.get("min_score", 0.0)))
    text = str(params.get("join_with", "")).join(texts)
    if not text and params.get("debug_text"):
        text = str(params["debug_text"])
    score_summary = f" {max(scores):.2f}" if scores else ""
    return NodeResult(text, preview=image, summary=(text or "no text")[:32] + score_summary)


def _parse_paddle_worker(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_PADDLE_WORKER_PREFIX):
            return json.loads(line[len(_PADDLE_WORKER_PREFIX) :])
    return None


def _signal_name(returncode: int) -> str:
    if returncode == -4:
        return "SIGILL illegal instruction"
    if returncode < 0:
        return f"signal {-returncode}"
    return f"exit {returncode}"


class _PaddleWorkerClient:
    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._request_id = 0

    def run(self, image_path: Path, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        with self._lock:
            restart_attempts = max(0, int(params.get("restart_attempts", 1)))
            crash_errors: list[str] = []
            for attempt in range(restart_attempts + 1):
                result = self._run_once(image_path, params, timeout_s)
                if not result.pop("_restartable", False):
                    if crash_errors and result.get("ok", False):
                        result["worker_restarts"] = len(crash_errors)
                        result["restart_errors"] = crash_errors
                    return result

                crash_errors.append(str(result.get("error", "PaddleOCR worker crashed")))
                if attempt < restart_attempts:
                    continue

                result["worker_restarts"] = len(crash_errors) - 1
                result["restart_errors"] = crash_errors
                result["error"] = f"{crash_errors[-1]} (restart attempts exhausted)"
                return result

            return {"ok": False, "error": "PaddleOCR worker failed"}

    def _run_once(self, image_path: Path, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        process = self._ensure_process(params)
        self._request_id += 1
        request_id = str(self._request_id)
        payload = json.dumps({"id": request_id, "image_path": str(image_path), "params": params}, ensure_ascii=False)
        assert process.stdin is not None
        assert process.stdout is not None
        try:
            process.stdin.write(payload + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            return self._crash_result(process)

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return self._crash_result(process)
            ready, _, _ = select.select([process.stdout], [], [], min(0.1, max(0.0, deadline - time.monotonic())))
            if not ready:
                continue
            line = process.stdout.readline()
            if not line:
                return self._crash_result(process)
            if not line.startswith(_PADDLE_WORKER_PREFIX):
                continue
            result = json.loads(line[len(_PADDLE_WORKER_PREFIX) :])
            if str(result.get("id", request_id)) == request_id:
                return result

        self.stop()
        return {"ok": False, "error": "PaddleOCR worker timeout"}

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=1)
            except Exception:
                pass

    def _ensure_process(self, params: dict[str, Any]) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._process = subprocess.Popen(
            [sys.executable, "-B", "-m", "openfrp_vision.workflow.paddle_worker", "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "OMP_NUM_THREADS": str(params.get("threads", 1))},
        )
        return self._process

    def _crash_result(self, process: subprocess.Popen[str]) -> dict[str, Any]:
        code = process.poll()
        self._discard_process(process)
        if code is None:
            code = process.returncode
        return {"ok": False, "error": f"PaddleOCR worker crashed: {_signal_name(code or -1)}", "_restartable": True}

    def _discard_process(self, process: subprocess.Popen[str]) -> None:
        if self._process is process:
            self._process = None
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=1)
            else:
                process.wait(timeout=0.1)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=1)
            except Exception:
                pass


def _paddle_ocr_subprocess_once(image: Any, params: dict[str, Any]) -> NodeResult:
    preview = image.copy()
    fd, path_text = tempfile.mkstemp(prefix="openfrp_paddle_", suffix=".png")
    os.close(fd)
    image_path = Path(path_text)
    try:
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"failed to write OCR worker input image: {image_path}")
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "openfrp_vision.workflow.paddle_worker",
                str(image_path),
                json.dumps(params, ensure_ascii=False),
            ],
            capture_output=True,
            text=True,
            timeout=max(1.0, float(params.get("timeout_s", 60.0))),
            env={**os.environ, "OMP_NUM_THREADS": str(params.get("threads", 1))},
            check=False,
        )
    except subprocess.TimeoutExpired:
        return NodeResult("", preview=preview, summary="PaddleOCR timeout")
    finally:
        try:
            image_path.unlink()
        except FileNotFoundError:
            pass

    result = _parse_paddle_worker(completed.stdout)
    if completed.returncode != 0:
        if result and not result.get("ok", False):
            return NodeResult("", preview=preview, summary=f"PaddleOCR failed: {str(result.get('error', 'worker error'))[:96]}")
        stderr_tail = (completed.stderr or completed.stdout).splitlines()[-1:] or [""]
        detail = stderr_tail[0].strip()
        reason = _signal_name(completed.returncode)
        if detail:
            reason = f"{reason}: {detail[:80]}"
        return NodeResult("", preview=preview, summary=f"PaddleOCR crashed: {reason}")

    if not result:
        return NodeResult("", preview=preview, summary="PaddleOCR failed: worker returned no result")
    if not result.get("ok", False):
        return NodeResult("", preview=preview, summary=f"PaddleOCR failed: {str(result.get('error', 'worker error'))[:96]}")

    text = str(result.get("text", ""))
    scores = result.get("scores") or []
    if not text and params.get("debug_text"):
        text = str(params["debug_text"])
    score_summary = f" {max(float(score) for score in scores):.2f}" if scores else ""
    count = int(result.get("count", 0) or 0)
    prefix = f"OCR {count}" if count else "OCR"
    return NodeResult(text, preview=preview, summary=f"{prefix}: {(text or 'no text')[:32]}{score_summary}")


def _paddle_worker_client() -> _PaddleWorkerClient:
    global _PADDLE_WORKER_CLIENT
    if _PADDLE_WORKER_CLIENT is None:
        _PADDLE_WORKER_CLIENT = _PaddleWorkerClient()
    return _PADDLE_WORKER_CLIENT


def shutdown_paddle_worker() -> None:
    global _PADDLE_WORKER_CLIENT
    if _PADDLE_WORKER_CLIENT is not None:
        _PADDLE_WORKER_CLIENT.stop()
        _PADDLE_WORKER_CLIENT = None


atexit.register(shutdown_paddle_worker)


def warm_paddle_worker(params: dict[str, Any]) -> None:
    mode = str(params.get("run_mode", "worker"))
    if mode in {"in_process", "subprocess_once", "one_shot"}:
        return
    blank = np.full((32, 128, 3), 255, dtype=np.uint8)
    _paddle_ocr_worker(blank, {**params, "debug_text": ""})


def _paddle_ocr_worker(image: Any, params: dict[str, Any]) -> NodeResult:
    preview = image.copy()
    fd, path_text = tempfile.mkstemp(prefix="openfrp_paddle_", suffix=".png")
    os.close(fd)
    image_path = Path(path_text)
    try:
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"failed to write OCR worker input image: {image_path}")
        result = _paddle_worker_client().run(
            image_path,
            params,
            timeout_s=max(1.0, float(params.get("timeout_s", 60.0))),
        )
    finally:
        try:
            image_path.unlink()
        except FileNotFoundError:
            pass

    if not result.get("ok", False):
        return NodeResult("", preview=preview, summary=str(result.get("error", "PaddleOCR worker failed"))[:120])
    text = str(result.get("text", ""))
    scores = result.get("scores") or []
    if not text and params.get("debug_text"):
        text = str(params["debug_text"])
    score_summary = f" {max(float(score) for score in scores):.2f}" if scores else ""
    count = int(result.get("count", 0) or 0)
    prefix = f"OCR {count}" if count else "OCR"
    restart_count = int(result.get("worker_restarts", 0) or 0)
    restart_word = "restart" if restart_count == 1 else "restarts"
    restart_summary = f" after {restart_count} {restart_word}" if restart_count else ""
    return NodeResult(text, preview=preview, summary=f"{prefix}{restart_summary}: {(text or 'no text')[:32]}{score_summary}")


def _paddle_ocr(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    image = _as_image(inputs["image"])
    mode = str(params.get("run_mode", "worker"))
    if mode == "in_process":
        return _paddle_ocr_in_process(image, params)
    if mode in {"subprocess_once", "one_shot"}:
        return _paddle_ocr_subprocess_once(image, params)
    return _paddle_ocr_worker(image, params)


def _qr_code_ocr(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    image = _as_image(inputs["image"])
    detector = cv2.QRCodeDetector()
    preview = _preview_bgr(image)
    join_with = str(params.get("join_with", "\n"))
    debug_text = str(params.get("debug_text", ""))
    decoded, points = _decode_qr_image(detector, image)
    method = "raw"
    if not decoded and bool(params.get("preprocess", True)):
        for variant_name, variant, scale, fallback_points in _qr_preprocess_variants(image, points, params):
            decoded, variant_points = _decode_qr_image(detector, variant)
            if decoded:
                if variant_points is not None:
                    points = _scale_points(variant_points, scale) if scale != 1.0 else variant_points
                else:
                    points = fallback_points
                method = variant_name
                break

    if not decoded and debug_text:
        decoded = [debug_text]
    _draw_detected_regions(preview, points, decoded or ["QR"])
    text = join_with.join(decoded)
    count = len(decoded)
    summary = f"QR {count} {method}: {(text or 'no text')[:32]}" if count else f"QR no text ({method})"
    return NodeResult(text, preview=preview, summary=summary)


def _barcode_ocr(inputs: dict[str, Any], params: dict[str, Any]) -> NodeResult:
    image = _as_image(inputs["image"])
    preview = _preview_bgr(image)
    join_with = str(params.get("join_with", "\n"))
    debug_text = str(params.get("debug_text", ""))
    if not hasattr(cv2, "barcode_BarcodeDetector"):
        if debug_text:
            return NodeResult(debug_text, preview=preview, summary=f"debug barcode {debug_text}")
        raise RuntimeError("OpenCV barcode detector is unavailable. Install opencv-contrib-python.")

    detector = cv2.barcode_BarcodeDetector()
    decoded: list[str] = []
    types: list[str] = []
    points = None
    try:
        ok, decoded_info, decoded_types, points = detector.detectAndDecodeWithType(image)
    except (AttributeError, cv2.error):
        ok = False
        decoded_info = ()
        decoded_types = ()
    if ok:
        decoded = [str(text) for text in decoded_info if str(text)]
        types = [str(code_type) for code_type in decoded_types]
    else:
        try:
            ok, decoded_info, points, _straight = detector.detectAndDecodeMulti(image)
        except cv2.error:
            ok = False
            decoded_info = ()
        if ok:
            decoded = [str(text) for text in decoded_info if str(text)]
        else:
            text, points, _straight = detector.detectAndDecode(image)
            if text:
                decoded = [str(text)]

    if not decoded and debug_text:
        decoded = [debug_text]
    labels = [f"{types[index]} {text}".strip() if index < len(types) else text for index, text in enumerate(decoded)]
    _draw_detected_regions(preview, points, labels or ["barcode"])
    text = join_with.join(decoded)
    type_summary = f" ({', '.join(types[:3])})" if types else ""
    count = len(decoded)
    summary = f"Barcode {count}{type_summary}: {(text or 'no text')[:32]}" if count else "Barcode no text"
    return NodeResult(text, preview=preview, summary=summary)


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
        NodeDefinition(
            "roi",
            "Region of Interest",
            "Image",
            image_in,
            PortSpec("image", PortType.IMAGE),
            {"x": 0, "y": 0, "width": 100, "height": 100, "live_overlay": False},
            _roi,
            True,
        ),
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
            {
                "lang": "en",
                "ocr_version": "PP-OCRv3",
                "run_mode": "worker",
                "timeout_s": 60,
                "threads": 1,
                "restart_attempts": 1,
                "use_angle_cls": True,
                "min_score": 0.0,
                "join_with": "",
                "debug_text": "",
                "enable_mkldnn": False,
                "ir_optim": False,
            },
            _paddle_ocr,
        ),
        NodeDefinition(
            "qr_code_ocr",
            "QR Code OCR",
            "Recognition",
            image_in,
            PortSpec("text", PortType.TEXT),
            {"join_with": "\n", "debug_text": "", "preprocess": True, "try_inverted": True, "max_scale": 4},
            _qr_code_ocr,
            True,
        ),
        NodeDefinition(
            "barcode_ocr",
            "Barcode OCR",
            "Recognition",
            image_in,
            PortSpec("text", PortType.TEXT),
            {"join_with": "\n", "debug_text": ""},
            _barcode_ocr,
            True,
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
        RecipeNode("serial_roi", "roi", "Serial ROI", 500, 220, {"x": 100, "y": 240, "width": 260, "height": 120, "live_overlay": True}),
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
