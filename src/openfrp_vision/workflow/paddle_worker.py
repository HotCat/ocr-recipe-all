from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2

from openfrp_vision.workflow.nodes import _collect_ocr_texts


RESULT_PREFIX = "__OPENFRP_OCR_RESULT__"
_OCR = None
_OCR_KEY = None


def _run_ocr(image_path: Path, params: dict[str, Any]) -> dict[str, Any]:
    global _OCR, _OCR_KEY
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read OCR input image: {image_path}")

    import paddleocr as paddleocr_module
    from paddleocr import PaddleOCR

    major_version = int(str(getattr(paddleocr_module, "__version__", "2")).split(".", 1)[0])
    lang = str(params.get("lang", "en"))
    ocr_version = str(params.get("ocr_version", "PP-OCRv3"))
    use_angle_cls = bool(params.get("use_angle_cls", True))
    key = (
        major_version,
        lang,
        ocr_version,
        use_angle_cls,
        bool(params.get("enable_mkldnn", False)),
        bool(params.get("ir_optim", False)),
    )
    if _OCR is None or _OCR_KEY != key:
        if major_version >= 3:
            _OCR = PaddleOCR(lang=lang, ocr_version=ocr_version, use_textline_orientation=use_angle_cls)
        else:
            _OCR = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                ocr_version=ocr_version,
                show_log=False,
                use_gpu=False,
                enable_mkldnn=bool(params.get("enable_mkldnn", False)),
                ir_optim=bool(params.get("ir_optim", False)),
            )
        _OCR_KEY = key

    if major_version >= 3:
        raw = _OCR.ocr(image)
    else:
        raw = _OCR.ocr(image, cls=use_angle_cls)

    texts, scores = _collect_ocr_texts(raw, float(params.get("min_score", 0.0)))
    text = str(params.get("join_with", "")).join(texts)
    if not text and params.get("debug_text"):
        text = str(params["debug_text"])
    return {
        "ok": True,
        "text": text,
        "scores": scores,
        "count": len(texts),
        "version": getattr(paddleocr_module, "__version__", "unknown"),
    }


def _print_result(result: dict[str, Any]) -> None:
    print(f"{RESULT_PREFIX}{json.dumps(result, ensure_ascii=False)}", flush=True)


def _serve() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = str(request.get("id", ""))
            result = _run_ocr(Path(str(request["image_path"])), dict(request.get("params", {})))
            result["id"] = request_id
        except BaseException as exc:
            result = {"ok": False, "id": locals().get("request_id", ""), "error": f"{type(exc).__name__}: {exc}"}
        _print_result(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if args == ["--serve"]:
        return _serve()
    if len(args) != 2:
        _print_result({"ok": False, "error": "usage: paddle_worker IMAGE PARAMS_JSON"})
        return 2
    image_path = Path(args[0])
    params = json.loads(args[1])
    try:
        result = _run_ocr(image_path, params)
    except BaseException as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    _print_result(result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
