from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QTimer


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openfrp_vision.camera.hikvision import HikvisionCameraAdapter, is_available, unavailable_reason  # noqa: E402


def main() -> int:
    app = QCoreApplication.instance() or QCoreApplication([])
    print("sdk_available", is_available())
    if not is_available():
        print("reason", unavailable_reason())
        return 1

    adapter = HikvisionCameraAdapter()
    result = {"code": 3}
    adapter.status.connect(lambda message: print("status", message))
    adapter.error.connect(lambda message: print("error", message))

    def on_frame(snapshot) -> None:  # type: ignore[no-untyped-def]
        print("frame", snapshot.frame_id, snapshot.image_bgr.shape, snapshot.image_bgr.dtype)
        print("result", "ok")
        result["code"] = 0
        app.quit()

    adapter.frame_ready.connect(on_frame)

    if not adapter.start():
        print("device_count", adapter.device_count)
        print("hud_status", adapter.hud_status)
        return 2

    deadline = time.monotonic() + 5.0
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(app.quit)
    timer.start(5000)
    try:
        app.exec()
        snapshot = adapter.latest_snapshot()
        if snapshot is None:
            elapsed = time.monotonic() - deadline + 5.0
            print("result", f"camera opened but no frame arrived within {elapsed:.1f} seconds")
            return int(result["code"])
        print("latest", snapshot.frame_id, snapshot.image_bgr.shape, snapshot.image_bgr.dtype)
        return int(result["code"])
    finally:
        timer.stop()
        adapter.stop()


if __name__ == "__main__":
    raise SystemExit(main())
