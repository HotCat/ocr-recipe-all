from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class CameraSettings:
    exposure_us: int = 30000
    gamma: int = 100
    contrast: int = 100
    analog_gain: int = 16
    ae_enabled: bool = False
    reverse_x: bool = False
    reverse_y: bool = False


@dataclass
class CameraSettingRanges:
    exposure_min_us: int = 100
    exposure_max_us: int = 1000000
    exposure_step_us: int = 100
    gamma_min: int = 1
    gamma_max: int = 500
    contrast_min: int = 1
    contrast_max: int = 500
    analog_gain_min: int = 0
    analog_gain_max: int = 100


@dataclass(frozen=True)
class FrameSnapshot:
    frame_id: int
    timestamp_s: float
    image_bgr: np.ndarray

    @property
    def size(self) -> tuple[int, int]:
        height, width = self.image_bgr.shape[:2]
        return width, height


class CameraAdapter(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def latest_snapshot(self) -> FrameSnapshot | None:
        ...

    def set_property(self, name: str, value: float | bool | str) -> None:
        ...
