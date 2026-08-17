from __future__ import annotations

import os
import sys
import threading
import time
from ctypes import POINTER, byref, c_bool, c_ubyte, cast, memset, sizeof
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from openfrp_vision.camera.base import CameraSettingRanges, CameraSettings, FrameSnapshot

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


MVS_ROOT = os.environ.get("HIKVISION_MVS_ROOT", "/opt/MVS")
MVS_RUNENV = os.environ.get("MVCAM_COMMON_RUNENV") or os.path.join(MVS_ROOT, "lib")
MVS_IMPORT_DIR = os.path.join(MVS_ROOT, "Samples", "64", "Python", "MvImport")

os.environ.setdefault("MVCAM_COMMON_RUNENV", MVS_RUNENV)
if MVS_IMPORT_DIR not in sys.path:
    sys.path.insert(0, MVS_IMPORT_DIR)

try:
    from MvCameraControl_class import (  # type: ignore
        MV_ACCESS_Exclusive,
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_FRAME_OUT,
        MV_FRAME_OUT_INFO_EX,
        MV_GENTL_CAMERALINK_DEVICE,
        MV_GENTL_CXP_DEVICE,
        MV_GENTL_GIGE_DEVICE,
        MV_GENTL_XOF_DEVICE,
        MV_GIGE_DEVICE,
        MV_OK,
        MV_TRIGGER_MODE_OFF,
        MV_USB_DEVICE,
        MV_GrabStrategy_LatestImagesOnly,
        MVCC_FLOATVALUE,
        MVCC_INTVALUE_EX,
        MvCamera,
    )
    _SDK_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local SDK install
    _SDK_IMPORT_ERROR = exc


_SDK_INITIALIZED = False


def is_available() -> bool:
    return _SDK_IMPORT_ERROR is None


def unavailable_reason() -> str:
    if _SDK_IMPORT_ERROR is None:
        return ""
    return f"Hikvision MVS SDK import failed: {_SDK_IMPORT_ERROR}"


def _ensure_sdk_initialized() -> None:
    global _SDK_INITIALIZED
    if _SDK_IMPORT_ERROR is not None:
        raise RuntimeError(unavailable_reason())
    if not _SDK_INITIALIZED:
        ret = MvCamera.MV_CC_Initialize()
        if ret != MV_OK:
            raise RuntimeError(f"Hikvision SDK initialize failed: 0x{ret:x}")
        _SDK_INITIALIZED = True


def _decode_ctypes_string(value) -> str:
    data = memoryview(value).tobytes().split(b"\x00", 1)[0]
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _check(ret: int, action: str) -> None:
    if ret != MV_OK:
        raise RuntimeError(f"{action} failed: 0x{ret:x}")


def _is_gige(tlayer_type: int) -> bool:
    return tlayer_type in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE)


def _device_name_and_sn(info: "MV_CC_DEVICE_INFO") -> tuple[str, str]:
    tlayer_type = int(info.nTLayerType)
    special = info.SpecialInfo
    if tlayer_type in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
        model = _decode_ctypes_string(special.stGigEInfo.chModelName)
        sn = _decode_ctypes_string(special.stGigEInfo.chSerialNumber)
    elif tlayer_type == MV_USB_DEVICE:
        model = _decode_ctypes_string(special.stUsb3VInfo.chModelName)
        sn = _decode_ctypes_string(special.stUsb3VInfo.chSerialNumber)
    elif tlayer_type == MV_GENTL_CAMERALINK_DEVICE:
        model = _decode_ctypes_string(special.stCMLInfo.chModelName)
        sn = _decode_ctypes_string(special.stCMLInfo.chSerialNumber)
    elif tlayer_type == MV_GENTL_CXP_DEVICE:
        model = _decode_ctypes_string(special.stCXPInfo.chModelName)
        sn = _decode_ctypes_string(special.stCXPInfo.chSerialNumber)
    elif tlayer_type == MV_GENTL_XOF_DEVICE:
        model = _decode_ctypes_string(special.stXoFInfo.chModelName)
        sn = _decode_ctypes_string(special.stXoFInfo.chSerialNumber)
    else:
        model = f"Transport 0x{tlayer_type:x}"
        sn = ""
    return model.strip() or "Hikvision Camera", sn.strip()


class _LiveThread(QThread):
    frame_ready = Signal(np.ndarray)
    error = Signal(str)
    status = Signal(str)

    def __init__(self, camera: "_MvsCamera", max_fps: float = 24.0) -> None:
        super().__init__()
        self._camera = camera
        self._running = False
        self._min_interval_s = 1.0 / max(max_fps, 0.1)
        self._miss_count = 0

    def run(self) -> None:
        self._running = True
        next_emit_at = 0.0
        last_status_at = 0.0
        while self._running and not self.isInterruptionRequested():
            now = time.monotonic()
            if now < next_emit_at:
                self.msleep(max(1, int((next_emit_at - now) * 1000)))
                continue
            try:
                frame = self._camera.grab_preview_frame(timeout_ms=200)
            except Exception as exc:
                if self._running:
                    self.error.emit(f"Live view error: {exc}")
                continue
            if frame is not None:
                self._miss_count = 0
                self.frame_ready.emit(frame)
                next_emit_at = time.monotonic() + self._min_interval_s
            else:
                self._miss_count += 1
                if now - last_status_at >= 1.0:
                    self.status.emit(f"Camera opened, waiting for frames ({self._miss_count} misses)")
                    last_status_at = now

    def stop(self) -> None:
        self._running = False
        self.requestInterruption()
        self.wait(3000)


class _MvsCamera:
    def __init__(self) -> None:
        _ensure_sdk_initialized()
        self._cam: Optional[MvCamera] = None
        self._width = 0
        self._height = 0
        self._grabbing = False
        self._bgr_buffer = None
        self._bgr_buffer_size = 0
        self._grab_lock = threading.RLock()
        self._preview_max_side = 1600
        self.last_error = ""

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height

    def enumerate_devices(self) -> list[dict]:
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = (
            MV_GIGE_DEVICE
            | MV_USB_DEVICE
            | MV_GENTL_GIGE_DEVICE
            | MV_GENTL_CAMERALINK_DEVICE
            | MV_GENTL_CXP_DEVICE
            | MV_GENTL_XOF_DEVICE
        )
        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != MV_OK:
            self.last_error = f"enumerate devices failed: 0x{ret:x}"
            return []
        devices: list[dict] = []
        for idx in range(int(device_list.nDeviceNum)):
            info = cast(device_list.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
            model, sn = _device_name_and_sn(info)
            label = f"Hikvision {model}" + (f" ({sn})" if sn else "")
            devices.append({"name": label, "sn": sn, "dev_info": info})
        return devices

    def open(self, dev_info) -> None:
        self.close()
        self._cam = MvCamera()
        try:
            _check(self._cam.MV_CC_CreateHandle(dev_info), "create camera handle")
            _check(self._cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0), "open camera")
            if _is_gige(int(dev_info.nTLayerType)):
                packet_size = self._cam.MV_CC_GetOptimalPacketSize()
                if int(packet_size) > 0:
                    self._try_set_int("GevSCPSPacketSize", int(packet_size))
            self._try_set_enum_string("AcquisitionMode", "Continuous")
            self._try_set_enum("TriggerMode", MV_TRIGGER_MODE_OFF)
            self._try_set_bool("AcquisitionFrameRateEnable", False)
            self._width = max(self._get_int("Width", 0), 0)
            self._height = max(self._get_int("Height", 0), 0)
            self.apply_settings(CameraSettings())
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._cam is not None:
            if self._grabbing:
                try:
                    self._cam.MV_CC_StopGrabbing()
                except Exception:
                    pass
            try:
                self._cam.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                self._cam.MV_CC_DestroyHandle()
            except Exception:
                pass
        self._cam = None
        self._grabbing = False

    def start_live(self) -> None:
        if self._cam is None:
            raise RuntimeError("Camera is not open")
        if self._grabbing:
            return
        self._try_set_enum("TriggerMode", MV_TRIGGER_MODE_OFF)
        self._try_set_grab_strategy(MV_GrabStrategy_LatestImagesOnly)
        _check(self._cam.MV_CC_StartGrabbing(), "start live grabbing")
        self._grabbing = True

    def capture_frame(self, timeout_ms: int = 1000) -> np.ndarray | None:
        if self._cam is None:
            return None
        if not self._grabbing:
            self.start_live()
        return self._grab_bgr_frame(timeout_ms=timeout_ms)

    def apply_settings(self, settings: CameraSettings) -> None:
        if self._cam is None:
            return
        self._try_set_enum_string("ExposureAuto", "Continuous" if settings.ae_enabled else "Off")
        if not settings.ae_enabled:
            self._try_set_float("ExposureTime", float(settings.exposure_us))
        self._try_set_float("Gain", float(settings.analog_gain))
        self._try_set_float("Gamma", float(settings.gamma))
        self._try_set_float("Contrast", float(settings.contrast))
        self._try_set_bool("ReverseX", bool(settings.reverse_x))
        self._try_set_bool("ReverseY", bool(settings.reverse_y))

    def get_setting_ranges(self) -> CameraSettingRanges:
        return CameraSettingRanges(
            exposure_min_us=int(self._get_float_min("ExposureTime", 100)),
            exposure_max_us=int(self._get_float_max("ExposureTime", 1000000)),
            gamma_min=int(self._get_float_min("Gamma", 1)),
            gamma_max=int(self._get_float_max("Gamma", 500)),
            contrast_min=1,
            contrast_max=500,
            analog_gain_min=int(self._get_float_min("Gain", 0)),
            analog_gain_max=int(self._get_float_max("Gain", 100)),
        )

    def grab_preview_frame(self, timeout_ms: int = 200) -> np.ndarray | None:
        if self._cam is None or not self._grabbing:
            return None
        frame = self._grab_bgr_frame(timeout_ms=timeout_ms)
        if frame is not None:
            self.last_error = ""
            return self._preview_frame(frame)

        with self._grab_lock:
            frame_out = MV_FRAME_OUT()
            memset(byref(frame_out), 0, sizeof(frame_out))
            ret = self._cam.MV_CC_GetImageBuffer(frame_out, timeout_ms)
            if ret != MV_OK:
                self.last_error = f"preview grab failed: bgr/raw ret=0x{ret:x}"
                return None
            try:
                info = frame_out.stFrameInfo
                width = int(info.nWidth)
                height = int(info.nHeight)
                frame_len = int(info.nFrameLen)
                if width > 0 and height > 0 and frame_len == width * height:
                    raw_ptr = cast(frame_out.pBufAddr, POINTER(c_ubyte))
                    raw = np.ctypeslib.as_array(raw_ptr, shape=(frame_len,))
                    raw = raw.reshape((height, width))
                    self.last_error = ""
                    return self._preview_frame(raw)
            finally:
                self._cam.MV_CC_FreeImageBuffer(frame_out)
        self.last_error = "preview raw frame had unsupported packed format"
        return None

    def _grab_bgr_frame(self, timeout_ms: int = 200) -> np.ndarray | None:
        if self._cam is None or not self._grabbing:
            return None
        with self._grab_lock:
            width = self._width or self._get_int("Width", 0)
            height = self._height or self._get_int("Height", 0)
            if width <= 0 or height <= 0:
                raise RuntimeError("Camera width/height unavailable")
            output = self._ensure_bgr_buffer(int(width * height * 3))
            frame_info = MV_FRAME_OUT_INFO_EX()
            memset(byref(frame_info), 0, sizeof(frame_info))
            ret = self._cam.MV_CC_GetImageForBGR(output, self._bgr_buffer_size, frame_info, timeout_ms)
            if ret != MV_OK:
                self.last_error = f"bgr grab failed: 0x{ret:x}"
                return None
            frame_width = int(frame_info.nWidth) or width
            frame_height = int(frame_info.nHeight) or height
            byte_count = frame_width * frame_height * 3
            frame = np.frombuffer(output, dtype=np.uint8, count=byte_count)
            return frame.reshape((frame_height, frame_width, 3)).copy()

    def _preview_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        max_side = max(h, w)
        if self._preview_max_side <= 0 or max_side <= self._preview_max_side:
            return frame.copy()
        scale = self._preview_max_side / max_side
        out_w = max(1, int(w * scale))
        out_h = max(1, int(h * scale))
        if cv2 is not None:
            return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        y_idx = np.linspace(0, h - 1, out_h).astype(np.intp)
        x_idx = np.linspace(0, w - 1, out_w).astype(np.intp)
        return frame[np.ix_(y_idx, x_idx)].copy()

    def _ensure_bgr_buffer(self, size: int):
        if self._bgr_buffer is None or self._bgr_buffer_size < size:
            self._bgr_buffer = (c_ubyte * size)()
            self._bgr_buffer_size = size
        return self._bgr_buffer

    def _try_set_grab_strategy(self, strategy: int) -> bool:
        try:
            return self._cam is not None and self._cam.MV_CC_SetGrabStrategy(int(strategy)) == MV_OK
        except Exception:
            return False

    def _try_set_float(self, key: str, value: float) -> bool:
        try:
            return self._cam is not None and self._cam.MV_CC_SetFloatValue(key, float(value)) == MV_OK
        except Exception:
            return False

    def _try_set_int(self, key: str, value: int) -> bool:
        try:
            return self._cam is not None and self._cam.MV_CC_SetIntValue(key, int(value)) == MV_OK
        except Exception:
            return False

    def _try_set_enum(self, key: str, value: int) -> bool:
        try:
            return self._cam is not None and self._cam.MV_CC_SetEnumValue(key, int(value)) == MV_OK
        except Exception:
            return False

    def _try_set_enum_string(self, key: str, value: str) -> bool:
        try:
            return self._cam is not None and self._cam.MV_CC_SetEnumValueByString(key, value) == MV_OK
        except Exception:
            return False

    def _try_set_bool(self, key: str, value: bool) -> bool:
        try:
            return self._cam is not None and self._cam.MV_CC_SetBoolValue(key, bool(value)) == MV_OK
        except Exception:
            return False

    def _get_int(self, key: str, default: int) -> int:
        try:
            value = MVCC_INTVALUE_EX()
            memset(byref(value), 0, sizeof(value))
            if self._cam is not None and self._cam.MV_CC_GetIntValueEx(key, value) == MV_OK:
                return int(value.nCurValue)
        except Exception:
            pass
        return default

    def _get_float_min(self, key: str, default: float) -> float:
        try:
            value = MVCC_FLOATVALUE()
            memset(byref(value), 0, sizeof(value))
            if self._cam is not None and self._cam.MV_CC_GetFloatValue(key, value) == MV_OK:
                return float(value.fMin)
        except Exception:
            pass
        return default

    def _get_float_max(self, key: str, default: float) -> float:
        try:
            value = MVCC_FLOATVALUE()
            memset(byref(value), 0, sizeof(value))
            if self._cam is not None and self._cam.MV_CC_GetFloatValue(key, value) == MV_OK:
                return float(value.fMax)
        except Exception:
            pass
        return default


class HikvisionCameraAdapter(QObject):
    frame_ready = Signal(FrameSnapshot)
    status = Signal(str)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._camera: _MvsCamera | None = None
        self._live_thread: _LiveThread | None = None
        self._frame_id = 0
        self._latest_snapshot: FrameSnapshot | None = None
        self._settings = CameraSettings()
        self.device_count = 0
        self.device_name = ""
        self.last_status = "Hikvision idle"
        self.last_error = ""
        self.last_frame_shape: tuple[int, ...] | None = None
        self._last_frame_at = 0.0

    @property
    def hud_status(self) -> str:
        if self._latest_snapshot is not None and self.last_frame_shape is not None:
            height, width = self.last_frame_shape[:2]
            return f"HIK {width}x{height}"
        if self._live_thread is not None:
            return "HIK WAIT"
        if self.device_count == 0 and self.last_status:
            return "NO CAM"
        return "HIK"

    def start(self) -> bool:
        if not is_available():
            self.last_status = unavailable_reason()
            self.status.emit(self.last_status)
            return False
        try:
            self.stop()
            self._frame_id = 0
            self._latest_snapshot = None
            self.last_frame_shape = None
            self.device_count = 0
            self.device_name = ""
            self.last_error = ""
            self._camera = _MvsCamera()
            devices = self._camera.enumerate_devices()
            self.device_count = len(devices)
            if not devices:
                detail = self._camera.last_error or "SDK OK, 0 devices detected"
                self.last_status = f"No Hikvision camera detected ({detail}); using synthetic frames"
                self.status.emit(self.last_status)
                return False
            self.device_name = str(devices[0]["name"])
            self._camera.open(devices[0]["dev_info"])
            self._camera.apply_settings(self._settings)
            self._camera.start_live()
            self._live_thread = _LiveThread(self._camera)
            self._live_thread.frame_ready.connect(self._on_frame)
            self._live_thread.error.connect(self.error)
            self._live_thread.status.connect(self._on_live_status)
            self._live_thread.start()
            width, height = self._camera.resolution
            suffix = f" {width}x{height}" if width and height else ""
            self.last_status = f"Live camera: {self.device_name}{suffix}"
            self.status.emit(self.last_status)
            return True
        except Exception as exc:
            self.last_error = f"Hikvision start failed: {exc}"
            self.error.emit(self.last_error)
            self.stop()
            return False

    def stop(self) -> None:
        if self._live_thread is not None:
            self._live_thread.stop()
            self._live_thread = None
        if self._camera is not None:
            self._camera.close()
            self._camera = None

    def latest_snapshot(self) -> FrameSnapshot | None:
        return self._latest_snapshot

    def apply_settings(self, settings: CameraSettings) -> None:
        self._settings = settings
        if self._camera is not None:
            self._camera.apply_settings(settings)

    def set_property(self, name: str, value: float | bool | str) -> None:
        data = self._settings.__dict__.copy()
        if name in data:
            data[name] = value
            self.apply_settings(CameraSettings(**data))

    def _on_frame(self, frame: np.ndarray) -> None:
        self._frame_id += 1
        snapshot = FrameSnapshot(self._frame_id, time.monotonic(), frame)
        self._latest_snapshot = snapshot
        self.last_frame_shape = tuple(frame.shape)
        self._last_frame_at = snapshot.timestamp_s
        self.frame_ready.emit(snapshot)

    def _on_live_status(self, message: str) -> None:
        if self._camera is not None and self._camera.last_error:
            message = f"{message}; {self._camera.last_error}"
        self.last_status = message
        self.status.emit(message)
