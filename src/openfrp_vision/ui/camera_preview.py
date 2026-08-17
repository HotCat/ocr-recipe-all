from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from openfrp_vision.camera.base import FrameSnapshot


_USE_OPENGL_PREVIEW = os.environ.get("OPENFRP_USE_OPENGL_PREVIEW") == "1"
_PreviewBase = QOpenGLWidget if _USE_OPENGL_PREVIEW and os.environ.get("QT_QPA_PLATFORM") != "offscreen" else QWidget


class CameraGLView(_PreviewBase):
    """Live camera preview surface under the node overlay."""

    roi_selected = Signal(tuple)

    def __init__(self) -> None:
        super().__init__()
        self._frame: FrameSnapshot | None = None
        self._roi_enabled = False
        self._roi_origin: QPoint | None = None
        self._roi_rect = QRect()
        self.setMinimumSize(960, 540)
        self.setMouseTracking(True)

    def set_frame(self, frame: FrameSnapshot) -> None:
        self._frame = frame
        self.update()

    def set_roi_selection_enabled(self, enabled: bool) -> None:
        self._roi_enabled = enabled
        self._roi_origin = None
        self._roi_rect = QRect()
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    def paintGL(self) -> None:
        self._paint_preview()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        if isinstance(self, QOpenGLWidget):
            return
        self._paint_preview()

    def _paint_preview(self) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._frame is not None:
            qimage = self._to_qimage(self._frame.image_bgr)
            if not qimage.isNull():
                painter.drawImage(self.rect(), qimage)
        if self._roi_enabled and not self._roi_rect.isNull():
            painter.setBrush(QColor(25, 164, 189, 42))
            painter.setPen(QPen(QColor("#19a4bd"), 2))
            painter.drawRect(self._roi_rect.normalized())
        painter.end()

    @staticmethod
    def _to_qimage(image: np.ndarray) -> QImage:
        array = np.asarray(image)
        if array.size == 0:
            return QImage()

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        if array.ndim == 2:
            gray = np.ascontiguousarray(array)
            height, width = gray.shape
            return QImage(gray.data, width, height, gray.strides[0], QImage.Format.Format_Grayscale8).copy()

        if array.ndim != 3:
            return QImage()

        channels = array.shape[2]
        if channels == 1:
            gray = np.ascontiguousarray(array[:, :, 0])
            height, width = gray.shape
            return QImage(gray.data, width, height, gray.strides[0], QImage.Format.Format_Grayscale8).copy()

        if channels == 3:
            rgb = np.ascontiguousarray(array[:, :, [2, 1, 0]])
            height, width = rgb.shape[:2]
            return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()

        if channels == 4:
            rgba = np.ascontiguousarray(array[:, :, [2, 1, 0, 3]])
            height, width = rgba.shape[:2]
            return QImage(rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBA8888).copy()

        return QImage()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._roi_enabled and event.button() == Qt.MouseButton.LeftButton:
            self._roi_origin = event.position().toPoint()
            self._roi_rect = QRect(self._roi_origin, self._roi_origin)
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._roi_enabled and self._roi_origin is not None:
            self._roi_rect = QRect(self._roi_origin, event.position().toPoint()).normalized()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._roi_enabled and event.button() == Qt.MouseButton.LeftButton and self._roi_origin is not None:
            rect = QRect(self._roi_origin, event.position().toPoint()).normalized()
            self._roi_origin = None
            if rect.width() >= 4 and rect.height() >= 4 and self._frame is not None:
                self.roi_selected.emit(self._widget_rect_to_frame_roi(rect))
            self._roi_rect = QRect()
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _widget_rect_to_frame_roi(self, rect: QRect) -> tuple[int, int, int, int]:
        if self._frame is None:
            return 0, 0, 1, 1
        frame_height, frame_width = self._frame.image_bgr.shape[:2]
        bounded = rect.intersected(self.rect())
        x = round(bounded.x() * frame_width / max(1, self.width()))
        y = round(bounded.y() * frame_height / max(1, self.height()))
        width = round(bounded.width() * frame_width / max(1, self.width()))
        height = round(bounded.height() * frame_height / max(1, self.height()))
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        width = max(1, min(width, frame_width - x))
        height = max(1, min(height, frame_height - y))
        return x, y, width, height
