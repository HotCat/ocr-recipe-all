from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class RoiHandleWidget(QWidget):
    roi_changed = Signal(str, tuple)

    HANDLE = 9
    MARGIN = 18
    MIN_ROI_SIZE = 6

    def __init__(self, node_id: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.node_id = node_id
        self.title = node_id
        self.roi = (0, 0, 100, 100)
        self.frame_size = (1280, 720)
        self._drag_mode: str | None = None
        self._drag_start = QPoint()
        self._start_roi = self.roi
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_roi(self, title: str, roi: tuple[int, int, int, int], frame_size: tuple[int, int]) -> None:
        self.title = title
        self.roi = self._clamp_roi(roi)
        self.frame_size = (max(1, int(frame_size[0])), max(1, int(frame_size[1])))
        self._sync_geometry()
        self.update()

    def parent_resized(self) -> None:
        self._sync_geometry()
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self._inner_rect()

        outside = self.rect().adjusted(1, 1, -1, -1)
        painter.setBrush(QColor(0, 0, 0, 40))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(outside, 4, 4)

        painter.setBrush(QColor(45, 212, 191, 28))
        painter.setPen(QPen(QColor(34, 211, 238, 230), 2.0))
        painter.drawRect(rect)

        painter.setPen(QPen(QColor(3, 7, 18, 150), 4.0))
        painter.drawRect(rect.adjusted(1, 1, -1, -1))
        painter.setPen(QPen(QColor(248, 250, 252, 230), 1.2))
        painter.drawRect(rect.adjusted(1, 1, -1, -1))

        painter.setBrush(QColor(15, 23, 42, 235))
        painter.setPen(QPen(QColor(226, 232, 240, 190), 1.0))
        for point in self._handle_points(rect):
            painter.drawRect(QRect(point.x() - self.HANDLE // 2, point.y() - self.HANDLE // 2, self.HANDLE, self.HANDLE))

        label_rect = QRect(rect.left(), max(1, rect.top() - 20), min(max(90, rect.width()), self.width() - rect.left() - 2), 18)
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(label_rect, 3, 3)
        painter.setPen(QColor(226, 232, 240))
        painter.setFont(QFont("Arial", 8, QFont.Weight.DemiBold))
        painter.drawText(label_rect.adjusted(6, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.title)
        painter.end()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        mode = self._hit_test(event.position().toPoint())
        if mode is None:
            super().mousePressEvent(event)
            return
        self._drag_mode = mode
        self._drag_start = self.parentWidget().mapFromGlobal(event.globalPosition().toPoint())
        self._start_roi = self.roi
        self.raise_()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drag_mode is None:
            self._update_cursor(self._hit_test(event.position().toPoint()))
            super().mouseMoveEvent(event)
            return

        parent = self.parentWidget()
        current = parent.mapFromGlobal(event.globalPosition().toPoint())
        delta = current - self._drag_start
        frame_width, frame_height = self.frame_size
        parent_width = max(1, parent.width())
        parent_height = max(1, parent.height())
        dx = round(delta.x() * frame_width / parent_width)
        dy = round(delta.y() * frame_height / parent_height)
        x, y, width, height = self._start_roi
        mode = self._drag_mode
        if mode == "move":
            roi = (x + dx, y + dy, width, height)
        else:
            left = x
            top = y
            right = x + width
            bottom = y + height
            if "l" in mode:
                left += dx
            if "r" in mode:
                right += dx
            if "t" in mode:
                top += dy
            if "b" in mode:
                bottom += dy
            roi = (left, top, right - left, bottom - top)
        self.roi = self._clamp_roi(roi)
        self._sync_geometry()
        self.roi_changed.emit(self.node_id, self.roi)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drag_mode is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = None
            self._update_cursor(self._hit_test(event.position().toPoint()))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drag_mode is None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().leaveEvent(event)

    def _sync_geometry(self) -> None:
        parent = self.parentWidget()
        frame_width, frame_height = self.frame_size
        x, y, width, height = self.roi
        sx = parent.width() / max(1, frame_width)
        sy = parent.height() / max(1, frame_height)
        widget_rect = QRect(
            round(x * sx) - self.MARGIN,
            round(y * sy) - self.MARGIN,
            max(1, round(width * sx)) + self.MARGIN * 2,
            max(1, round(height * sy)) + self.MARGIN * 2,
        )
        self.setGeometry(widget_rect)

    def _inner_rect(self) -> QRect:
        return self.rect().adjusted(self.MARGIN, self.MARGIN, -self.MARGIN, -self.MARGIN)

    def _handle_points(self, rect: QRect) -> list[QPoint]:
        return [rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()]

    def _hit_test(self, position: QPoint) -> str | None:
        rect = self._inner_rect()
        hit = self.HANDLE + 4
        corners = {
            "tl": rect.topLeft(),
            "tr": rect.topRight(),
            "bl": rect.bottomLeft(),
            "br": rect.bottomRight(),
        }
        for name, point in corners.items():
            if abs(position.x() - point.x()) <= hit and abs(position.y() - point.y()) <= hit:
                return name
        if rect.contains(position):
            return "move"
        return None

    def _update_cursor(self, mode: str | None) -> None:
        cursor = {
            "tl": Qt.CursorShape.SizeFDiagCursor,
            "br": Qt.CursorShape.SizeFDiagCursor,
            "tr": Qt.CursorShape.SizeBDiagCursor,
            "bl": Qt.CursorShape.SizeBDiagCursor,
            "move": Qt.CursorShape.SizeAllCursor,
        }.get(mode, Qt.CursorShape.OpenHandCursor)
        self.setCursor(cursor)

    def _clamp_roi(self, roi: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        frame_width, frame_height = self.frame_size
        x, y, width, height = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
        width = max(self.MIN_ROI_SIZE, min(width, frame_width))
        height = max(self.MIN_ROI_SIZE, min(height, frame_height))
        x = max(0, min(x, frame_width - width))
        y = max(0, min(y, frame_height - height))
        return x, y, width, height
