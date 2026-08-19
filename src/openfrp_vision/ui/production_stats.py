from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from openfrp_vision.core.i18n import tr


DEFAULT_STATS_SETTINGS: dict[str, Any] = {
    "geometry": [36, 86, 420, 150],
}


class SevenSegmentDigits(QWidget):
    SEGMENTS = {
        "0": "abcfed",
        "1": "bc",
        "2": "abged",
        "3": "abgcd",
        "4": "fgbc",
        "5": "afgcd",
        "6": "afgecd",
        "7": "abc",
        "8": "abcdefg",
        "9": "abfgcd",
        "-": "g",
    }

    def __init__(self, color: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = "0000"
        self._color = color
        self.setMinimumSize(80, 42)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_value(self, value: int) -> None:
        text = f"{max(0, int(value)) % 10000:04d}"
        if text != self._text:
            self._text = text
            self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        text = self._text or "0"
        gap = max(3.0, self.width() * 0.012)
        digit_width = max(18.0, (self.width() - gap * (len(text) - 1)) / max(1, len(text)))
        digit_height = max(32.0, self.height() - 4)
        x = max(0.0, self.width() - (digit_width * len(text) + gap * (len(text) - 1)))
        first_lit = next((index for index, char in enumerate(text) if char != "0"), len(text))
        for index, char in enumerate(text):
            self._draw_digit(painter, QRectF(x, 2, digit_width, digit_height), char, index >= first_lit)
            x += digit_width + gap
        painter.end()

    def _draw_digit(self, painter: QPainter, rect: QRectF, char: str, prominent: bool) -> None:
        active = self.SEGMENTS.get(char, "")
        inactive = QColor(self._color)
        inactive.setAlpha(28 if prominent else 16)
        lit = QColor(self._color)
        lit.setAlpha(235 if prominent else 58)
        for segment in "abcdefg":
            painter.setBrush(lit if segment in active else inactive)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(self._segment_path(rect, segment))

    def _segment_path(self, rect: QRectF, segment: str) -> QPainterPath:
        thickness = max(5.0, min(rect.width(), rect.height()) * 0.16)
        bevel = thickness * 0.45
        left = rect.left() + thickness * 0.45
        right = rect.right() - thickness * 0.45
        top = rect.top()
        middle = rect.center().y()
        bottom = rect.bottom() - thickness
        if segment == "a":
            return self._horizontal(left, right, top, thickness, bevel)
        if segment == "g":
            return self._horizontal(left, right, middle - thickness / 2, thickness, bevel)
        if segment == "d":
            return self._horizontal(left, right, bottom, thickness, bevel)
        if segment == "f":
            return self._vertical(rect.left(), top + thickness * 0.55, middle - thickness * 0.3, thickness, bevel)
        if segment == "b":
            return self._vertical(rect.right() - thickness, top + thickness * 0.55, middle - thickness * 0.3, thickness, bevel)
        if segment == "e":
            return self._vertical(rect.left(), middle + thickness * 0.3, bottom + thickness * 0.05, thickness, bevel)
        return self._vertical(rect.right() - thickness, middle + thickness * 0.3, bottom + thickness * 0.05, thickness, bevel)

    def _horizontal(self, left: float, right: float, top: float, thickness: float, bevel: float) -> QPainterPath:
        path = QPainterPath()
        path.moveTo(left + bevel, top)
        path.lineTo(right - bevel, top)
        path.lineTo(right, top + thickness / 2)
        path.lineTo(right - bevel, top + thickness)
        path.lineTo(left + bevel, top + thickness)
        path.lineTo(left, top + thickness / 2)
        path.closeSubpath()
        return path

    def _vertical(self, left: float, top: float, bottom: float, thickness: float, bevel: float) -> QPainterPath:
        path = QPainterPath()
        path.moveTo(left + thickness / 2, top)
        path.lineTo(left + thickness, top + bevel)
        path.lineTo(left + thickness, bottom - bevel)
        path.lineTo(left + thickness / 2, bottom)
        path.lineTo(left, bottom - bevel)
        path.lineTo(left, top + bevel)
        path.closeSubpath()
        return path


class ProductionStatsWidget(QWidget):
    settings_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._drag_mode: str | None = None
        self._drag_start = QPoint()
        self._start_geometry = QRect()
        self._resize_margin = 10
        self._minimum_float_size = (360, 110)
        self._ok = 0
        self._ng = 0
        self._fps = 0.0
        self._ok_label_rect = QRect()
        self._ng_label_rect = QRect()
        self.setObjectName("productionStats")
        self.setMinimumSize(*self._minimum_float_size)
        self.resize(420, 150)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.ok_digits = SevenSegmentDigits(QColor(39, 230, 54), self)
        self.ng_digits = SevenSegmentDigits(QColor(255, 34, 64), self)

    def settings(self) -> dict[str, Any]:
        geometry = self.geometry()
        return {"geometry": [geometry.x(), geometry.y(), geometry.width(), geometry.height()]}

    def apply_settings(self, settings: dict[str, Any] | None) -> None:
        merged = {**DEFAULT_STATS_SETTINGS, **(settings or {})}
        geometry = merged.get("geometry", DEFAULT_STATS_SETTINGS["geometry"])
        if isinstance(geometry, list | tuple) and len(geometry) == 4:
            self.setGeometry(self._bounded_geometry(QRect(*(int(value) for value in geometry))))

    def set_counts(self, ok: int, ng: int) -> None:
        self._ok = max(0, int(ok))
        self._ng = max(0, int(ng))
        self.ok_digits.set_value(self._ok)
        self.ng_digits.set_value(self._ng)
        self.update()

    def set_fps(self, fps: float) -> None:
        self._fps = max(0.0, float(fps))
        self.update()

    def retranslate(self) -> None:
        self.update()

    def clamp_to_parent(self) -> None:
        self.setGeometry(self._bounded_geometry(self.geometry()))

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        margin = max(14, round(self.width() * 0.04))
        group_gap = max(36, round(self.width() * 0.1))
        label_gap = max(6, round(self.width() * 0.015))
        label_width = max(34, round(self.width() * 0.095))
        top = max(22, round(self.height() * 0.16))
        digit_height = max(46, round(self.height() * 0.45))
        group_width = max(120, (self.width() - margin * 2 - group_gap) // 2)
        digit_width = max(84, group_width - label_width - label_gap)
        ok_left = margin
        ng_left = margin + group_width + group_gap
        self.ok_digits.setGeometry(ok_left, top, digit_width, digit_height)
        self.ng_digits.setGeometry(ng_left, top, digit_width, digit_height)
        self._ok_label_rect = QRect(ok_left + digit_width + label_gap, top, label_width, digit_height)
        self._ng_label_rect = QRect(ng_left + digit_width + label_gap, top, label_width, digit_height)
        self.settings_changed.emit(self.settings())

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.setBrush(QColor(2, 6, 12, 64))
        painter.setPen(QPen(QColor(226, 232, 240, 55), 1.0))
        painter.drawRoundedRect(rect, 8, 8)

        painter.setPen(QColor(125, 211, 252, 210))
        label_y = self.ok_digits.y() + max(18, self.ok_digits.height() // 3)
        painter.drawText(self._ok_label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, tr("stats.ok"))
        painter.drawText(self._ng_label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, tr("stats.ng"))
        painter.setPen(QColor(43, 255, 79, 230))
        painter.drawText(16, max(label_y + self.ok_digits.height() // 2, self.height() - 26), f"{tr('stats.fps')}: {self._fps:05.2f}")
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
        self._drag_start = event.globalPosition().toPoint()
        self._start_geometry = self.geometry()
        self.raise_()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        position = event.position().toPoint()
        if self._drag_mode is None:
            self._update_cursor(self._hit_test(position))
            super().mouseMoveEvent(event)
            return

        delta = event.globalPosition().toPoint() - self._drag_start
        geometry = QRect(self._start_geometry)
        mode = self._drag_mode
        if mode == "move":
            geometry.moveTopLeft(self._start_geometry.topLeft() + delta)
        else:
            if "l" in mode:
                geometry.setLeft(geometry.left() + delta.x())
            if "r" in mode:
                geometry.setRight(geometry.right() + delta.x())
            if "t" in mode:
                geometry.setTop(geometry.top() + delta.y())
            if "b" in mode:
                geometry.setBottom(geometry.bottom() + delta.y())
        self.setGeometry(self._bounded_geometry(geometry))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drag_mode is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = None
            self._update_cursor(self._hit_test(event.position().toPoint()))
            self.settings_changed.emit(self.settings())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drag_mode is None:
            self.unsetCursor()
        super().leaveEvent(event)

    def _hit_test(self, position: QPoint) -> str | None:
        margin = self._resize_margin
        left = position.x() <= margin
        right = position.x() >= self.width() - margin
        top = position.y() <= margin
        bottom = position.y() >= self.height() - margin
        if top and left:
            return "tl"
        if top and right:
            return "tr"
        if bottom and left:
            return "bl"
        if bottom and right:
            return "br"
        if left:
            return "l"
        if right:
            return "r"
        if top:
            return "t"
        if bottom:
            return "b"
        return "move" if self.rect().contains(position) else None

    def _update_cursor(self, mode: str | None) -> None:
        cursor = {
            "tl": Qt.CursorShape.SizeFDiagCursor,
            "br": Qt.CursorShape.SizeFDiagCursor,
            "tr": Qt.CursorShape.SizeBDiagCursor,
            "bl": Qt.CursorShape.SizeBDiagCursor,
            "l": Qt.CursorShape.SizeHorCursor,
            "r": Qt.CursorShape.SizeHorCursor,
            "t": Qt.CursorShape.SizeVerCursor,
            "b": Qt.CursorShape.SizeVerCursor,
            "move": Qt.CursorShape.OpenHandCursor,
        }.get(mode)
        if cursor is None:
            self.unsetCursor()
        else:
            self.setCursor(cursor)

    def _bounded_geometry(self, geometry: QRect) -> QRect:
        parent = self.parentWidget()
        bounds = parent.rect() if parent is not None else QRect(0, 0, 4096, 2160)
        min_width, min_height = self._minimum_float_size
        width = max(min_width, min(geometry.width(), max(min_width, bounds.width() - 24)))
        height = max(min_height, min(geometry.height(), max(min_height, bounds.height() - 64)))
        x = max(12, min(geometry.x(), max(12, bounds.width() - width - 12)))
        y = max(54, min(geometry.y(), max(54, bounds.height() - height - 12)))
        return QRect(x, y, width, height)
