from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from openfrp_vision.core.i18n import tr


DEFAULT_INDICATOR_SETTINGS: dict[str, Any] = {
    "blink_interval_s": 0.2,
    "blink_duration_s": 2.0,
    "geometry": [24, 72, 260, 190],
}


class ResultIndicatorWidget(QWidget):
    settings_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = dict(DEFAULT_INDICATOR_SETTINGS)
        self._drag_mode: str | None = None
        self._drag_start = QPoint()
        self._start_geometry = QRect()
        self._resize_margin = 10
        self._minimum_float_size = (210, 150)
        self._blink_color = QColor(0, 0, 0, 0)
        self._lit = False
        self._state_key = "state.off"
        self._state_text = tr(self._state_key)
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_lit)
        self._off_timer = QTimer(self)
        self._off_timer.setSingleShot(True)
        self._off_timer.timeout.connect(self.off)

        self.setObjectName("resultIndicator")
        self.setMinimumSize(*self._minimum_float_size)
        self.resize(260, 190)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(
            """
            QLabel {
                color: #e5edf4;
                font-weight: 700;
            }
            QLabel#sectionTitle {
                color: #94a3b8;
                font-size: 10px;
            }
            QDoubleSpinBox {
                background: rgba(255, 255, 255, 230);
                border: 1px solid rgba(15, 23, 42, 115);
                color: #0f172a;
                min-height: 22px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("sectionTitle")
        self.state_label = QLabel(self._state_text)
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.state_label)
        layout.addLayout(header)
        layout.addStretch(1)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        self.interval_spin = self._make_spin(0.05, 10.0, 0.05, " s")
        self.duration_spin = self._make_spin(0.1, 60.0, 0.1, " s")
        self.interval_spin.valueChanged.connect(self._controls_changed)
        self.duration_spin.valueChanged.connect(self._controls_changed)
        self.blink_label = QLabel()
        self.duration_label = QLabel()
        form.addRow(self.blink_label, self.interval_spin)
        form.addRow(self.duration_label, self.duration_spin)
        layout.addLayout(form)
        self.retranslate()

    def settings(self) -> dict[str, Any]:
        geometry = self.geometry()
        return {
            "blink_interval_s": float(self.interval_spin.value()),
            "blink_duration_s": float(self.duration_spin.value()),
            "geometry": [geometry.x(), geometry.y(), geometry.width(), geometry.height()],
        }

    def apply_settings(self, settings: dict[str, Any] | None) -> None:
        merged = {**DEFAULT_INDICATOR_SETTINGS, **(settings or {})}
        self._settings = merged
        blocked_interval = self.interval_spin.blockSignals(True)
        blocked_duration = self.duration_spin.blockSignals(True)
        self.interval_spin.setValue(float(merged.get("blink_interval_s", DEFAULT_INDICATOR_SETTINGS["blink_interval_s"])))
        self.duration_spin.setValue(float(merged.get("blink_duration_s", DEFAULT_INDICATOR_SETTINGS["blink_duration_s"])))
        self.interval_spin.blockSignals(blocked_interval)
        self.duration_spin.blockSignals(blocked_duration)

        geometry = merged.get("geometry", DEFAULT_INDICATOR_SETTINGS["geometry"])
        if isinstance(geometry, list | tuple) and len(geometry) == 4:
            self.setGeometry(self._bounded_geometry(QRect(*(int(value) for value in geometry))))

    def blink(self, passed: bool) -> None:
        self._blink_timer.stop()
        self._off_timer.stop()
        self._blink_color = QColor(34, 197, 94) if passed else QColor(239, 68, 68)
        self._state_key = "state.pass" if passed else "state.fail"
        self._state_text = tr(self._state_key)
        self._lit = True
        self.state_label.setText(self._state_text)
        self._blink_timer.start(max(50, round(float(self.interval_spin.value()) * 1000)))
        self._off_timer.start(max(100, round(float(self.duration_spin.value()) * 1000)))
        self.raise_()
        self.update()

    def off(self) -> None:
        self._blink_timer.stop()
        self._off_timer.stop()
        self._lit = False
        self._state_key = "state.off"
        self._state_text = tr(self._state_key)
        self.state_label.setText(self._state_text)
        self.update()

    def retranslate(self) -> None:
        self.title.setText(tr("indicator.title").upper())
        self.blink_label.setText(tr("indicator.blink"))
        self.duration_label.setText(tr("indicator.duration"))
        self._state_text = tr(self._state_key)
        self.state_label.setText(self._state_text)
        self.update()

    def clamp_to_parent(self) -> None:
        self.setGeometry(self._bounded_geometry(self.geometry()))

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self.settings_changed.emit(self.settings())

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(5, 5, -5, -5)
        painter.setBrush(QColor(15, 23, 42, 225))
        painter.setPen(QPen(QColor(226, 232, 240, 82), 1.0))
        painter.drawRoundedRect(rect, 8, 8)
        painter.setPen(QPen(QColor(255, 255, 255, 36), 1.0))
        painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 7, 7)

        diameter = max(56, min(self.width() - 76, self.height() - 98))
        cx = self.width() // 2
        cy = max(70, self.height() // 2 - 4)
        light_rect = QRect(cx - diameter // 2, cy - diameter // 2, diameter, diameter)
        color = self._blink_color if self._lit else QColor(30, 41, 59)
        glow = QColor(color)
        glow.setAlpha(75 if self._lit else 0)
        painter.setBrush(glow)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(light_rect.adjusted(-12, -12, 12, 12))
        painter.setBrush(color)
        painter.setPen(QPen(QColor(248, 250, 252, 145 if self._lit else 60), 1.2))
        painter.drawEllipse(light_rect)
        painter.setBrush(QColor(255, 255, 255, 90 if self._lit else 18))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(light_rect.adjusted(diameter // 5, diameter // 7, -diameter // 3, -diameter // 2))

        painter.setPen(QColor(226, 232, 240, 205))
        painter.setFont(QFont("Arial", 10, QFont.Weight.DemiBold))
        painter.drawText(light_rect.adjusted(0, diameter // 2 + 9, 0, diameter // 2 + 31), Qt.AlignmentFlag.AlignCenter, self._state_text)
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

    def _make_spin(self, minimum: float, maximum: float, step: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setSuffix(suffix)
        return spin

    def _controls_changed(self) -> None:
        self.settings_changed.emit(self.settings())

    def _toggle_lit(self) -> None:
        self._lit = not self._lit
        self.update()

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
        if position.y() <= 44 or self.childAt(position) in {None, self.title, self.state_label}:
            return "move"
        return None

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
