from __future__ import annotations

import json
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QFormLayout,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from openfrp_vision.ui.camera_preview import CameraGLView
from openfrp_vision.workflow.model import RecipeGraph


class NodeInspector(QWidget):
    parameter_changed = Signal(str, str, object)
    request_roi = Signal(str)

    def __init__(self, graph: RecipeGraph, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.graph = graph
        self.current_node_id: str | None = None
        self._parameter_widgets: dict[str, QWidget] = {}
        self._drag_mode: str | None = None
        self._drag_start = QPoint()
        self._start_geometry = QRect()
        self._resize_margin = 10
        self._minimum_float_size = (280, 360)
        self.setObjectName("nodeInspector")
        self.resize(340, 560)
        self.setMinimumSize(*self._minimum_float_size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setStyleSheet(
            """
            QWidget#nodeInspector {
                background: transparent;
                border: none;
                border-radius: 6px;
            }
            QLabel { color: #e5edf4; }
            QLabel#sectionTitle {
                color: #93a4b5;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#nodeTitle {
                color: #ffffff;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#nodeMeta {
                color: #b6c2cf;
                font-size: 11px;
            }
            QLabel#preview {
                background: rgba(2, 6, 12, 180);
                border: 1px solid rgba(148, 163, 184, 95);
                color: #94a3b8;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
                background: rgba(255, 255, 255, 225);
                border: 1px solid rgba(15, 23, 42, 110);
                color: #0f172a;
                selection-background-color: #1496aa;
            }
            QCheckBox {
                color: #e5edf4;
                spacing: 7px;
            }
            QPushButton {
                background: rgba(36, 51, 64, 220);
                border: 1px solid rgba(148, 163, 184, 100);
                border-radius: 4px;
                color: #f8fafc;
                padding: 5px 8px;
            }
            QFrame#rule {
                background: rgba(148, 163, 184, 80);
                max-height: 1px;
            }
            """
        )
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 155))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self.section_label = QLabel("INSPECTOR")
        self.section_label.setObjectName("sectionTitle")
        self.roi_button = QPushButton("Pick ROI")
        self.roi_button.setVisible(False)
        self.roi_button.clicked.connect(self._request_roi)
        header.addWidget(self.section_label)
        header.addStretch(1)
        header.addWidget(self.roi_button)
        layout.addLayout(header)

        self.node_title = QLabel("No node selected")
        self.node_title.setObjectName("nodeTitle")
        self.node_title.setWordWrap(True)
        self.node_meta = QLabel("")
        self.node_meta.setObjectName("nodeMeta")
        self.node_meta.setWordWrap(True)
        layout.addWidget(self.node_title)
        layout.addWidget(self.node_meta)

        rule = QFrame()
        rule.setObjectName("rule")
        layout.addWidget(rule)

        params_title = QLabel("PARAMETERS")
        params_title.setObjectName("sectionTitle")
        layout.addWidget(params_title)

        self.parameter_widget = QWidget()
        self.parameter_form = QFormLayout(self.parameter_widget)
        self.parameter_form.setContentsMargins(0, 0, 0, 0)
        self.parameter_form.setSpacing(6)
        self.parameter_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        parameter_scroll = QScrollArea()
        parameter_scroll.setWidgetResizable(True)
        parameter_scroll.setFrameShape(QFrame.Shape.NoFrame)
        parameter_scroll.setWidget(self.parameter_widget)
        layout.addWidget(parameter_scroll, 1)

        preview_title = QLabel("PREVIEW")
        preview_title.setObjectName("sectionTitle")
        layout.addWidget(preview_title)
        self.preview = QLabel("No preview")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(150)
        self.preview.setScaledContents(False)
        layout.addWidget(self.preview)

        output_title = QLabel("OUTPUT")
        output_title.setObjectName("sectionTitle")
        layout.addWidget(output_title)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumHeight(130)
        self.output.setPlaceholderText("Run workflow to inspect output")
        layout.addWidget(self.output)

    def set_graph(self, graph: RecipeGraph) -> None:
        self.graph = graph
        self.inspect(None)

    def inspect(self, node_id: str | None) -> None:
        self.current_node_id = node_id if node_id in self.graph.nodes else None
        self._clear_form()
        if self.current_node_id is None:
            self.node_title.setText("No node selected")
            self.node_meta.setText("Select a node to edit profile-scoped parameters.")
            self.roi_button.setVisible(False)
            self.preview.setPixmap(QPixmap())
            self.preview.setText("No preview")
            self.output.clear()
            return

        node = self.graph.nodes[self.current_node_id]
        definition = self.graph.definitions[node.type_name]
        self.node_title.setText(node.title)
        self.node_meta.setText(f"{definition.category} / {definition.title} / revision {self.graph.revision}")
        self.roi_button.setVisible(node.type_name == "roi")
        for key, value in node.params.items():
            if key == "snapshot":
                continue
            widget = self._parameter_widget(self.current_node_id, node.type_name, key, value)
            self._parameter_widgets[key] = widget
            self.parameter_form.addRow(self._label(key), widget)
        if not self._parameter_widgets:
            self.parameter_form.addRow(QLabel("No editable parameters"))
        self.refresh_result()

    def refresh_result(self) -> None:
        if self.current_node_id is None or self.current_node_id not in self.graph.nodes:
            return
        node = self.graph.nodes[self.current_node_id]
        result = self.graph.results.get(self.current_node_id)
        if result is None:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("No preview")
            self.output.clear()
            return

        if isinstance(result.preview, np.ndarray):
            qimage = CameraGLView._to_qimage(result.preview)
            if qimage.isNull():
                self.preview.setPixmap(QPixmap())
                self.preview.setText("No preview")
            else:
                pixmap = QPixmap.fromImage(qimage)
                self.preview.setText("")
                self.preview.setPixmap(
                    pixmap.scaled(
                        max(1, self.preview.width() - 8),
                        max(1, self.preview.height() - 8),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        else:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("No preview")

        value = result.value
        if isinstance(value, np.ndarray):
            text = self._image_summary(value)
        elif isinstance(value, (dict, list, tuple)):
            text = json.dumps(value, indent=2, default=str)
        else:
            text = str(value)
        summary = result.summary or node.title
        self.output.setPlainText(f"{summary}\n\n{text}" if text != summary else text)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self.refresh_result()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(6, 6, -6, -6)

        painter.setBrush(QColor(17, 24, 31, 235))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 8, 8)

        painter.setPen(QPen(QColor(226, 232, 240, 92), 1.0))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)
        painter.setPen(QPen(QColor(255, 255, 255, 42), 1.0))
        painter.drawRoundedRect(rect.adjusted(3, 3, -3, -3), 7, 7)

        accent = rect.adjusted(12, 10, -12, -rect.height() + 13)
        painter.setPen(QPen(QColor(56, 189, 248, 150), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(accent.left(), accent.top(), accent.right(), accent.top())
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
        if mode == "move":
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
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drag_mode is None:
            self.unsetCursor()
        super().leaveEvent(event)

    def clamp_to_parent(self) -> None:
        self.setGeometry(self._bounded_geometry(self.geometry()))

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
        if position.y() <= 88 or self.childAt(position) in {None, self.section_label, self.node_title, self.node_meta}:
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

    def _clear_form(self) -> None:
        while self.parameter_form.rowCount():
            self.parameter_form.removeRow(0)
        self._parameter_widgets.clear()

    def _parameter_widget(self, node_id: str, node_type: str, key: str, value: Any) -> QWidget:
        options = self._combo_options(node_type, key)
        if options:
            combo = QComboBox()
            combo.addItems(options)
            combo.setCurrentText(str(value))
            combo.currentTextChanged.connect(lambda text: self.parameter_changed.emit(node_id, key, text))
            return combo

        if isinstance(value, bool):
            check = QCheckBox()
            check.setChecked(value)
            check.toggled.connect(lambda checked: self.parameter_changed.emit(node_id, key, bool(checked)))
            return check

        if isinstance(value, int):
            spin = QSpinBox()
            minimum, maximum, step, suffix = self._int_range(node_type, key)
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setSuffix(suffix)
            spin.setValue(value)
            spin.valueChanged.connect(lambda number: self.parameter_changed.emit(node_id, key, int(number)))
            return spin

        if isinstance(value, float):
            spin = QDoubleSpinBox()
            minimum, maximum, step, decimals = self._float_range(key)
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setValue(value)
            spin.valueChanged.connect(lambda number: self.parameter_changed.emit(node_id, key, float(number)))
            return spin

        edit = QLineEdit(str(value))
        edit.editingFinished.connect(lambda: self.parameter_changed.emit(node_id, key, edit.text()))
        return edit

    def _request_roi(self) -> None:
        if self.current_node_id is not None:
            self.request_roi.emit(self.current_node_id)

    def _combo_options(self, node_type: str, key: str) -> list[str]:
        if node_type == "trigger_switch" and key == "source":
            return ["keyboard", "external", "manual"]
        if node_type == "ocr" and key == "lang":
            return ["en", "ch"]
        return []

    def _int_range(self, node_type: str, key: str) -> tuple[int, int, int, str]:
        if node_type == "camera_settings" and key == "exposure_us":
            return 100, 1_000_000, 100, " us"
        if node_type == "camera_settings" and key in {"gamma", "contrast"}:
            return 1, 500, 1, ""
        if node_type == "camera_settings" and key == "analog_gain":
            return 0, 100, 1, ""
        if key in {"x", "y", "width", "height"}:
            return 0, 100_000, 1, " px"
        if "threshold" in key:
            return 0, 1000, 1, ""
        if "fps" in key:
            return 1, 240, 1, " fps"
        if key.endswith("_ms"):
            return 0, 10_000, 10, " ms"
        return -1_000_000, 1_000_000, 1, ""

    def _float_range(self, key: str) -> tuple[float, float, float, int]:
        if "score" in key:
            return 0.0, 1.0, 0.05, 3
        return -1_000_000.0, 1_000_000.0, 1.0, 3

    def _label(self, key: str) -> str:
        return key.replace("_", " ").title()

    def _image_summary(self, image: np.ndarray) -> str:
        shape = " x ".join(str(part) for part in image.shape)
        return f"image {shape} {image.dtype}"
