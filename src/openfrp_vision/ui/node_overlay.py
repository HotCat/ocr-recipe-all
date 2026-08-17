from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QMenu,
    QSpinBox,
    QLineEdit,
    QWidget,
)

from openfrp_vision.core.events import OverlayMode
from openfrp_vision.workflow.model import Edge, GraphError, PortType, RecipeGraph, RecipeNode

if TYPE_CHECKING:
    from PySide6.QtWidgets import QStyleOptionGraphicsItem, QWidget


TYPE_COLORS = {
    PortType.FRAME: QColor("#5d7286"),
    PortType.IMAGE: QColor("#2f7d8c"),
    PortType.CAMERA_SETTINGS: QColor("#526a9d"),
    PortType.TRIGGER: QColor("#7c3aed"),
    PortType.TEXT: QColor("#956522"),
    PortType.VERDICT: QColor("#9a4d45"),
    PortType.RESULT: QColor("#526a9d"),
}
FALLBACK_TYPE_COLOR = QColor("#64748b")

CATEGORY_COLORS = {
    "Input": QColor("#4d6c92"),
    "Camera": QColor("#526a9d"),
    "Image": QColor("#2f7d73"),
    "Recognition": QColor("#7a5f9a"),
    "Decision": QColor("#a06b31"),
    "Output": QColor("#8a4b4b"),
}


class CameraSettingsNodeWidget(QWidget):
    def __init__(self, node_item: "NodeItem") -> None:
        super().__init__()
        self._node_item = node_item
        self._block = False
        self.setStyleSheet(
            """
            QWidget { background: transparent; }
            QSpinBox {
                background: rgba(255, 255, 255, 210);
                color: #111827;
                border: 1px solid rgba(30, 41, 59, 90);
                padding: 1px;
                min-height: 18px;
            }
            QCheckBox { color: #111827; font-size: 9px; }
            """
        )
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.exposure = self._spin("exposure_us", 100, 1_000_000, 100, " us")
        self.gamma = self._spin("gamma", 1, 500, 1, "")
        self.contrast = self._spin("contrast", 1, 500, 1, "")
        self.gain = self._spin("analog_gain", 0, 100, 1, "")
        self.ae = self._check("ae_enabled", "Auto Exposure")
        self.reverse_x = self._check("reverse_x", "Mirror X")
        self.reverse_y = self._check("reverse_y", "Mirror Y")

        layout.addRow("Exposure", self.exposure)
        layout.addRow("Gamma", self.gamma)
        layout.addRow("Contrast", self.contrast)
        layout.addRow("Gain", self.gain)
        layout.addRow(self.ae)
        layout.addRow(self.reverse_x)
        layout.addRow(self.reverse_y)

    def _spin(self, key: str, minimum: int, maximum: int, step: int, suffix: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setValue(int(self._node_item.node.params.get(key, minimum)))
        spin.valueChanged.connect(lambda value, param_key=key: self._update_param(param_key, int(value)))
        return spin

    def _check(self, key: str, label: str) -> QCheckBox:
        check = QCheckBox(label)
        check.setChecked(bool(self._node_item.node.params.get(key, False)))
        check.toggled.connect(lambda value, param_key=key: self._update_param(param_key, bool(value)))
        return check

    def _update_param(self, key: str, value: int | bool) -> None:
        if self._block:
            return
        self._node_item.node.params[key] = value
        scene = self._node_item.scene()
        if isinstance(scene, GraphScene):
            scene.camera_settings_changed.emit(dict(self._node_item.node.params))
            scene.message.emit("Camera parameters changed")


class TriggerSwitchNodeWidget(QWidget):
    def __init__(self, node_item: "NodeItem") -> None:
        super().__init__()
        self._node_item = node_item
        self.setStyleSheet(
            """
            QWidget { background: transparent; }
            QComboBox, QLineEdit, QSpinBox {
                background: rgba(255, 255, 255, 210);
                color: #111827;
                border: 1px solid rgba(30, 41, 59, 90);
                padding: 1px;
                min-height: 18px;
            }
            QCheckBox { color: #111827; font-size: 9px; }
            """
        )
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.source = QComboBox()
        self.source.addItems(["keyboard", "external", "manual"])
        self.source.setCurrentText(str(node_item.node.params.get("source", "keyboard")))
        self.source.currentTextChanged.connect(lambda value: self._update_param("source", value))

        self.shortcut = QLineEdit(str(node_item.node.params.get("shortcut", "Ctrl+Return")))
        self.shortcut.editingFinished.connect(lambda: self._update_param("shortcut", self.shortcut.text()))

        self.topic = QLineEdit(str(node_item.node.params.get("external_topic", "")))
        self.topic.editingFinished.connect(lambda: self._update_param("external_topic", self.topic.text()))

        self.debounce = QSpinBox()
        self.debounce.setRange(0, 10000)
        self.debounce.setSuffix(" ms")
        self.debounce.setValue(int(node_item.node.params.get("debounce_ms", 250)))
        self.debounce.valueChanged.connect(lambda value: self._update_param("debounce_ms", int(value)))

        self.armed = QCheckBox("Armed")
        self.armed.setChecked(bool(node_item.node.params.get("armed", True)))
        self.armed.toggled.connect(lambda value: self._update_param("armed", bool(value)))

        layout.addRow("Source", self.source)
        layout.addRow("Shortcut", self.shortcut)
        layout.addRow("External", self.topic)
        layout.addRow("Debounce", self.debounce)
        layout.addRow(self.armed)

    def _update_param(self, key: str, value: str | int | bool) -> None:
        self._node_item.node.params[key] = value
        scene = self._node_item.scene()
        if isinstance(scene, GraphScene):
            scene.trigger_settings_changed.emit(dict(self._node_item.node.params))
            scene.message.emit("Trigger switch changed")


class ParamNodeWidget(QWidget):
    def __init__(self, node_item: "NodeItem") -> None:
        super().__init__()
        self._node_item = node_item
        self.setStyleSheet(
            """
            QWidget { background: transparent; }
            QSpinBox, QDoubleSpinBox, QLineEdit {
                background: rgba(255, 255, 255, 210);
                color: #111827;
                border: 1px solid rgba(30, 41, 59, 90);
                padding: 1px;
                min-height: 18px;
            }
            QCheckBox { color: #111827; font-size: 9px; }
            """
        )
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for key, value in node_item.node.params.items():
            if key == "snapshot":
                continue
            if isinstance(value, bool):
                check = QCheckBox(self._label(key))
                check.setChecked(value)
                check.toggled.connect(lambda checked, param_key=key: self._update_param(param_key, bool(checked)))
                layout.addRow(check)
            elif isinstance(value, int):
                spin = QSpinBox()
                minimum, maximum, step = self._int_range(key)
                spin.setRange(minimum, maximum)
                spin.setSingleStep(step)
                spin.setValue(value)
                spin.valueChanged.connect(lambda number, param_key=key: self._update_param(param_key, int(number)))
                layout.addRow(self._label(key), spin)
            elif isinstance(value, float):
                spin = QDoubleSpinBox()
                spin.setRange(0.0, 1.0 if "score" in key else 1000000.0)
                spin.setSingleStep(0.05 if "score" in key else 1.0)
                spin.setDecimals(3 if "score" in key else 2)
                spin.setValue(value)
                spin.valueChanged.connect(lambda number, param_key=key: self._update_param(param_key, float(number)))
                layout.addRow(self._label(key), spin)
            else:
                edit = QLineEdit(str(value))
                edit.editingFinished.connect(lambda widget=edit, param_key=key: self._update_param(param_key, widget.text()))
                layout.addRow(self._label(key), edit)

    def _update_param(self, key: str, value: str | int | float | bool) -> None:
        self._node_item.node.params[key] = value
        scene = self._node_item.scene()
        if isinstance(scene, GraphScene):
            scene.graph.results.clear()
            scene.graph_changed.emit()
            scene.message.emit(f"{self._node_item.node.title} changed")

    def _label(self, key: str) -> str:
        return key.replace("_", " ").title()

    def _int_range(self, key: str) -> tuple[int, int, int]:
        if key in {"x", "y", "width", "height"}:
            return 0, 100000, 1
        if "threshold" in key:
            return 0, 1000, 1
        if "fps" in key:
            return 1, 240, 1
        if "exposure" in key:
            return 1, 1000000, 100
        return -100000, 100000, 1


class PortItem(QGraphicsObject):
    RADIUS = 7.0

    def __init__(self, node_item: "NodeItem", port_name: str, data_type: PortType, is_output: bool) -> None:
        super().__init__(node_item)
        self.node_item = node_item
        self.port_name = port_name
        self.data_type = data_type
        self.is_output = is_output
        self.pending = False
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip(f"{port_name}: {data_type.value}")

    def boundingRect(self) -> QRectF:
        radius = self.RADIUS + 3
        return QRectF(-radius, -radius, radius * 2, radius * 2)

    def paint(
        self,
        painter: QPainter,
        option: "QStyleOptionGraphicsItem",
        widget: "QWidget | None" = None,
    ) -> None:
        del option, widget
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(TYPE_COLORS.get(self.data_type, FALLBACK_TYPE_COLOR))
        painter.setPen(QPen(QColor("#ffffff") if self.pending else QColor("#1f2933"), 3 if self.pending else 1.5))
        painter.drawEllipse(QPointF(0, 0), self.RADIUS, self.RADIUS)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        scene = self.scene()
        if isinstance(scene, GraphScene):
            scene.port_clicked(self)
        event.accept()

    def scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))


class NodeItem(QGraphicsObject):
    WIDTH = 210.0
    BASE_HEIGHT = 116.0

    def __init__(self, graph: RecipeGraph, node_id: str) -> None:
        super().__init__()
        self.graph = graph
        self.node_id = node_id
        self.node = graph.nodes[node_id]
        self.definition = graph.definitions[self.node.type_name]
        self.input_ports: dict[str, PortItem] = {}
        self.setPos(self.node.x, self.node.y)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

        title = QGraphicsSimpleTextItem(self.node.title, self)
        title.setBrush(QColor("#ffffff"))
        title.setFont(QFont("Arial", 10, QFont.Weight.DemiBold))
        title.setPos(14, 8)

        subtitle = QGraphicsSimpleTextItem(self.definition.title, self)
        subtitle.setBrush(QColor("#5b6570"))
        subtitle.setFont(QFont("Arial", 8))
        subtitle.setPos(14, 42)

        for index, port in enumerate(self.definition.inputs):
            item = PortItem(self, port.name, port.data_type, False)
            item.setPos(0, 70 + index * 22)
            self.input_ports[port.name] = item
            label = QGraphicsSimpleTextItem(port.name, self)
            label.setBrush(QColor("#334155"))
            label.setFont(QFont("Arial", 8))
            label.setPos(14, 63 + index * 22)

        self.output_port = PortItem(self, self.definition.output.name, self.definition.output.data_type, True)
        self.output_port.setPos(self.WIDTH, 70)
        output_label = QGraphicsSimpleTextItem(self.definition.output.name, self)
        output_label.setBrush(QColor("#334155"))
        output_label.setFont(QFont("Arial", 8))
        output_label.setPos(self.WIDTH - output_label.boundingRect().width() - 14, 63)

        self._proxy: QGraphicsProxyWidget | None = None
        if self.node.type_name == "camera_settings":
            self._proxy = QGraphicsProxyWidget(self)
            self._proxy.setWidget(CameraSettingsNodeWidget(self))
            self._proxy.setPos(14, 90)
        elif self.node.type_name == "trigger_switch":
            self._proxy = QGraphicsProxyWidget(self)
            self._proxy.setWidget(TriggerSwitchNodeWidget(self))
            self._proxy.setPos(14, 90)
        elif self.node.params:
            self._proxy = QGraphicsProxyWidget(self)
            self._proxy.setWidget(ParamNodeWidget(self))
            self._proxy.setPos(14, 90)

        self.summary_item = QGraphicsSimpleTextItem("Not run", self)
        self.summary_item.setBrush(QColor("#475569"))
        self.summary_item.setFont(QFont("Arial", 8))
        self.summary_item.setPos(14, self.height() - 23)

    def height(self) -> float:
        if self.node.type_name == "camera_settings":
            return 285.0
        if self.node.type_name == "trigger_switch":
            return 250.0
        if self.node.params:
            visible_params = [key for key in self.node.params if key != "snapshot"]
            return max(self.BASE_HEIGHT, 124 + len(visible_params) * 27)
        return max(self.BASE_HEIGHT, 96 + len(self.definition.inputs) * 22)

    def boundingRect(self) -> QRectF:
        return QRectF(-10, -3, self.WIDTH + 20, self.height() + 6)

    def paint(
        self,
        painter: QPainter,
        option: "QStyleOptionGraphicsItem",
        widget: "QWidget | None" = None,
    ) -> None:
        del option, widget
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = QRectF(0, 0, self.WIDTH, self.height())
        painter.setBrush(QColor(255, 255, 255, 224))
        painter.setPen(QPen(QColor("#19a4bd") if self.isSelected() else QColor(255, 255, 255, 150), 2.5 if self.isSelected() else 1.0))
        painter.drawRoundedRect(body, 6, 6)

        header = QPainterPath()
        header.addRoundedRect(QRectF(0, 0, self.WIDTH, 32), 6, 6)
        header.addRect(QRectF(0, 24, self.WIDTH, 8))
        painter.fillPath(header, CATEGORY_COLORS.get(self.definition.category, QColor("#56616b")))

        result = self.graph.results.get(self.node_id)
        if result is not None:
            color = QColor("#1f9d68")
            if result.summary.startswith("FAIL"):
                color = QColor("#c2413a")
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(self.WIDTH - 14, 16), 5, 5)

    def update_result(self) -> None:
        result = self.graph.results.get(self.node_id)
        if result is None:
            self.summary_item.setText("Not run")
        else:
            summary = result.summary.replace("\n", " ")
            self.summary_item.setText(f"{result.elapsed_ms:.1f} ms  {summary[:28]}")
        self.update()

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):  # type: ignore[no-untyped-def]
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.node.x = float(value.x())
            self.node.y = float(value.y())
            scene = self.scene()
            if isinstance(scene, GraphScene):
                scene.update_edges_for(self.node_id)
        return super().itemChange(change, value)


class EdgeItem(QGraphicsPathItem):
    def __init__(self, edge: Edge, source_port: PortItem, target_port: PortItem) -> None:
        super().__init__()
        self.edge = edge
        self.source_port = source_port
        self.target_port = target_port
        self.setZValue(-1)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.update_path()

    def update_path(self) -> None:
        start = self.source_port.scene_center()
        end = self.target_port.scene_center()
        distance = max(80.0, abs(end.x() - start.x()) * 0.48)
        path = QPainterPath(start)
        path.cubicTo(start.x() + distance, start.y(), end.x() - distance, end.y(), end.x(), end.y())
        self.setPath(path)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[no-untyped-def]
        del option, widget
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = TYPE_COLORS.get(self.source_port.data_type, FALLBACK_TYPE_COLOR)
        painter.setPen(QPen(QColor("#ef4444") if self.isSelected() else color, 4 if self.isSelected() else 2.4))
        painter.drawPath(self.path())

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(14)
        return stroker.createStroke(self.path())


class GraphScene(QGraphicsScene):
    message = Signal(str)
    graph_changed = Signal()
    camera_settings_changed = Signal(dict)
    trigger_settings_changed = Signal(dict)

    def __init__(self, graph: RecipeGraph) -> None:
        super().__init__()
        self.graph = graph
        self.node_items: dict[str, NodeItem] = {}
        self.edge_items: list[EdgeItem] = []
        self.pending_port: PortItem | None = None
        self.setSceneRect(-60, -60, 1560, 760)
        self.rebuild()

    def add_node_at(self, type_name: str, pos: QPointF) -> str:
        definition = self.graph.definitions[type_name]
        base = type_name.replace("_", "-")
        index = 1
        node_id = f"{base}-{index}"
        while node_id in self.graph.nodes:
            index += 1
            node_id = f"{base}-{index}"
        node = RecipeNode(node_id, type_name, definition.title, float(pos.x()), float(pos.y()))
        self.graph.add_node(node)
        self.graph.results.clear()
        self.rebuild()
        self.message.emit(f"Added {definition.title}")
        return node_id

    def rebuild(self) -> None:
        self.clear()
        self.node_items = {}
        self.edge_items = []
        self.pending_port = None
        for node_id in self.graph.nodes:
            node_item = NodeItem(self.graph, node_id)
            self.addItem(node_item)
            self.node_items[node_id] = node_item
        for edge in self.graph.edges:
            self._add_edge_item(edge)
        self.graph_changed.emit()

    def _add_edge_item(self, edge: Edge) -> None:
        source_item = self.node_items[edge.source]
        target_item = self.node_items[edge.target]
        edge_item = EdgeItem(edge, source_item.output_port, target_item.input_ports[edge.target_port])
        self.addItem(edge_item)
        self.edge_items.append(edge_item)

    def port_clicked(self, port: PortItem) -> None:
        if self.pending_port is None:
            self.pending_port = port
            port.pending = True
            port.update()
            self.message.emit(f"Connect {port.data_type.value} port")
            return

        first = self.pending_port
        first.pending = False
        first.update()
        self.pending_port = None
        if first is port:
            self.message.emit("Connection cancelled")
            return
        if first.is_output == port.is_output:
            self.message.emit("Choose one output and one input port")
            return
        source = first if first.is_output else port
        target = port if first.is_output else first
        try:
            self.graph.connect(source.node_item.node_id, target.node_item.node_id, target.port_name)
        except GraphError as exc:
            self.message.emit(str(exc))
            return
        self.rebuild()
        self.message.emit("Connection added")

    def update_edges_for(self, node_id: str) -> None:
        for edge_item in self.edge_items:
            if edge_item.edge.source == node_id or edge_item.edge.target == node_id:
                edge_item.update_path()

    def update_results(self) -> None:
        for item in self.node_items.values():
            item.update_result()

    def delete_selection(self) -> None:
        selected = list(self.selectedItems())
        edge_values = [item.edge for item in selected if isinstance(item, EdgeItem)]
        node_ids = [item.node_id for item in selected if isinstance(item, NodeItem)]
        for edge in edge_values:
            self.graph.disconnect(edge)
        for node_id in node_ids:
            self.graph.remove_node(node_id)
        if edge_values or node_ids:
            self.graph.results.clear()
            self.rebuild()
            self.message.emit("Selection deleted")

    def update_roi_node(self, node_id: str, roi: tuple[int, int, int, int]) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None or node.type_name != "roi":
            self.message.emit("Select or create an ROI node first")
            return
        x, y, width, height = roi
        node.params.update({"x": x, "y": y, "width": width, "height": height})
        self.graph.results.clear()
        self.graph.revision += 1
        self.rebuild()
        self.message.emit(f"{node.title}: x={x}, y={y}, {width} x {height}")


class NodeOverlayView(QGraphicsView):
    def __init__(self, graph: RecipeGraph) -> None:
        self.graph_scene = GraphScene(graph)
        super().__init__(self.graph_scene)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setStyleSheet("background: transparent")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.set_overlay_mode(OverlayMode.GRAPH)

    def add_node(self, type_name: str) -> str:
        center = self.mapToScene(self.viewport().rect().center())
        return self.graph_scene.add_node_at(type_name, center)

    def set_overlay_mode(self, mode: OverlayMode) -> None:
        self.setVisible(mode != OverlayMode.HIDDEN)
        self.setInteractive(mode in {OverlayMode.GRAPH, OverlayMode.EDIT})
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, mode in {OverlayMode.HIDDEN, OverlayMode.ROI})
        opacity = {
            OverlayMode.HIDDEN: 0.0,
            OverlayMode.ROI: 0.18,
            OverlayMode.GRAPH: 0.72,
            OverlayMode.EDIT: 1.0,
        }[mode]
        for item in self.graph_scene.items():
            item.setOpacity(opacity)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}:
            self.graph_scene.delete_selection()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        menu = QMenu(self)
        add_menu = menu.addMenu("Add Node")
        scene_pos = self.mapToScene(event.pos())
        for type_name, definition in sorted(self.graph_scene.graph.definitions.items(), key=lambda item: (item[1].category, item[1].title)):
            action = add_menu.addAction(f"{definition.category}: {definition.title}")
            action.triggered.connect(lambda _checked=False, node_type=type_name: self.graph_scene.add_node_at(node_type, scene_pos))
        menu.addAction("Delete Selection", self.graph_scene.delete_selection)
        menu.exec(event.globalPos())
