from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QVBoxLayout,
)

from openfrp_vision.core.events import OverlayMode
from openfrp_vision.core.i18n import category_label, node_label, node_type_label, port_label, tr
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


class WorkflowGroupItem(QGraphicsRectItem):
    def __init__(self) -> None:
        super().__init__()
        self.setZValue(-20)
        self.setPen(QPen(QColor(255, 255, 255, 70), 1.2, Qt.PenStyle.DashLine))
        self.setBrush(QColor(8, 13, 20, 36))

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[no-untyped-def]
        del option, widget
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawRoundedRect(rect, 8, 8)
        painter.setPen(QColor(225, 235, 245, 105))
        painter.setFont(QFont("Arial", 9, QFont.Weight.DemiBold))
        painter.drawText(rect.adjusted(14, 10, -14, -10), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, tr("graph.group"))


class NodeGroupItem(QGraphicsRectItem):
    def __init__(self, group_id: str, title: str) -> None:
        super().__init__()
        self.group_id = group_id
        self.title = title
        self.member_count = 0
        self.member_preview = ""
        self.setZValue(-10)
        self.setPen(QPen(QColor(96, 165, 250, 150), 1.4, Qt.PenStyle.DashLine))
        self.setBrush(QColor(14, 23, 37, 58))
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def set_title(self, title: str) -> None:
        self.title = title
        self.update()

    def set_member_count(self, count: int) -> None:
        self.member_count = count
        self.update()

    def set_member_preview(self, preview: str) -> None:
        self.member_preview = preview
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[no-untyped-def]
        del option, widget
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawRoundedRect(rect, 8, 8)

        header = QRectF(rect.left() + 8, rect.top() + 8, min(rect.width() - 16, 320), 42)
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(QColor(8, 13, 20, 175))
        painter.drawRoundedRect(header, 6, 6)

        painter.setPen(QColor("#e2e8f0"))
        painter.setFont(QFont("Arial", 9, QFont.Weight.DemiBold))
        title_rect = header.adjusted(10, 5, -10, -18)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.title)

        painter.setPen(QColor("#93c5fd"))
        painter.setFont(QFont("Arial", 8))
        subtitle = f"{self.member_count} node(s)"
        if self.member_preview:
            subtitle = f"{subtitle}  {self.member_preview}"
        count_rect = header.adjusted(10, 20, -10, -4)
        painter.drawText(count_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, subtitle)


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
        self.setToolTip(f"{port_label(port_name)}: {data_type.value}")

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

        title = QGraphicsSimpleTextItem(node_label(self.node, self.definition.title), self)
        title.setBrush(QColor("#ffffff"))
        title.setFont(QFont("Arial", 10, QFont.Weight.DemiBold))
        title.setPos(14, 8)

        subtitle = QGraphicsSimpleTextItem(node_type_label(self.node.type_name, self.definition.title), self)
        subtitle.setBrush(QColor("#5b6570"))
        subtitle.setFont(QFont("Arial", 8))
        subtitle.setPos(14, 42)

        for index, port in enumerate(self.definition.inputs):
            item = PortItem(self, port.name, port.data_type, False)
            item.setPos(0, 70 + index * 22)
            self.input_ports[port.name] = item
            label = QGraphicsSimpleTextItem(port_label(port.name), self)
            label.setBrush(QColor("#334155"))
            label.setFont(QFont("Arial", 8))
            label.setPos(14, 63 + index * 22)

        self.output_port = PortItem(self, self.definition.output.name, self.definition.output.data_type, True)
        self.output_port.setPos(self.WIDTH, 70)
        output_label = QGraphicsSimpleTextItem(port_label(self.definition.output.name), self)
        output_label.setBrush(QColor("#334155"))
        output_label.setFont(QFont("Arial", 8))
        output_label.setPos(self.WIDTH - output_label.boundingRect().width() - 14, 63)

        self.summary_item = QGraphicsSimpleTextItem(tr("graph.not_run"), self)
        self.summary_item.setBrush(QColor("#475569"))
        self.summary_item.setFont(QFont("Arial", 8))
        self.summary_item.setPos(14, self.height() - 23)
        self.update_result()

    def height(self) -> float:
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
        painter.setBrush(QColor(255, 255, 255, 224) if self.node.enabled else QColor(148, 163, 184, 176))
        border = QColor("#19a4bd") if self.isSelected() else QColor(255, 255, 255, 150)
        if not self.node.enabled and not self.isSelected():
            border = QColor(100, 116, 139, 170)
        painter.setPen(QPen(border, 2.5 if self.isSelected() else 1.0))
        painter.drawRoundedRect(body, 6, 6)

        header = QPainterPath()
        header.addRoundedRect(QRectF(0, 0, self.WIDTH, 32), 6, 6)
        header.addRect(QRectF(0, 24, self.WIDTH, 8))
        header_color = CATEGORY_COLORS.get(self.definition.category, QColor("#56616b"))
        if not self.node.enabled:
            header_color = QColor("#64748b")
        painter.fillPath(header, header_color)

        if not self.node.enabled:
            painter.setPen(QPen(QColor(15, 23, 42, 165), 1.0))
            painter.drawLine(QPointF(14, self.height() - 36), QPointF(self.WIDTH - 14, 44))

        result = self.graph.results.get(self.node_id)
        if result is not None:
            color = QColor("#1f9d68")
            if result.summary.startswith("FAIL"):
                color = QColor("#c2413a")
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(self.WIDTH - 14, 16), 5, 5)

    def update_result(self) -> None:
        if not self.node.enabled:
            self.summary_item.setText(tr("graph.disabled"))
            self.summary_item.setBrush(QColor("#334155"))
            self.update()
            return
        result = self.graph.results.get(self.node_id)
        if result is None:
            self.summary_item.setText(tr("graph.not_run"))
            self.summary_item.setBrush(QColor("#475569"))
        else:
            summary = result.summary.replace("\n", " ")
            self.summary_item.setBrush(QColor("#475569") if not result.skipped else QColor("#8a5a17"))
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
        edge_style = getattr(self.source_port.node_item.graph, "edge_style", "curved")
        path = QPainterPath(start)
        if edge_style == "segmented":
            delta_x = end.x() - start.x()
            offset = max(42.0, abs(delta_x) * 0.45)
            elbow_x = start.x() + offset if delta_x >= 0 else start.x() - offset
            path.lineTo(elbow_x, start.y())
            path.lineTo(elbow_x, end.y())
            path.lineTo(end)
        else:
            distance = max(80.0, abs(end.x() - start.x()) * 0.48)
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
        self.group_items: dict[str, NodeGroupItem] = {}
        self.pending_port: PortItem | None = None
        self.group_item = WorkflowGroupItem()
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
        self.message.emit(tr("graph.added", node=node_type_label(type_name, definition.title)))
        return node_id

    def rebuild(self) -> None:
        self.clear()
        self.node_items = {}
        self.edge_items = []
        self.group_items = {}
        self.pending_port = None
        self.group_item = WorkflowGroupItem()
        self.addItem(self.group_item)
        for group_id, group in self.graph.groups.items():
            group_item = NodeGroupItem(group_id, group.title)
            self.addItem(group_item)
            self.group_items[group_id] = group_item
        for node_id in self.graph.nodes:
            node_item = NodeItem(self.graph, node_id)
            self.addItem(node_item)
            self.node_items[node_id] = node_item
        for edge in self.graph.edges:
            self._add_edge_item(edge)
        self.update_group_bounds()
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
            self.message.emit(tr("graph.connect_port", port=port.data_type.value))
            return

        first = self.pending_port
        first.pending = False
        first.update()
        self.pending_port = None
        if first is port:
            self.message.emit(tr("graph.connection_cancelled"))
            return
        if first.is_output == port.is_output:
            self.message.emit(tr("graph.choose_ports"))
            return
        source = first if first.is_output else port
        target = port if first.is_output else first
        try:
            self.graph.connect(source.node_item.node_id, target.node_item.node_id, target.port_name)
        except GraphError as exc:
            self.message.emit(str(exc))
            return
        self.rebuild()
        self.message.emit(tr("graph.connection_added"))

    def update_edges_for(self, node_id: str) -> None:
        for edge_item in self.edge_items:
            if edge_item.edge.source == node_id or edge_item.edge.target == node_id:
                edge_item.update_path()
        self.update_group_bounds()

    def workflow_rect(self) -> QRectF:
        if not self.node_items:
            return QRectF(-300, -220, 600, 440)
        rect = QRectF()
        for item in self.node_items.values():
            item_rect = item.mapRectToScene(item.boundingRect())
            rect = item_rect if rect.isNull() else rect.united(item_rect)
        return rect

    def update_group_bounds(self) -> None:
        rect = self.workflow_rect().adjusted(-70, -70, 70, 90)
        self.group_item.setRect(rect)
        for group_id, group in self.graph.groups.items():
            group_item = self.group_items.get(group_id)
            if group_item is None:
                continue
            nodes = [self.node_items.get(node_id) for node_id in group.node_ids]
            nodes = [item for item in nodes if item is not None]
            if len(nodes) < 2:
                group_item.setVisible(False)
                continue
            bounds = QRectF()
            preview_titles: list[str] = []
            for item in nodes:
                item_rect = item.mapRectToScene(item.boundingRect())
                bounds = item_rect if bounds.isNull() else bounds.united(item_rect)
                if len(preview_titles) < 3:
                    preview_titles.append(node_label(item.node, self.graph.definitions[item.node.type_name].title))
            bounds = bounds.adjusted(-16, -20, 16, 16)
            group_item.setRect(bounds)
            group_item.setVisible(True)
            group_item.set_member_count(len(nodes))
            group_item.set_title(group.title)
            group_item.set_member_preview(" • ".join(preview_titles))
        self.setSceneRect(rect.adjusted(-900, -650, 900, 650))

    def update_results(self) -> None:
        for item in self.node_items.values():
            item.update_result()

    def set_edge_style(self, style: str) -> None:
        self.graph.set_edge_style(style)
        self.rebuild()
        label = tr("menu.curved_connections") if style == "curved" else tr("menu.segmented_connections")
        self.message.emit(tr("graph.edge_style_changed", style=label))

    def group_selected_nodes(self, title: str) -> None:
        node_ids = [item.node_id for item in self.selectedItems() if isinstance(item, NodeItem)]
        self.group_nodes(node_ids, title)

    def group_nodes(self, node_ids: list[str], title: str) -> None:
        try:
            self.graph.group_nodes(node_ids, title)
        except GraphError as exc:
            self.message.emit(str(exc))
            return
        self.graph.results.clear()
        self.rebuild()
        self.message.emit(tr("graph.group_created"))

    def ungroup_selected_nodes(self) -> None:
        node_ids = [item.node_id for item in self.selectedItems() if isinstance(item, NodeItem)]
        removed = self.graph.ungroup_nodes(node_ids)
        if removed:
            self.graph.results.clear()
            self.rebuild()
            self.message.emit(tr("graph.group_removed"))

    def auto_arrange(self) -> bool:
        if not self.graph.nodes:
            return False
        incoming: dict[str, list[str]] = {node_id: [] for node_id in self.graph.nodes}
        for edge in self.graph.edges:
            incoming.setdefault(edge.target, []).append(edge.source)
        try:
            topo = self.graph.topological_order()
        except GraphError as exc:
            self.message.emit(str(exc))
            return False

        levels: dict[str, int] = {}
        for node_id in topo:
            parents = incoming.get(node_id, [])
            if parents:
                levels[node_id] = max(levels.get(parent, 0) + 1 for parent in parents)
            else:
                levels[node_id] = 0

        grouped: dict[int, list[str]] = defaultdict(list)
        for node_id in topo:
            grouped[levels[node_id]].append(node_id)

        category_order = {"Input": 0, "Camera": 1, "Image": 2, "Recognition": 3, "Decision": 4, "Output": 5}
        x_gap = 300.0
        y_gap = 170.0
        origin_x = 70.0
        origin_y = 70.0
        moved = False
        for level in sorted(grouped):
            node_ids = grouped[level]
            node_ids.sort(
                key=lambda node_id: (
                    category_order.get(self.graph.definitions[self.graph.nodes[node_id].type_name].category, 99),
                    self.graph.nodes[node_id].title.lower(),
                    node_id,
                )
            )
            for row, node_id in enumerate(node_ids):
                node = self.graph.nodes[node_id]
                x = origin_x + level * x_gap
                y = origin_y + row * y_gap
                if node.x != x or node.y != y:
                    moved = True
                node.x = x
                node.y = y
                item = self.node_items.get(node_id)
                if item is not None:
                    item.setPos(x, y)

        self.update_group_bounds()
        if moved:
            self.graph.revision += 1
            self.graph_changed.emit()
        self.message.emit(tr("graph.auto_arranged"))
        return moved

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
            self.message.emit(tr("graph.selection_deleted"))

    def set_selection_enabled(self, enabled: bool) -> None:
        node_ids = [item.node_id for item in self.selectedItems() if isinstance(item, NodeItem)]
        if not node_ids:
            return
        for node_id in node_ids:
            self.graph.set_node_enabled(node_id, enabled)
        self.update_results()
        for node_id in node_ids:
            item = self.node_items.get(node_id)
            if item is not None:
                item.update()
        state = tr("state.enabled") if enabled else tr("state.disabled")
        self.graph_changed.emit()
        self.message.emit(tr("graph.nodes_enabled", count=len(node_ids), state=state))

    def toggle_selection_enabled(self) -> None:
        node_ids = [item.node_id for item in self.selectedItems() if isinstance(item, NodeItem)]
        if not node_ids:
            return
        should_enable = any(not self.graph.nodes[node_id].enabled for node_id in node_ids)
        self.set_selection_enabled(should_enable)

    def update_roi_node(self, node_id: str, roi: tuple[int, int, int, int]) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None or node.type_name != "roi":
            self.message.emit(tr("graph.select_roi_first"))
            return
        x, y, width, height = roi
        node.params.update({"x": x, "y": y, "width": width, "height": height})
        self.graph.results.clear()
        self.graph.revision += 1
        self.rebuild()
        self.message.emit(tr("status.roi_changed", node=node_label(node, self.graph.definitions[node.type_name].title), x=x, y=y, width=width, height=height))


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
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._zoom = 1.0
        self._min_zoom = 0.25
        self._max_zoom = 2.8
        self._panning = False
        self._pan_start = QPoint()
        self._space_pan = False
        self.set_overlay_mode(OverlayMode.GRAPH)

    def set_graph(self, graph: RecipeGraph) -> None:
        self.graph_scene.graph = graph
        self.graph_scene.rebuild()
        self.fit_to_nodes()

    def retranslate(self) -> None:
        self.graph_scene.rebuild()

    def add_node(self, type_name: str) -> str:
        center = self.mapToScene(self.viewport().rect().center())
        return self.graph_scene.add_node_at(type_name, center)

    def zoom_in(self) -> None:
        self._scale_by(1.18)

    def zoom_out(self) -> None:
        self._scale_by(1 / 1.18)

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._zoom = 1.0

    def fit_to_nodes(self) -> None:
        rect = self.graph_scene.workflow_rect().adjusted(-120, -120, 120, 120)
        if rect.isEmpty():
            return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = max(self._min_zoom, min(self._max_zoom, float(self.transform().m11())))

    def auto_arrange(self) -> bool:
        moved = self.graph_scene.auto_arrange()
        if moved:
            self.fit_to_nodes()
        return moved

    def _group_selected_nodes(self) -> None:
        node_ids = [item.node_id for item in self.graph_scene.selectedItems() if isinstance(item, NodeItem)]
        if len(node_ids) < 2:
            return
        dialog = QDialog(self)
        dialog.setObjectName("groupDialog")
        dialog.setWindowTitle(tr("dialog.group.title"))
        dialog.setModal(True)
        dialog.setStyleSheet(
            """
            QDialog#groupDialog {
                background: #111827;
            }
            QLabel {
                color: #e5edf4;
                font-weight: 600;
            }
            QLineEdit {
                background: #0f172a;
                color: #f8fafc;
                border: 1px solid #f97316;
                border-radius: 4px;
                padding: 6px 8px;
                selection-background-color: #f97316;
            }
            QPushButton {
                background: #0f172a;
                color: #f8fafc;
                border: 1px solid #f97316;
                border-radius: 4px;
                padding: 5px 12px;
                min-width: 72px;
            }
            QPushButton:hover {
                background: #1f2937;
            }
            """
        )
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(QLabel(tr("dialog.group.label")))
        edit = QLineEdit(tr("dialog.group.default"))
        layout.addWidget(edit)
        buttons = QDialogButtonBox()
        cancel_button = QPushButton(tr("dialog.cancel"))
        ok_button = QPushButton(tr("dialog.ok"))
        buttons.addButton(cancel_button, QDialogButtonBox.ButtonRole.RejectRole)
        buttons.addButton(ok_button, QDialogButtonBox.ButtonRole.AcceptRole)
        cancel_button.clicked.connect(dialog.reject)
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(buttons)
        edit.selectAll()
        edit.setFocus()
        if not dialog.exec():
            return
        self.graph_scene.group_nodes(node_ids, edit.text().strip() or tr("dialog.group.default"))

    def _ungroup_selected_nodes(self) -> None:
        node_ids = [item.node_id for item in self.graph_scene.selectedItems() if isinstance(item, NodeItem)]
        if not node_ids:
            return
        removed = self.graph_scene.graph.ungroup_nodes(node_ids)
        if removed:
            self.graph_scene.graph.results.clear()
            self.graph_scene.rebuild()
            self.graph_scene.message.emit(tr("graph.group_removed"))

    def _scale_by(self, factor: float) -> None:
        target = max(self._min_zoom, min(self._max_zoom, self._zoom * factor))
        if target == self._zoom:
            return
        actual = target / self._zoom
        self.scale(actual, actual)
        self._zoom = target

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
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.ZoomIn):
            self.zoom_in()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.ZoomOut):
            self.zoom_out()
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_0:
            self.reset_zoom()
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_F:
            self.fit_to_nodes()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = False
            if not self._panning:
                self.viewport().unsetCursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        self._scale_by(1.15 if delta > 0 else 1 / 1.15)
        event.accept()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        item = self.itemAt(event.position().toPoint())
        background_drag = event.button() == Qt.MouseButton.LeftButton and (
            item is None or isinstance(item, WorkflowGroupItem)
        )
        if (
            event.button() == Qt.MouseButton.MiddleButton
            or (event.button() == Qt.MouseButton.LeftButton and self._space_pan)
            or (event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.AltModifier)
            or background_drag
        ):
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._panning:
            pos = event.position().toPoint()
            delta = pos - self._pan_start
            self._pan_start = pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._panning and event.button() in {Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton}:
            self._panning = False
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor if self._space_pan else Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _node_item_for(self, item: QGraphicsItem | None) -> NodeItem | None:
        while item is not None:
            if isinstance(item, NodeItem):
                return item
            item = item.parentItem()
        return None

    def contextMenuEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        menu = QMenu(self)
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #f8fafc;
                border: 1px solid #94a3b8;
                color: #0f172a;
                padding: 5px;
            }
            QMenu::item {
                background-color: transparent;
                color: #0f172a;
                padding: 6px 26px 6px 12px;
                min-width: 220px;
            }
            QMenu::item:selected {
                background-color: #dbeafe;
                color: #0f172a;
            }
            QMenu::item:disabled {
                color: #94a3b8;
            }
            QMenu::separator {
                height: 1px;
                background: #cbd5e1;
                margin: 5px 8px;
            }
            QMenu::right-arrow {
                width: 8px;
                height: 8px;
            }
            """
        )
        clicked_node = self._node_item_for(self.itemAt(event.pos()))
        if clicked_node is not None and not clicked_node.isSelected():
            self.graph_scene.clearSelection()
            clicked_node.setSelected(True)

        selected_nodes = [
            item.node_id for item in self.graph_scene.selectedItems() if isinstance(item, NodeItem)
        ]
        grouped_nodes = {node_id for group in self.graph_scene.graph.groups.values() for node_id in group.node_ids}
        connections = menu.addMenu(tr("menu.connection_style"))
        curved_action = connections.addAction(tr("menu.curved_connections"))
        curved_action.setCheckable(True)
        curved_action.setChecked(self.graph_scene.graph.edge_style == "curved")
        curved_action.triggered.connect(lambda _checked=False: self.graph_scene.set_edge_style("curved"))
        segmented_action = connections.addAction(tr("menu.segmented_connections"))
        segmented_action.setCheckable(True)
        segmented_action.setChecked(self.graph_scene.graph.edge_style == "segmented")
        segmented_action.triggered.connect(lambda _checked=False: self.graph_scene.set_edge_style("segmented"))
        if selected_nodes:
            menu.addSeparator()
            group_action = menu.addAction(tr("menu.group_selected"), self._group_selected_nodes)
            group_action.setEnabled(len(selected_nodes) >= 2 and not any(node_id in grouped_nodes for node_id in selected_nodes))
            ungroup_action = menu.addAction(tr("menu.ungroup_selected"), self._ungroup_selected_nodes)
            ungroup_action.setEnabled(any(node_id in grouped_nodes for node_id in selected_nodes))
        if selected_nodes:
            if any(not self.graph_scene.graph.nodes[node_id].enabled for node_id in selected_nodes):
                menu.addAction(tr("menu.enable_selected"), lambda _checked=False: self.graph_scene.set_selection_enabled(True))
            if any(self.graph_scene.graph.nodes[node_id].enabled for node_id in selected_nodes):
                menu.addAction(tr("menu.disable_selected"), lambda _checked=False: self.graph_scene.set_selection_enabled(False))
            menu.addAction(tr("menu.toggle_selected"), self.graph_scene.toggle_selection_enabled)
            menu.addSeparator()
        menu.addAction(tr("menu.auto_arrange"), self.auto_arrange)
        menu.addSeparator()
        add_menu = menu.addMenu(tr("menu.add_node"))
        scene_pos = self.mapToScene(event.pos())
        for type_name, definition in sorted(self.graph_scene.graph.definitions.items(), key=lambda item: (item[1].category, item[1].title)):
            action = add_menu.addAction(f"{category_label(definition.category)}: {node_type_label(type_name, definition.title)}")
            action.triggered.connect(lambda _checked=False, node_type=type_name: self.graph_scene.add_node_at(node_type, scene_pos))
        menu.addAction(tr("menu.delete_selection"), self.graph_scene.delete_selection)
        menu.exec(event.globalPos())
