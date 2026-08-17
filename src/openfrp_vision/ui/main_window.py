from __future__ import annotations

import os

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from openfrp_vision.camera.base import CameraSettings, FrameSnapshot
from openfrp_vision.camera.hikvision import HikvisionCameraAdapter
from openfrp_vision.core.events import AppState, OverlayMode, ShortcutPressed, WorkflowFinished, WorkflowRequested, reduce_state
from openfrp_vision.core.profiles import ProfileStore
from openfrp_vision.ui.camera_preview import CameraGLView
from openfrp_vision.ui.inspector import NodeInspector
from openfrp_vision.ui.node_overlay import NodeOverlayView
from openfrp_vision.ui.roi_overlay import RoiHandleWidget
from openfrp_vision.workflow.executor import WorkflowExecutor, WorkflowRun
from openfrp_vision.workflow.model import GraphError, RecipeGraph
from openfrp_vision.workflow.nodes import build_default_graph, build_definitions


class CameraWorkbench(QWidget):
    roi_overlay_changed = Signal(str, tuple)

    def __init__(self, graph: RecipeGraph) -> None:
        super().__init__()
        self.camera = CameraGLView()
        self.overlay = NodeOverlayView(graph)
        self.inspector = NodeInspector(graph, self)
        self.roi_overlays: dict[str, RoiHandleWidget] = {}
        self._frame_size = (1280, 720)
        self._inspector_enabled = True
        self._inspector_placed = False
        self.hud = QWidget(self)
        self.hud.setObjectName("hud")
        self.hud.setStyleSheet(
            """
            QWidget#hud {
                background: rgba(20, 28, 35, 165);
                border: 1px solid rgba(255, 255, 255, 120);
                border-radius: 6px;
            }
            QLabel { color: white; font-weight: 600; }
            """
        )

        hud_layout = QHBoxLayout(self.hud)
        hud_layout.setContentsMargins(8, 8, 8, 8)
        self.mode_label = QLabel("ROI")
        self.camera_label = QLabel("SYNTH")
        self.shortcut_label = QLabel("Run Ctrl+Return")
        self.profile_combo = QComboBox()
        self.profile_combo.setFixedWidth(150)
        self.new_profile_button = QPushButton("New")
        self.save_profile_button = QPushButton("Save")
        self.node_combo = QComboBox()
        self.node_combo.setFixedWidth(145)
        for type_name, definition in sorted(graph.definitions.items(), key=lambda item: (item[1].category, item[1].title)):
            self.node_combo.addItem(definition.title, type_name)
        self.add_button = QPushButton("Add")
        self.delete_button = QPushButton("Delete")
        self.roi_button = QPushButton("ROI")
        self.camera_button = QPushButton("Camera")
        self.inspector_button = QPushButton("Inspector")
        self.zoom_out_button = QPushButton("-")
        self.zoom_in_button = QPushButton("+")
        self.zoom_fit_button = QPushButton("Fit")
        self.zoom_reset_button = QPushButton("100%")
        self.toggle_button = QPushButton("Overlay")
        for button in (self.zoom_out_button, self.zoom_in_button):
            button.setFixedWidth(30)
        for button in (self.zoom_fit_button, self.zoom_reset_button):
            button.setFixedWidth(44)
        hud_layout.addWidget(self.mode_label)
        hud_layout.addWidget(self.camera_label)
        hud_layout.addWidget(self.shortcut_label)
        hud_layout.addWidget(self.profile_combo)
        hud_layout.addWidget(self.new_profile_button)
        hud_layout.addWidget(self.save_profile_button)
        hud_layout.addWidget(self.node_combo)
        hud_layout.addWidget(self.add_button)
        hud_layout.addWidget(self.delete_button)
        hud_layout.addWidget(self.roi_button)
        hud_layout.addWidget(self.camera_button)
        hud_layout.addWidget(self.inspector_button)
        hud_layout.addWidget(self.zoom_out_button)
        hud_layout.addWidget(self.zoom_in_button)
        hud_layout.addWidget(self.zoom_fit_button)
        hud_layout.addWidget(self.zoom_reset_button)
        hud_layout.addWidget(self.toggle_button)
        self.hud.setFixedHeight(46)

        layout = QStackedLayout(self)
        layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        layout.addWidget(self.camera)
        layout.addWidget(self.overlay)
        layout.setCurrentWidget(self.overlay)
        self._raise_floaters()

    def apply_overlay_mode(self, mode: OverlayMode) -> None:
        self.overlay.set_overlay_mode(mode)
        self.mode_label.setText(mode.value.upper())
        self.inspector.setVisible(self._inspector_enabled and mode in {OverlayMode.GRAPH, OverlayMode.EDIT})
        self._raise_floaters()

    def set_camera_status(self, status: str) -> None:
        self.camera_label.setText(status)
        self._raise_floaters()

    def set_run_shortcut(self, shortcut: str) -> None:
        self.shortcut_label.setText(f"Run {shortcut}")
        self._raise_floaters()

    def set_graph(self, graph: RecipeGraph) -> None:
        self.overlay.set_graph(graph)
        self.inspector.set_graph(graph)
        self.sync_roi_overlays(graph)
        self._raise_floaters()

    def toggle_inspector(self, mode: OverlayMode) -> None:
        self._inspector_enabled = not self._inspector_enabled
        self.apply_overlay_mode(mode)

    def set_profiles(self, profiles: list[tuple[str, str]], active_profile_id: str) -> None:
        blocked = self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for profile_id, name in profiles:
            self.profile_combo.addItem(name, profile_id)
        index = self.profile_combo.findData(active_profile_id)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
        self.profile_combo.blockSignals(blocked)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self.hud.setFixedWidth(min(max(420, self.hud.sizeHint().width()), max(420, self.width() - 24)))
        self.hud.move(12, 12)
        if not self._inspector_placed:
            self.inspector.resize(340, max(430, min(640, self.height() - 82)))
            self.inspector.move(max(12, self.width() - self.inspector.width() - 16), 62)
            self._inspector_placed = True
        else:
            self.inspector.clamp_to_parent()
        for overlay in self.roi_overlays.values():
            overlay.parent_resized()
        self._raise_floaters()

    def set_frame_size(self, frame_size: tuple[int, int]) -> None:
        self._frame_size = frame_size
        for overlay in self.roi_overlays.values():
            overlay.set_roi(overlay.title, overlay.roi, self._frame_size)

    def sync_roi_overlays(self, graph: RecipeGraph) -> None:
        enabled_nodes = {
            node_id: node
            for node_id, node in graph.nodes.items()
            if node.type_name == "roi" and bool(node.params.get("live_overlay", False))
        }
        for node_id in list(self.roi_overlays):
            if node_id not in enabled_nodes:
                widget = self.roi_overlays.pop(node_id)
                widget.hide()
                widget.deleteLater()
        for node_id, node in enabled_nodes.items():
            widget = self.roi_overlays.get(node_id)
            if widget is None:
                widget = RoiHandleWidget(node_id, self)
                widget.roi_changed.connect(self.roi_overlay_changed)
                self.roi_overlays[node_id] = widget
            roi = (
                int(node.params.get("x", 0)),
                int(node.params.get("y", 0)),
                int(node.params.get("width", 100)),
                int(node.params.get("height", 100)),
            )
            widget.set_roi(node.title, roi, self._frame_size)
            widget.show()
        self._raise_floaters()

    def _raise_floaters(self) -> None:
        self.overlay.raise_()
        for overlay in self.roi_overlays.values():
            overlay.raise_()
        self.inspector.raise_()
        self.hud.raise_()


class MainWindow(QMainWindow):
    workflow_done = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.state = AppState()
        self.definitions = build_definitions()
        default_graph = build_default_graph(self.definitions)
        self.profile_store = ProfileStore.load()
        self.profile_store.ensure_default(default_graph)
        self.active_profile_id = self.profile_store.active_profile_id()
        self.graph = self._load_profile_graph(self.active_profile_id, default_graph)
        self.executor = WorkflowExecutor()
        self.latest_snapshot: FrameSnapshot | None = None
        self.camera_adapter = HikvisionCameraAdapter()
        self._using_real_camera = False
        self._roi_target_node_id: str | None = None
        self._closing = False
        self.workbench = CameraWorkbench(self.graph)
        self.setCentralWidget(self.workbench)
        self.setWindowTitle("OpenFRP Vision Workbench")
        self.resize(1280, 800)

        self.workbench.toggle_button.clicked.connect(lambda: self.dispatch(ShortcutPressed("toggle_overlay")))
        self.workbench.profile_combo.currentIndexChanged.connect(self._profile_selected)
        self.workbench.new_profile_button.clicked.connect(self._new_profile)
        self.workbench.save_profile_button.clicked.connect(self._save_active_profile)
        self.workbench.add_button.clicked.connect(self._add_node_from_hud)
        self.workbench.delete_button.clicked.connect(self.workbench.overlay.graph_scene.delete_selection)
        self.workbench.roi_button.clicked.connect(lambda: self._start_roi_selection())
        self.workbench.camera_button.clicked.connect(self._restart_camera)
        self.workbench.inspector_button.clicked.connect(lambda: self.workbench.toggle_inspector(self.state.overlay_mode))
        self.workbench.zoom_out_button.clicked.connect(self.workbench.overlay.zoom_out)
        self.workbench.zoom_in_button.clicked.connect(self.workbench.overlay.zoom_in)
        self.workbench.zoom_fit_button.clicked.connect(self.workbench.overlay.fit_to_nodes)
        self.workbench.zoom_reset_button.clicked.connect(self.workbench.overlay.reset_zoom)
        self.workbench.camera.roi_selected.connect(self._roi_selected)
        self.workbench.apply_overlay_mode(self.state.overlay_mode)
        self.workbench.overlay.graph_scene.message.connect(self.statusBar().showMessage)
        self.workbench.overlay.graph_scene.graph_changed.connect(self._graph_changed)
        self.workbench.overlay.graph_scene.camera_settings_changed.connect(self._apply_camera_settings)
        self.workbench.overlay.graph_scene.trigger_settings_changed.connect(lambda _params: self._configure_trigger_action())
        self.workbench.overlay.graph_scene.selectionChanged.connect(self._selection_changed)
        self.workbench.inspector.parameter_changed.connect(self._inspector_parameter_changed)
        self.workbench.inspector.request_roi.connect(self._start_roi_selection)
        self.workbench.roi_overlay_changed.connect(self._roi_overlay_changed)
        self.workflow_done.connect(self._workflow_finished)
        self._populate_profiles()
        self._sync_roi_overlays()
        self.statusBar().showMessage(f"Loaded {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        self.camera_adapter.frame_ready.connect(self._on_live_camera_frame)
        self.camera_adapter.status.connect(self._camera_status)
        self.camera_adapter.error.connect(self._camera_error)
        self._run_action = QAction("Trigger Workflow", self)
        self._run_action.triggered.connect(self._trigger_workflow)
        self.addAction(self._run_action)
        self._configure_trigger_action()
        self._apply_active_profile_camera_settings()

        overlay_action = QAction("Toggle Overlay", self)
        overlay_action.setShortcut(QKeySequence("Tab"))
        overlay_action.triggered.connect(lambda: self.dispatch(ShortcutPressed("toggle_overlay")))
        self.addAction(overlay_action)

        self._frame_id = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._synthetic_frame)
        self._timer.start(33)
        self._camera_watchdog = QTimer(self)
        self._camera_watchdog.setInterval(1000)
        self._camera_watchdog.timeout.connect(self._camera_wait_tick)
        self._debug_dialogs: list[QDialog] = []
        QTimer.singleShot(0, self.workbench.overlay.fit_to_nodes)
        QTimer.singleShot(0, self._start_camera_if_available)

    def dispatch(self, event) -> None:  # type: ignore[no-untyped-def]
        self.state = reduce_state(self.state, event)
        self.workbench.apply_overlay_mode(self.state.overlay_mode)

    def _graph_changed(self) -> None:
        self._configure_trigger_action()
        self._sync_roi_overlays()

    def _populate_profiles(self) -> None:
        profiles = [(profile_id, self.profile_store.profile_name(profile_id)) for profile_id in self.profile_store.profile_ids()]
        self.workbench.set_profiles(profiles, self.active_profile_id)

    def _load_profile_graph(self, profile_id: str, fallback: RecipeGraph | None = None) -> RecipeGraph:
        data = self.profile_store.graph_data(profile_id)
        if data is not None:
            try:
                return RecipeGraph.from_dict(self.definitions, data)
            except (GraphError, KeyError, TypeError, ValueError) as exc:
                print(f"Profile {profile_id} could not be loaded: {exc}")
        return fallback or build_default_graph(self.definitions)

    def _install_graph(self, graph: RecipeGraph) -> None:
        self.graph = graph
        self.workbench.set_graph(graph)
        self.workbench.inspector.inspect(None)
        self._configure_trigger_action()
        self._apply_active_profile_camera_settings()
        self._sync_roi_overlays()

    def _save_active_profile(self, silent: bool = False) -> None:
        self.profile_store.save_profile(self.active_profile_id, self.graph)
        if not silent:
            self.statusBar().showMessage(
                f"Saved profile {self.profile_store.profile_name(self.active_profile_id)} to {self.profile_store.path}"
            )

    def _new_profile(self) -> None:
        name, accepted = QInputDialog.getText(self, "New Product Profile", "Product profile name:")
        if not accepted:
            return
        name = name.strip()
        if not name:
            self.statusBar().showMessage("Profile name is empty")
            return
        self._save_active_profile(silent=True)
        self.active_profile_id = self.profile_store.create_profile(name, self.graph)
        self._populate_profiles()
        self.statusBar().showMessage(f"Created profile {name}")

    def _profile_selected(self, index: int) -> None:
        if index < 0:
            return
        profile_id = str(self.workbench.profile_combo.itemData(index))
        if not profile_id or profile_id == self.active_profile_id:
            return
        self._save_active_profile(silent=True)
        graph = self._load_profile_graph(profile_id)
        self.active_profile_id = profile_id
        self.profile_store.set_active(profile_id)
        self._install_graph(graph)
        self.statusBar().showMessage(f"Loaded profile {self.profile_store.profile_name(profile_id)}")

    def _synthetic_frame(self) -> None:
        if self._using_real_camera:
            return
        self._frame_id += 1
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[:, :, 0] = 28
        image[:, :, 1] = np.linspace(35, 95, image.shape[1], dtype=np.uint8)
        image[:, :, 2] = 58
        x = 100 + (self._frame_id * 3) % 900
        image[240:360, x : x + 260] = (210, 220, 225)
        snapshot = FrameSnapshot(self._frame_id, self._frame_id / 30.0, image)
        self.latest_snapshot = snapshot
        self.workbench.camera.set_frame(snapshot)
        self.workbench.set_frame_size(snapshot.size)

    def _start_camera_if_available(self) -> None:
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            self.workbench.set_camera_status("SYNTH")
            self.statusBar().showMessage("Offscreen Qt platform: camera auto-start skipped")
            return
        self.workbench.set_camera_status("SEARCH")
        if self.camera_adapter.start():
            self._apply_active_profile_camera_settings()
            self._using_real_camera = False
            if not self._timer.isActive():
                self._timer.start(33)
            self._camera_watchdog.start()
            self.workbench.set_camera_status(self.camera_adapter.hud_status)
        else:
            self._using_real_camera = False
            self._camera_watchdog.stop()
            if not self._timer.isActive():
                self._timer.start(33)
            self.workbench.set_camera_status(f"{self.camera_adapter.hud_status}/SYNTH")

    def _restart_camera(self) -> None:
        self.statusBar().showMessage("Retrying Hikvision camera...")
        self._using_real_camera = False
        self._camera_watchdog.stop()
        self.camera_adapter.stop()
        if not self._timer.isActive():
            self._timer.start(33)
        self._start_camera_if_available()

    def _on_live_camera_frame(self, snapshot: FrameSnapshot) -> None:
        if not self._using_real_camera:
            self._using_real_camera = True
            self._timer.stop()
            self._camera_watchdog.stop()
        self.latest_snapshot = snapshot
        self.workbench.camera.set_frame(snapshot)
        self.workbench.set_frame_size(snapshot.size)
        if snapshot.frame_id == 1 or snapshot.frame_id % 20 == 0:
            self.workbench.set_camera_status(self.camera_adapter.hud_status)

    def _camera_wait_tick(self) -> None:
        if self.camera_adapter.latest_snapshot() is not None:
            return
        self.workbench.set_camera_status("HIK WAIT/SYNTH")
        if self.camera_adapter.last_status:
            self.statusBar().showMessage(self.camera_adapter.last_status)

    def _camera_status(self, message: str) -> None:
        self.statusBar().showMessage(message)
        if not self._using_real_camera and self.camera_adapter.device_count == 0:
            self.workbench.set_camera_status(f"{self.camera_adapter.hud_status}/SYNTH")
        elif not self._using_real_camera and self.camera_adapter.device_count > 0:
            self.workbench.set_camera_status("HIK WAIT/SYNTH")

    def _camera_error(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self._using_real_camera = False
        self._camera_watchdog.stop()
        if not self._timer.isActive():
            self._timer.start(33)
        self.workbench.set_camera_status("ERR/SYNTH")

    def _add_node_from_hud(self) -> None:
        type_name = str(self.workbench.node_combo.currentData())
        node_id = self.workbench.overlay.add_node(type_name)
        item = self.workbench.overlay.graph_scene.node_items.get(node_id)
        if item is not None:
            item.setSelected(True)
        self.workbench.inspector.inspect(node_id)

    def _start_roi_selection(self, requested_node_id: str | None = None) -> None:
        node_id = requested_node_id if isinstance(requested_node_id, str) and requested_node_id in self.graph.nodes else self._target_roi_node_id()
        if node_id is None:
            self.statusBar().showMessage("Add an ROI node before selecting ROI")
            return
        if self.graph.nodes[node_id].type_name != "roi":
            self.statusBar().showMessage("Select an ROI node before picking ROI")
            return
        self._roi_target_node_id = node_id
        self.state.overlay_mode = OverlayMode.ROI
        self.workbench.apply_overlay_mode(self.state.overlay_mode)
        self.workbench.camera.set_roi_selection_enabled(True)
        self.statusBar().showMessage(f"Selecting ROI for {self.graph.nodes[node_id].title}")

    def _roi_selected(self, roi: tuple[int, int, int, int]) -> None:
        self.workbench.camera.set_roi_selection_enabled(False)
        node_id = self._roi_target_node_id or self._target_roi_node_id()
        self._roi_target_node_id = None
        if node_id is None:
            self.statusBar().showMessage("No ROI node available")
            return
        self.workbench.overlay.graph_scene.update_roi_node(node_id, roi)
        self._sync_roi_overlays()
        item = self.workbench.overlay.graph_scene.node_items.get(node_id)
        if item is not None:
            item.setSelected(True)
        self.workbench.inspector.inspect(node_id)
        self.state.overlay_mode = OverlayMode.GRAPH
        self.workbench.apply_overlay_mode(self.state.overlay_mode)

    def _target_roi_node_id(self) -> str | None:
        selected = self.workbench.overlay.graph_scene.selectedItems()
        for item in selected:
            node_id = getattr(item, "node_id", None)
            node = self.graph.nodes.get(node_id)
            if node is not None and node.type_name == "roi":
                return node.node_id
        if "serial_roi" in self.graph.nodes and self.graph.nodes["serial_roi"].type_name == "roi":
            return "serial_roi"
        for node in self.graph.nodes.values():
            if node.type_name == "roi":
                return node.node_id
        return None

    def _apply_camera_settings(self, params: dict) -> None:
        settings = CameraSettings(
            exposure_us=int(params.get("exposure_us", 30000)),
            gamma=int(params.get("gamma", 100)),
            contrast=int(params.get("contrast", 100)),
            analog_gain=int(params.get("analog_gain", 16)),
            ae_enabled=bool(params.get("ae_enabled", False)),
            reverse_x=bool(params.get("reverse_x", False)),
            reverse_y=bool(params.get("reverse_y", False)),
        )
        self.camera_adapter.apply_settings(settings)

    def _sync_roi_overlays(self) -> None:
        if self.latest_snapshot is not None:
            self.workbench.set_frame_size(self.latest_snapshot.size)
        self.workbench.sync_roi_overlays(self.graph)

    def _roi_overlay_changed(self, node_id: str, roi: tuple[int, int, int, int]) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None or node.type_name != "roi":
            self._sync_roi_overlays()
            return
        x, y, width, height = roi
        node.params.update({"x": int(x), "y": int(y), "width": int(width), "height": int(height)})
        self.graph.results.clear()
        self.graph.revision += 1
        self.workbench.overlay.graph_scene.update_results()
        if self.workbench.inspector.current_node_id == node_id:
            self.workbench.inspector.inspect(node_id)
        self.statusBar().showMessage(f"{node.title}: x={x}, y={y}, {width} x {height}")

    def _active_camera_settings_node_params(self) -> dict | None:
        for node in self.graph.nodes.values():
            if node.type_name == "camera_settings":
                return node.params
        return None

    def _apply_active_profile_camera_settings(self) -> None:
        params = self._active_camera_settings_node_params()
        if params is not None:
            self._apply_camera_settings(params)

    def _selection_changed(self) -> None:
        if self._closing:
            return
        try:
            selected = self.workbench.overlay.graph_scene.selectedItems()
        except RuntimeError:
            return
        for item in selected:
            node_id = getattr(item, "node_id", None)
            if node_id in self.graph.nodes:
                self.workbench.inspector.inspect(node_id)
                return
        self.workbench.inspector.inspect(None)

    def _inspector_parameter_changed(self, node_id: str, key: str, value: object) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None:
            self.workbench.inspector.inspect(None)
            return
        old_value = node.params.get(key)
        if old_value == value:
            return
        node.params[key] = value
        self.graph.results.clear()
        self.graph.revision += 1
        self.workbench.overlay.graph_scene.update_results()
        self.workbench.inspector.refresh_result()
        self._sync_roi_overlays()
        if node.type_name == "camera_settings":
            self._apply_camera_settings(node.params)
            self.statusBar().showMessage(f"{node.title}: {key} updated for profile {self.profile_store.profile_name(self.active_profile_id)}")
        elif node.type_name == "trigger_switch":
            self._configure_trigger_action()
            self.statusBar().showMessage(f"{node.title}: trigger updated")
        else:
            self.statusBar().showMessage(f"{node.title}: {key} updated")

    def _configure_trigger_action(self) -> None:
        node = self.graph.nodes.get("trigger")
        params = node.params if node is not None else {}
        source = str(params.get("source", "keyboard"))
        shortcut = str(params.get("shortcut", "Ctrl+Return"))
        armed = bool(params.get("armed", True))
        if source == "keyboard" and armed:
            self._run_action.setShortcut(QKeySequence(shortcut))
            self.workbench.set_run_shortcut(shortcut)
        else:
            self._run_action.setShortcut(QKeySequence())
            label = "disabled" if not armed else source
            self.workbench.set_run_shortcut(label)

    def _trigger_workflow(self) -> None:
        trigger_node = self.graph.nodes.get("trigger")
        if trigger_node is not None and not bool(trigger_node.params.get("armed", True)):
            self.statusBar().showMessage("Trigger switch is disabled")
            return
        if self.latest_snapshot is None:
            self.statusBar().showMessage("No frame available")
            return
        if self.state.workflow_busy:
            self.statusBar().showMessage("Workflow already running")
            return
        self.dispatch(WorkflowRequested(self.latest_snapshot, self.graph.revision))
        future = self.executor.submit(self.graph, self.latest_snapshot)
        future.add_done_callback(lambda done: self.workflow_done.emit(done))
        self.statusBar().showMessage(f"Workflow submitted for frame {self.latest_snapshot.frame_id}")

    def _workflow_finished(self, future) -> None:  # type: ignore[no-untyped-def]
        try:
            run: WorkflowRun = future.result()
        except Exception as exc:
            self.state.workflow_busy = False
            self.statusBar().showMessage(f"Workflow failed: {exc}")
            return
        final_result = run.final_result
        passed = bool(final_result.get("passed", False))
        self.dispatch(WorkflowFinished(run.run_id, passed, final_result))
        self.workbench.overlay.graph_scene.update_results()
        self.workbench.inspector.refresh_result()
        self._show_debug_popups(run)
        self.statusBar().showMessage(f"Workflow {final_result.get('decision', 'DONE')} on frame {run.frame_id}")

    def _show_debug_popups(self, run: WorkflowRun) -> None:
        self._debug_dialogs = [dialog for dialog in self._debug_dialogs if dialog.isVisible()]
        for node_id, result in run.results.items():
            node = self.graph.nodes.get(node_id)
            if node is None or node.type_name != "debug_image" or not bool(node.params.get("popup", False)):
                continue
            image = result.preview
            if image is None:
                continue
            qimage = CameraGLView._to_qimage(image)
            if qimage.isNull():
                continue
            dialog = QDialog(self)
            dialog.setWindowTitle(node.title)
            layout = QVBoxLayout(dialog)
            label = QLabel()
            label.setPixmap(QPixmap.fromImage(qimage).scaled(900, 700, aspectMode=Qt.AspectRatioMode.KeepAspectRatio))
            layout.addWidget(label)
            dialog.resize(label.pixmap().size())
            dialog.show()
            self._debug_dialogs.append(dialog)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._closing = True
        try:
            self.workbench.overlay.graph_scene.selectionChanged.disconnect(self._selection_changed)
        except (RuntimeError, TypeError):
            pass
        self._save_active_profile(silent=True)
        self.camera_adapter.stop()
        self.executor.shutdown()
        super().closeEvent(event)
