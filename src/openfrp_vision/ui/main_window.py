from __future__ import annotations

import os

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
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
from openfrp_vision.ui.camera_preview import CameraGLView
from openfrp_vision.ui.node_overlay import NodeOverlayView
from openfrp_vision.workflow.executor import WorkflowExecutor, WorkflowRun
from openfrp_vision.workflow.model import RecipeGraph
from openfrp_vision.workflow.nodes import build_default_graph, build_definitions


class CameraWorkbench(QWidget):
    def __init__(self, graph: RecipeGraph) -> None:
        super().__init__()
        self.camera = CameraGLView()
        self.overlay = NodeOverlayView(graph)
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
        self.node_combo = QComboBox()
        self.node_combo.setFixedWidth(145)
        for type_name, definition in sorted(graph.definitions.items(), key=lambda item: (item[1].category, item[1].title)):
            self.node_combo.addItem(definition.title, type_name)
        self.add_button = QPushButton("Add")
        self.delete_button = QPushButton("Delete")
        self.roi_button = QPushButton("ROI")
        self.camera_button = QPushButton("Camera")
        self.toggle_button = QPushButton("Overlay")
        hud_layout.addWidget(self.mode_label)
        hud_layout.addWidget(self.camera_label)
        hud_layout.addWidget(self.shortcut_label)
        hud_layout.addWidget(self.node_combo)
        hud_layout.addWidget(self.add_button)
        hud_layout.addWidget(self.delete_button)
        hud_layout.addWidget(self.roi_button)
        hud_layout.addWidget(self.camera_button)
        hud_layout.addWidget(self.toggle_button)
        self.hud.setFixedSize(860, 46)

        layout = QStackedLayout(self)
        layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        layout.addWidget(self.camera)
        layout.addWidget(self.overlay)
        layout.setCurrentWidget(self.overlay)
        self.overlay.raise_()
        self.hud.raise_()

    def apply_overlay_mode(self, mode: OverlayMode) -> None:
        self.overlay.set_overlay_mode(mode)
        self.mode_label.setText(mode.value.upper())
        self.overlay.raise_()
        self.hud.raise_()

    def set_camera_status(self, status: str) -> None:
        self.camera_label.setText(status)
        self.overlay.raise_()
        self.hud.raise_()

    def set_run_shortcut(self, shortcut: str) -> None:
        self.shortcut_label.setText(f"Run {shortcut}")
        self.overlay.raise_()
        self.hud.raise_()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self.hud.move(12, 12)
        self.overlay.raise_()
        self.hud.raise_()


class MainWindow(QMainWindow):
    workflow_done = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.state = AppState()
        self.definitions = build_definitions()
        self.graph = build_default_graph(self.definitions)
        self.executor = WorkflowExecutor()
        self.latest_snapshot: FrameSnapshot | None = None
        self.camera_adapter = HikvisionCameraAdapter()
        self._using_real_camera = False
        self.workbench = CameraWorkbench(self.graph)
        self.setCentralWidget(self.workbench)
        self.setWindowTitle("OpenFRP Vision Workbench")
        self.resize(1280, 800)

        self.workbench.toggle_button.clicked.connect(lambda: self.dispatch(ShortcutPressed("toggle_overlay")))
        self.workbench.add_button.clicked.connect(self._add_node_from_hud)
        self.workbench.delete_button.clicked.connect(self.workbench.overlay.graph_scene.delete_selection)
        self.workbench.roi_button.clicked.connect(self._start_roi_selection)
        self.workbench.camera_button.clicked.connect(self._restart_camera)
        self.workbench.camera.roi_selected.connect(self._roi_selected)
        self.workbench.apply_overlay_mode(self.state.overlay_mode)
        self.workbench.overlay.graph_scene.message.connect(self.statusBar().showMessage)
        self.workbench.overlay.graph_scene.graph_changed.connect(self._graph_changed)
        self.workbench.overlay.graph_scene.camera_settings_changed.connect(self._apply_camera_settings)
        self.workbench.overlay.graph_scene.trigger_settings_changed.connect(lambda _params: self._configure_trigger_action())
        self.workflow_done.connect(self._workflow_finished)
        self.statusBar().showMessage(f"Loaded {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        self.camera_adapter.frame_ready.connect(self._on_live_camera_frame)
        self.camera_adapter.status.connect(self._camera_status)
        self.camera_adapter.error.connect(self._camera_error)
        self._run_action = QAction("Trigger Workflow", self)
        self._run_action.triggered.connect(self._trigger_workflow)
        self.addAction(self._run_action)
        self._configure_trigger_action()

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
        QTimer.singleShot(0, self._start_camera_if_available)

    def dispatch(self, event) -> None:  # type: ignore[no-untyped-def]
        self.state = reduce_state(self.state, event)
        self.workbench.apply_overlay_mode(self.state.overlay_mode)

    def _graph_changed(self) -> None:
        self._configure_trigger_action()

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

    def _start_camera_if_available(self) -> None:
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            self.workbench.set_camera_status("SYNTH")
            self.statusBar().showMessage("Offscreen Qt platform: camera auto-start skipped")
            return
        self.workbench.set_camera_status("SEARCH")
        if self.camera_adapter.start():
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

    def _start_roi_selection(self) -> None:
        node_id = self._target_roi_node_id()
        if node_id is None:
            self.statusBar().showMessage("Add an ROI node before selecting ROI")
            return
        self.state.overlay_mode = OverlayMode.ROI
        self.workbench.apply_overlay_mode(self.state.overlay_mode)
        self.workbench.camera.set_roi_selection_enabled(True)
        self.statusBar().showMessage(f"Selecting ROI for {self.graph.nodes[node_id].title}")

    def _roi_selected(self, roi: tuple[int, int, int, int]) -> None:
        self.workbench.camera.set_roi_selection_enabled(False)
        node_id = self._target_roi_node_id()
        if node_id is None:
            self.statusBar().showMessage("No ROI node available")
            return
        self.workbench.overlay.graph_scene.update_roi_node(node_id, roi)
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
        self.camera_adapter.stop()
        self.executor.shutdown()
        super().closeEvent(event)
