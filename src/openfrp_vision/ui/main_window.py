from __future__ import annotations

from collections import deque
import os
import threading
import time

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
from openfrp_vision.core.i18n import (
    SUPPORTED_LANGUAGES,
    category_label,
    current_language,
    language_name,
    node_label,
    node_type_label,
    param_label,
    set_language,
    tr,
)
from openfrp_vision.core.production_log import reset_serial_state, resolve_log_db_path, save_serial_state, serial_state
from openfrp_vision.core.profiles import ProfileStore
from openfrp_vision.ui.camera_preview import CameraGLView
from openfrp_vision.ui.inspector import NodeInspector
from openfrp_vision.ui.node_overlay import NodeOverlayView
from openfrp_vision.ui.production_log_viewer import ProductionLogViewer
from openfrp_vision.ui.production_stats import ProductionStatsWidget
from openfrp_vision.ui.result_indicator import ResultIndicatorWidget
from openfrp_vision.ui.roi_overlay import RoiHandleWidget
from openfrp_vision.workflow.executor import WorkflowExecutor, WorkflowRun
from openfrp_vision.workflow.model import GraphError, RecipeGraph
from openfrp_vision.workflow.nodes import build_default_graph, build_definitions, shutdown_paddle_worker, warm_paddle_worker


class CameraWorkbench(QWidget):
    roi_overlay_changed = Signal(str, tuple)

    def __init__(self, graph: RecipeGraph) -> None:
        super().__init__()
        self.camera = CameraGLView()
        self.overlay = NodeOverlayView(graph)
        self.inspector = NodeInspector(graph, self)
        self.result_indicator = ResultIndicatorWidget(self)
        self.production_stats = ProductionStatsWidget(self)
        self.production_log_viewer = ProductionLogViewer(self)
        self.production_log_viewer.hide()
        self.roi_overlays: dict[str, RoiHandleWidget] = {}
        self._frame_size = (1280, 720)
        self._inspector_enabled = True
        self._inspector_placed = False
        self._run_shortcut = "Ctrl+Return"
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
        self.mode_label = QLabel()
        self.camera_label = QLabel("SYNTH")
        self.shortcut_label = QLabel()
        self.language_combo = QComboBox()
        self.language_combo.setFixedWidth(112)
        self.profile_combo = QComboBox()
        self.profile_combo.setFixedWidth(150)
        self.new_profile_button = QPushButton()
        self.save_profile_button = QPushButton()
        self.node_combo = QComboBox()
        self.node_combo.setFixedWidth(145)
        self.add_button = QPushButton()
        self.delete_button = QPushButton()
        self.roi_button = QPushButton()
        self.camera_button = QPushButton()
        self.inspector_button = QPushButton()
        self.indicator_button = QPushButton()
        self.stats_button = QPushButton()
        self.logs_button = QPushButton()
        self.zoom_out_button = QPushButton("-")
        self.zoom_in_button = QPushButton("+")
        self.zoom_fit_button = QPushButton()
        self.zoom_reset_button = QPushButton("100%")
        self.toggle_button = QPushButton()
        for button in (self.zoom_out_button, self.zoom_in_button):
            button.setFixedWidth(30)
        for button in (self.zoom_fit_button, self.zoom_reset_button):
            button.setFixedWidth(44)
        hud_layout.addWidget(self.mode_label)
        hud_layout.addWidget(self.camera_label)
        hud_layout.addWidget(self.shortcut_label)
        hud_layout.addWidget(self.language_combo)
        hud_layout.addWidget(self.profile_combo)
        hud_layout.addWidget(self.new_profile_button)
        hud_layout.addWidget(self.save_profile_button)
        hud_layout.addWidget(self.node_combo)
        hud_layout.addWidget(self.add_button)
        hud_layout.addWidget(self.delete_button)
        hud_layout.addWidget(self.roi_button)
        hud_layout.addWidget(self.camera_button)
        hud_layout.addWidget(self.inspector_button)
        hud_layout.addWidget(self.indicator_button)
        hud_layout.addWidget(self.stats_button)
        hud_layout.addWidget(self.logs_button)
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
        self.set_language_choices(current_language())
        self.retranslate()
        self._raise_floaters()

    def apply_overlay_mode(self, mode: OverlayMode) -> None:
        self.overlay.set_overlay_mode(mode)
        self.mode_label.setText(tr(f"overlay.{mode.value}", default=mode.value.upper()).upper())
        self.inspector.setVisible(self._inspector_enabled and mode in {OverlayMode.GRAPH, OverlayMode.EDIT})
        self._raise_floaters()

    def set_camera_status(self, status: str) -> None:
        self.camera_label.setText(status)
        self._raise_floaters()

    def set_run_shortcut(self, shortcut: str) -> None:
        self._run_shortcut = shortcut
        self.shortcut_label.setText(tr("hud.run_shortcut", shortcut=shortcut))
        self._raise_floaters()

    def set_graph(self, graph: RecipeGraph) -> None:
        self.overlay.set_graph(graph)
        self.inspector.set_graph(graph)
        self._populate_node_combo()
        self.sync_roi_overlays(graph)
        self._raise_floaters()

    def set_indicator_settings(self, settings: dict) -> None:
        self.result_indicator.apply_settings(settings)
        self.result_indicator.show()
        self._raise_floaters()

    def set_stats_settings(self, settings: dict) -> None:
        self.production_stats.apply_settings(settings)
        self.production_stats.show()
        self._raise_floaters()

    def set_log_viewer_settings(self, settings: dict) -> None:
        self.production_log_viewer.apply_settings(settings)
        self._place_log_viewer_away_from_hud()
        self._raise_floaters()

    def configure_log_viewer(self, profile_id: str, db_path: str = "") -> None:
        self.production_log_viewer.configure(profile_id, db_path)
        self._raise_floaters()

    def set_stats_counts(self, ok: int, ng: int) -> None:
        self.production_stats.set_counts(ok, ng)

    def set_stats_fps(self, fps: float) -> None:
        self.production_stats.set_fps(fps)

    def toggle_inspector(self, mode: OverlayMode) -> None:
        self._inspector_enabled = not self._inspector_enabled
        self.apply_overlay_mode(mode)

    def toggle_indicator(self) -> None:
        self.result_indicator.setVisible(not self.result_indicator.isVisible())
        self._raise_floaters()

    def toggle_stats(self) -> None:
        self.production_stats.setVisible(not self.production_stats.isVisible())
        self._raise_floaters()

    def toggle_logs(self) -> None:
        show_logs = self.production_log_viewer.isHidden()
        self.production_log_viewer.setVisible(show_logs)
        if show_logs:
            self._place_log_viewer_away_from_hud()
            self.production_log_viewer.refresh()
        self._raise_floaters()

    def set_profiles(self, profiles: list[tuple[str, str]], active_profile_id: str) -> None:
        blocked = self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for profile_id, name in profiles:
            self.profile_combo.addItem(name, profile_id)
        index = self.profile_combo.findData(active_profile_id)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
        self.profile_combo.blockSignals(blocked)

    def set_language_choices(self, active_language: str) -> None:
        blocked = self.language_combo.blockSignals(True)
        self.language_combo.clear()
        for language in SUPPORTED_LANGUAGES:
            self.language_combo.addItem(language_name(language), language)
        index = self.language_combo.findData(active_language)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
        self.language_combo.blockSignals(blocked)

    def retranslate(self) -> None:
        self.new_profile_button.setText(tr("hud.new"))
        self.save_profile_button.setText(tr("hud.save"))
        self.add_button.setText(tr("hud.add"))
        self.delete_button.setText(tr("hud.delete"))
        self.roi_button.setText(tr("hud.roi"))
        self.camera_button.setText(tr("hud.camera"))
        self.inspector_button.setText(tr("hud.inspector"))
        self.indicator_button.setText(tr("hud.result"))
        self.stats_button.setText(tr("hud.stats"))
        self.logs_button.setText(tr("hud.logs"))
        self.zoom_fit_button.setText(tr("hud.fit"))
        self.toggle_button.setText(tr("hud.overlay"))
        self.set_run_shortcut(self._run_shortcut)
        self.set_language_choices(current_language())
        self._populate_node_combo()
        self.overlay.retranslate()
        self.inspector.retranslate()
        self.result_indicator.retranslate()
        self.production_stats.retranslate()
        self.production_log_viewer.retranslate()

    def _populate_node_combo(self) -> None:
        current_type = self.node_combo.currentData()
        blocked = self.node_combo.blockSignals(True)
        self.node_combo.clear()
        items = sorted(
            self.overlay.graph_scene.graph.definitions.items(),
            key=lambda item: (category_label(item[1].category), node_type_label(item[0], item[1].title)),
        )
        for type_name, definition in items:
            self.node_combo.addItem(node_type_label(type_name, definition.title), type_name)
        index = self.node_combo.findData(current_type)
        if index >= 0:
            self.node_combo.setCurrentIndex(index)
        self.node_combo.blockSignals(blocked)

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
        self.result_indicator.clamp_to_parent()
        self.production_stats.clamp_to_parent()
        self.production_log_viewer.clamp_to_parent()
        self._place_log_viewer_away_from_hud()
        for overlay in self.roi_overlays.values():
            overlay.parent_resized()
        self._raise_floaters()

    def set_frame_size(self, frame_size: tuple[int, int]) -> None:
        self._frame_size = frame_size
        self.sync_roi_overlays(self.overlay.graph_scene.graph)

    def sync_roi_overlays(self, graph: RecipeGraph) -> None:
        enabled_nodes = {
            node_id: node
            for node_id, node in graph.nodes.items()
            if node.enabled and node.type_name == "roi" and bool(node.params.get("live_overlay", False))
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
        self.production_log_viewer.raise_()
        self.production_stats.raise_()
        self.result_indicator.raise_()
        self.hud.raise_()

    def _place_log_viewer_away_from_hud(self) -> None:
        self.production_log_viewer.avoid_rect(self.hud.geometry().adjusted(-4, -4, 4, 8), 14)


class MainWindow(QMainWindow):
    workflow_done = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.state = AppState()
        self.profile_store = ProfileStore.load()
        set_language(self.profile_store.language())
        self.definitions = build_definitions()
        default_graph = build_default_graph(self.definitions)
        self.profile_store.ensure_default(default_graph)
        self.active_profile_id = self.profile_store.active_profile_id()
        self.graph = self._load_profile_graph(self.active_profile_id, default_graph)
        self.executor = WorkflowExecutor()
        self.latest_snapshot: FrameSnapshot | None = None
        self.camera_adapter = HikvisionCameraAdapter()
        self._using_real_camera = False
        self._roi_target_node_id: str | None = None
        self._fps_samples: deque[tuple[int, float]] = deque(maxlen=240)
        self._fps_ema = 0.0
        self._fps_last_publish = 0.0
        self._closing = False
        self._pending_serial_sample_node_id: str | None = None
        self._pending_serial_sample_future = None
        self.workbench = CameraWorkbench(self.graph)
        self.setCentralWidget(self.workbench)
        self.setWindowTitle(tr("app.title"))
        self.resize(1280, 800)

        self.workbench.toggle_button.clicked.connect(lambda: self.dispatch(ShortcutPressed("toggle_overlay")))
        self.workbench.language_combo.currentIndexChanged.connect(self._language_selected)
        self.workbench.profile_combo.currentIndexChanged.connect(self._profile_selected)
        self.workbench.new_profile_button.clicked.connect(self._new_profile)
        self.workbench.save_profile_button.clicked.connect(self._save_active_profile)
        self.workbench.add_button.clicked.connect(self._add_node_from_hud)
        self.workbench.delete_button.clicked.connect(self.workbench.overlay.graph_scene.delete_selection)
        self.workbench.roi_button.clicked.connect(lambda: self._start_roi_selection())
        self.workbench.camera_button.clicked.connect(self._restart_camera)
        self.workbench.inspector_button.clicked.connect(lambda: self.workbench.toggle_inspector(self.state.overlay_mode))
        self.workbench.indicator_button.clicked.connect(self.workbench.toggle_indicator)
        self.workbench.stats_button.clicked.connect(self.workbench.toggle_stats)
        self.workbench.logs_button.clicked.connect(self.workbench.toggle_logs)
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
        self.workbench.inspector.node_enabled_changed.connect(self._inspector_node_enabled_changed)
        self.workbench.inspector.request_roi.connect(self._start_roi_selection)
        self.workbench.inspector.request_serial_reset.connect(self._reset_serial_node_state)
        self.workbench.roi_overlay_changed.connect(self._roi_overlay_changed)
        self.workbench.result_indicator.settings_changed.connect(self._indicator_settings_changed)
        self.workbench.production_stats.settings_changed.connect(self._stats_settings_changed)
        self.workbench.production_log_viewer.settings_changed.connect(self._log_viewer_settings_changed)
        self.workflow_done.connect(self._workflow_finished)
        self._populate_profiles()
        self.workbench.set_indicator_settings(self.profile_store.indicator_settings(self.active_profile_id))
        self.workbench.set_stats_settings(self.profile_store.stats_settings(self.active_profile_id))
        self.workbench.set_log_viewer_settings(self.profile_store.log_viewer_settings(self.active_profile_id))
        self.workbench.configure_log_viewer(self.active_profile_id, self._active_log_db_path())
        self._sync_stats_counts()
        self._sync_roi_overlays()
        self.statusBar().showMessage(tr("status.loaded_nodes", nodes=len(self.graph.nodes), edges=len(self.graph.edges)))
        self.camera_adapter.frame_ready.connect(self._on_live_camera_frame)
        self.camera_adapter.status.connect(self._camera_status)
        self.camera_adapter.error.connect(self._camera_error)
        self._run_action = QAction(tr("action.trigger_workflow"), self)
        self._run_action.triggered.connect(self._trigger_workflow)
        self.addAction(self._run_action)
        self._configure_trigger_action()
        self._apply_active_profile_camera_settings()
        QTimer.singleShot(0, self._warm_active_profile_ocr)

        self._overlay_action = QAction(tr("action.toggle_overlay"), self)
        self._overlay_action.setShortcut(QKeySequence("Tab"))
        self._overlay_action.triggered.connect(lambda: self.dispatch(ShortcutPressed("toggle_overlay")))
        self.addAction(self._overlay_action)

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
        self.workbench.configure_log_viewer(self.active_profile_id, self._active_log_db_path())
        current = self.workbench.inspector.current_node_id
        if current in self.graph.nodes:
            self.workbench.inspector.inspect(current)

    def _populate_profiles(self) -> None:
        profiles = [(profile_id, self.profile_store.profile_name(profile_id)) for profile_id in self.profile_store.profile_ids()]
        self.workbench.set_profiles(profiles, self.active_profile_id)

    def _language_selected(self, index: int) -> None:
        if index < 0:
            return
        language = str(self.workbench.language_combo.itemData(index))
        if language == current_language():
            return
        set_language(language)
        self.profile_store.set_language(language)
        self.setWindowTitle(tr("app.title"))
        self._run_action.setText(tr("action.trigger_workflow"))
        self._overlay_action.setText(tr("action.toggle_overlay"))
        self.workbench.retranslate()
        self.workbench.apply_overlay_mode(self.state.overlay_mode)
        self._configure_trigger_action()
        self.statusBar().showMessage(tr("status.language_changed", language=language_name(language)))

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
        self.workbench.set_indicator_settings(self.profile_store.indicator_settings(self.active_profile_id))
        self.workbench.set_stats_settings(self.profile_store.stats_settings(self.active_profile_id))
        self.workbench.set_log_viewer_settings(self.profile_store.log_viewer_settings(self.active_profile_id))
        self.workbench.configure_log_viewer(self.active_profile_id, self._active_log_db_path())
        self.workbench.inspector.inspect(None)
        self._configure_trigger_action()
        self._apply_active_profile_camera_settings()
        self._sync_roi_overlays()
        self._warm_active_profile_ocr()

    def _save_active_profile(self, silent: bool = False) -> None:
        self.profile_store.save_profile(
            self.active_profile_id,
            self.graph,
            indicator=self.workbench.result_indicator.settings(),
            stats=self.workbench.production_stats.settings(),
            log_viewer=self.workbench.production_log_viewer.settings(),
        )
        if not silent:
            self.statusBar().showMessage(
                tr(
                    "status.saved_profile",
                    profile=self.profile_store.profile_name(self.active_profile_id),
                    path=self.profile_store.path,
                )
            )

    def _new_profile(self) -> None:
        name, accepted = QInputDialog.getText(self, tr("dialog.new_profile.title"), tr("dialog.new_profile.label"))
        if not accepted:
            return
        name = name.strip()
        if not name:
            self.statusBar().showMessage(tr("status.profile_empty"))
            return
        self._save_active_profile(silent=True)
        self.active_profile_id = self.profile_store.create_profile(
            name,
            self.graph,
            indicator=self.workbench.result_indicator.settings(),
            stats=self.workbench.production_stats.settings(),
            log_viewer=self.workbench.production_log_viewer.settings(),
        )
        self._populate_profiles()
        self.statusBar().showMessage(tr("status.created_profile", profile=name))

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
        self.statusBar().showMessage(tr("status.loaded_profile", profile=self.profile_store.profile_name(profile_id)))

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
        self._update_fps(snapshot)

    def _start_camera_if_available(self) -> None:
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            self.workbench.set_camera_status("SYNTH")
            self.statusBar().showMessage(tr("status.offscreen_camera_skipped"))
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
        self.statusBar().showMessage(tr("status.retry_camera"))
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
        self._update_fps(snapshot)
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
            self.statusBar().showMessage(tr("status.add_roi_first"))
            return
        if self.graph.nodes[node_id].type_name != "roi":
            self.statusBar().showMessage(tr("status.select_roi_node"))
            return
        self._roi_target_node_id = node_id
        self.state.overlay_mode = OverlayMode.ROI
        self.workbench.apply_overlay_mode(self.state.overlay_mode)
        self.workbench.camera.set_roi_selection_enabled(True)
        self.statusBar().showMessage(tr("status.selecting_roi", node=node_label(self.graph.nodes[node_id], self.graph.definitions[self.graph.nodes[node_id].type_name].title)))

    def _roi_selected(self, roi: tuple[int, int, int, int]) -> None:
        self.workbench.camera.set_roi_selection_enabled(False)
        node_id = self._roi_target_node_id or self._target_roi_node_id()
        self._roi_target_node_id = None
        if node_id is None:
            self.statusBar().showMessage(tr("status.no_roi"))
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
        self.statusBar().showMessage(tr("status.roi_changed", node=node_label(node, self.graph.definitions[node.type_name].title), x=x, y=y, width=width, height=height))

    def _active_camera_settings_node_params(self) -> dict | None:
        for node in self.graph.nodes.values():
            if node.type_name == "camera_settings":
                return node.params
        return None

    def _apply_active_profile_camera_settings(self) -> None:
        params = self._active_camera_settings_node_params()
        if params is not None:
            self._apply_camera_settings(params)

    def _warm_active_profile_ocr(self) -> None:
        ocr_params = [
            dict(node.params)
            for node in self.graph.nodes.values()
            if node.enabled and node.type_name == "ocr" and str(node.params.get("run_mode", "worker")) != "in_process"
        ]
        for params in ocr_params:
            threading.Thread(target=warm_paddle_worker, args=(params,), daemon=True).start()

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
            self.statusBar().showMessage(
                tr(
                    "status.camera_param_updated",
                    node=node_label(node, self.graph.definitions[node.type_name].title),
                    param=param_label(key),
                    profile=self.profile_store.profile_name(self.active_profile_id),
                )
            )
        elif node.type_name == "trigger_switch":
            self._configure_trigger_action()
            self.statusBar().showMessage(tr("status.trigger_updated", node=node_label(node, self.graph.definitions[node.type_name].title)))
        else:
            if node.type_name == "serial_generator" and key == "hold" and bool(value):
                self._sync_generator_hold_state(node_id)
            if node.type_name == "sqlite_log":
                self.workbench.configure_log_viewer(self.active_profile_id, self._active_log_db_path())
            self.statusBar().showMessage(
                tr("status.param_updated", node=node_label(node, self.graph.definitions[node.type_name].title), param=param_label(key))
            )

    def _inspector_node_enabled_changed(self, node_id: str, enabled: bool) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None:
            self.workbench.inspector.inspect(None)
            return
        self.graph.set_node_enabled(node_id, enabled)
        self.workbench.overlay.graph_scene.update_results()
        item = self.workbench.overlay.graph_scene.node_items.get(node_id)
        if item is not None:
            item.update()
        self._configure_trigger_action()
        self._sync_roi_overlays()
        self.workbench.inspector.inspect(node_id)
        state = tr("state.enabled") if enabled else tr("state.disabled")
        self.statusBar().showMessage(tr("status.node_enabled", node=node_label(node, self.graph.definitions[node.type_name].title), state=state))

    def _reset_serial_node_state(self, node_id: str) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None or node.type_name not in {"serial_continuity", "serial_generator"}:
            return
        kind = "continuity" if node.type_name == "serial_continuity" else "generator"
        if node.type_name == "serial_continuity":
            if self.latest_snapshot is None:
                self.statusBar().showMessage(tr("status.no_frame"))
                return
            if self.state.workflow_busy:
                self.statusBar().showMessage(tr("status.workflow_busy"))
                return
        try:
            removed = reset_serial_state(resolve_log_db_path(self._active_log_db_path()), self.active_profile_id, node_id, kind)
        except Exception as exc:
            self.statusBar().showMessage(tr("status.serial_reset_failed", node=node_label(node, self.graph.definitions[node.type_name].title), error=exc))
            return
        if node.type_name == "serial_continuity":
            self._start_serial_sample_run(node_id, removed)
            return
        self.graph.results.pop(node_id, None)
        self.workbench.overlay.graph_scene.update_results()
        self.workbench.inspector.refresh_result()
        self.statusBar().showMessage(
            tr(
                "status.serial_reset",
                node=node_label(node, self.graph.definitions[node.type_name].title),
                count=removed,
                profile=self.profile_store.profile_name(self.active_profile_id),
            )
        )

    def _sync_generator_hold_state(self, generator_id: str) -> None:
        generator = self.graph.nodes.get(generator_id)
        if generator is None or generator.type_name != "serial_generator":
            return
        db_path = resolve_log_db_path(self._active_log_db_path())
        for edge in self.graph.edges:
            target = self.graph.nodes.get(edge.target)
            if edge.source != generator_id or target is None or target.type_name != "serial_continuity":
                continue
            state = serial_state(db_path, self.active_profile_id, target.node_id, "continuity")
            if state is None:
                return
            save_serial_state(db_path, self.active_profile_id, generator_id, "generator", str(state["text"]), int(state["value"]))
            return

    def _start_serial_sample_run(self, node_id: str, reset_count: int) -> None:
        self._pending_serial_sample_node_id = node_id
        self.dispatch(WorkflowRequested(self.latest_snapshot, self.graph.revision))
        future = self.executor.submit(
            self.graph,
            self.latest_snapshot,
            {
                "_profile_id": self.active_profile_id,
                "_profile_name": self.profile_store.profile_name(self.active_profile_id),
                "_log_db_path": self._active_log_db_path(),
                "_serial_sample_node_id": node_id,
                "_suppress_sqlite_log": True,
            },
        )
        self._pending_serial_sample_future = future
        future.add_done_callback(lambda done: self.workflow_done.emit(done))
        self.statusBar().showMessage(
            tr(
                "status.serial_sample_submitted",
                node=node_label(self.graph.nodes[node_id], self.graph.definitions[self.graph.nodes[node_id].type_name].title),
                count=reset_count,
            )
        )

    def _indicator_settings_changed(self, settings: dict) -> None:
        profile = self.profile_store.data.setdefault("profiles", {}).get(self.active_profile_id)
        if isinstance(profile, dict):
            profile["indicator"] = settings

    def _stats_settings_changed(self, settings: dict) -> None:
        profile = self.profile_store.data.setdefault("profiles", {}).get(self.active_profile_id)
        if isinstance(profile, dict):
            profile["stats"] = settings

    def _log_viewer_settings_changed(self, settings: dict) -> None:
        profile = self.profile_store.data.setdefault("profiles", {}).get(self.active_profile_id)
        if isinstance(profile, dict):
            profile["log_viewer"] = settings
            self.profile_store.save()

    def _sync_stats_counts(self) -> None:
        counters = self.state.counters
        self.workbench.set_stats_counts(int(counters.get("ok", 0)), int(counters.get("ng", 0)))

    def _update_fps(self, snapshot: FrameSnapshot) -> None:
        timestamp = float(snapshot.timestamp_s)
        if self._fps_samples and timestamp <= self._fps_samples[-1][1]:
            self._fps_samples.clear()
            self._fps_ema = 0.0
        self._fps_samples.append((int(snapshot.frame_id), timestamp))
        while len(self._fps_samples) > 2 and timestamp - self._fps_samples[0][1] > 2.5:
            self._fps_samples.popleft()

        now = time.perf_counter()
        if now - self._fps_last_publish < 0.25 or len(self._fps_samples) < 2:
            return
        first_id, first_timestamp = self._fps_samples[0]
        last_id, last_timestamp = self._fps_samples[-1]
        span = last_timestamp - first_timestamp
        if span < 0.35:
            return
        frame_count = last_id - first_id if last_id > first_id else len(self._fps_samples) - 1
        measured = max(0.0, frame_count / max(1e-6, span))
        self._fps_ema = measured if self._fps_ema <= 0 else self._fps_ema * 0.75 + measured * 0.25
        self._fps_last_publish = now
        self.workbench.set_stats_fps(self._fps_ema)

    def _configure_trigger_action(self) -> None:
        node = self.graph.nodes.get("trigger")
        params = node.params if node is not None else {}
        source = str(params.get("source", "keyboard"))
        shortcut = str(params.get("shortcut", "Ctrl+Return"))
        armed = bool(params.get("armed", True)) and bool(node.enabled if node is not None else True)
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
            self.statusBar().showMessage(tr("status.trigger_disabled"))
            return
        if self.latest_snapshot is None:
            self.statusBar().showMessage(tr("status.no_frame"))
            return
        if self.state.workflow_busy:
            self.statusBar().showMessage(tr("status.workflow_busy"))
            return
        self.dispatch(WorkflowRequested(self.latest_snapshot, self.graph.revision))
        future = self.executor.submit(
            self.graph,
            self.latest_snapshot,
            {
                "_profile_id": self.active_profile_id,
                "_profile_name": self.profile_store.profile_name(self.active_profile_id),
                "_log_db_path": self._active_log_db_path(),
            },
        )
        future.add_done_callback(lambda done: self.workflow_done.emit(done))
        self.statusBar().showMessage(tr("status.workflow_submitted", frame=self.latest_snapshot.frame_id))

    def _workflow_finished(self, future) -> None:  # type: ignore[no-untyped-def]
        try:
            run: WorkflowRun = future.result()
        except Exception as exc:
            if future is self._pending_serial_sample_future:
                self._pending_serial_sample_future = None
                self._pending_serial_sample_node_id = None
            self.state.workflow_busy = False
            self.statusBar().showMessage(tr("status.workflow_failed", error=exc))
            return
        if future is self._pending_serial_sample_future:
            self._pending_serial_sample_future = None
            node_id = self._pending_serial_sample_node_id
            self._pending_serial_sample_node_id = None
            self.state.workflow_busy = False
            self.workbench.overlay.graph_scene.update_results()
            if node_id is not None:
                self._complete_serial_sample(node_id, run)
            return
        final_result = self._aggregate_result(run)
        passed = bool(final_result.get("passed", False))
        self.dispatch(WorkflowFinished(run.run_id, passed, final_result))
        self._sync_stats_counts()
        self.workbench.overlay.graph_scene.update_results()
        self.workbench.inspector.refresh_result()
        self._show_debug_popups(run)
        if self.workbench.production_log_viewer.isVisible():
            self.workbench.production_log_viewer.refresh()
        self.workbench.result_indicator.blink(passed)
        self.statusBar().showMessage(tr("status.workflow_done", decision=final_result.get("decision", "DONE"), frame=run.frame_id))

    def _complete_serial_sample(self, node_id: str, run: WorkflowRun) -> None:
        node = self.graph.nodes.get(node_id)
        result = run.results.get(node_id)
        if node is None or result is None:
            self.statusBar().showMessage(tr("status.serial_sample_failed", node=node_id, error="no result"))
            return
        value = result.value
        text = ""
        if isinstance(value, dict):
            text = str(value.get("text", ""))
        elif value is not None:
            text = str(value)
        text = text.replace("\n", "").strip()
        if not text:
            self.statusBar().showMessage(
                tr("status.serial_sample_failed", node=node_label(node, self.graph.definitions[node.type_name].title), error="empty text")
            )
            return
        old_start = int(node.params.get("segment_start", 0) or 0)
        old_end = int(node.params.get("segment_end", min(len(text), old_start + 1)) or min(len(text), old_start + 1))
        node.params["sample_text"] = text
        node.params["serial_length"] = len(text)
        node.params["segment_start"] = max(0, min(old_start, len(text)))
        node.params["segment_end"] = max(node.params["segment_start"], min(old_end, len(text)))
        self.graph.revision += 1
        self.workbench.overlay.graph_scene.update_results()
        self.workbench.inspector.inspect(node_id)
        self._save_active_profile(silent=True)
        self.statusBar().showMessage(
            tr(
                "status.serial_sample_latched",
                node=node_label(node, self.graph.definitions[node.type_name].title),
                text=text,
            )
        )

    def _active_log_db_path(self) -> str:
        for node in self.graph.nodes.values():
            if node.enabled and node.type_name == "sqlite_log":
                return str(node.params.get("db_path", ""))
        return ""

    def _aggregate_result(self, run: WorkflowRun) -> dict:
        for node_id, result in reversed(run.results.items()):
            node = self.graph.nodes.get(node_id)
            if node is not None and node.enabled and node.type_name == "aggregate" and isinstance(result.value, dict):
                return result.value
        return run.final_result

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
        shutdown_paddle_worker()
        super().closeEvent(event)
