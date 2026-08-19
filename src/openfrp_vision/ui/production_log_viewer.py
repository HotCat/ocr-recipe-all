from __future__ import annotations

import json
import html
from pathlib import Path
from typing import Any

from PySide6.QtCore import QDate, QEvent, QObject, QPoint, QRect, Qt, Signal, Slot
from PySide6.QtGui import QColor, QCursor, QPainter, QPen, QTextCharFormat
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCalendarWidget,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from openfrp_vision.core.i18n import tr
from openfrp_vision.core.production_log import (
    ProductionLogRow,
    month_day_counts,
    query_rows_for_day,
    resolve_log_db_path,
)


class ProductionLogViewer(QWidget):
    settings_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._drag_mode: str | None = None
        self._drag_start = QPoint()
        self._start_geometry = QRect()
        self._resize_margin = 10
        self._minimum_float_size = (640, 460)
        self._profile_id = ""
        self._db_path = resolve_log_db_path("")
        self._rows: list[ProductionLogRow] = []
        self._row_by_id: dict[int, ProductionLogRow] = {}
        self.setObjectName("productionLogViewer")
        self.setMinimumSize(*self._minimum_float_size)
        self.resize(820, 560)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(
            """
            QWidget#content {
                background: rgba(8, 10, 12, 238);
                border: 1px solid rgba(148, 163, 184, 82);
                border-radius: 7px;
            }
            QLabel {
                color: #e5edf4;
                font-weight: 600;
            }
            QLabel#meta, QLabel#monthSummary {
                color: #9fb0c2;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton {
                background: rgba(31, 45, 58, 230);
                border: 1px solid rgba(148, 163, 184, 100);
                border-radius: 4px;
                color: #f8fafc;
                padding: 5px 9px;
            }
            QSpinBox {
                background: rgba(255, 255, 255, 230);
                border: 1px solid rgba(15, 23, 42, 120);
                color: #0f172a;
            }
            QCalendarWidget {
                background: rgba(22, 22, 24, 235);
                color: #e5edf4;
                border: 1px solid rgba(148, 163, 184, 55);
                alternate-background-color: rgba(32, 32, 34, 235);
                selection-background-color: #d94b24;
                selection-color: #ffffff;
            }
            QCalendarWidget QWidget#qt_calendar_navigationbar {
                background: rgba(36, 36, 38, 245);
            }
            QCalendarWidget QToolButton {
                color: #f97316;
                background: transparent;
                border: none;
                font-weight: 700;
            }
            QCalendarWidget QMenu {
                background: #202124;
                color: #e5edf4;
            }
            QTreeWidget {
                background: rgba(15, 17, 20, 225);
                border: 1px solid rgba(148, 163, 184, 62);
                color: #dbe7f3;
                gridline-color: rgba(148, 163, 184, 38);
                selection-background-color: rgba(20, 150, 170, 170);
            }
            QTreeWidget::item {
                min-height: 23px;
            }
            QHeaderView::section {
                background: rgba(30, 32, 36, 235);
                color: #dbe7f3;
                border: none;
                padding: 5px;
            }
            QPlainTextEdit {
                background: rgba(2, 6, 12, 215);
                border: 1px solid rgba(148, 163, 184, 70);
                color: #dbe7f3;
                font-family: monospace;
                font-size: 11px;
            }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        self.content = QWidget()
        self.content.setObjectName("content")
        root.addWidget(self.content)

        layout = QVBoxLayout(self.content)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)

        self.header_bar = QWidget()
        self.header_bar.setObjectName("floatHeader")
        header = QHBoxLayout(self.header_bar)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        self.title = QLabel()
        self.meta = QLabel()
        self.meta.setObjectName("meta")
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(10, 5000)
        self.limit_spin.setSingleStep(50)
        self.limit_spin.setValue(1000)
        self.refresh_button = QPushButton()
        self.close_button = QPushButton("x")
        self.close_button.setFixedWidth(28)
        self.refresh_button.clicked.connect(self.refresh)
        self.close_button.clicked.connect(self.hide)
        header.addWidget(self.title)
        header.addWidget(self.meta, 1)
        header.addWidget(self.limit_spin)
        header.addWidget(self.refresh_button)
        header.addWidget(self.close_button)
        layout.addWidget(self.header_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(7)

        self.calendar = QCalendarWidget()
        self.calendar.setGridVisible(True)
        self.calendar.setVerticalHeaderFormat(QCalendarWidget.VerticalHeaderFormat.ISOWeekNumbers)
        self.calendar.selectionChanged.connect(self._calendar_selection_changed)
        self.calendar.currentPageChanged.connect(self._calendar_page_changed)
        left_layout.addWidget(self.calendar)

        self.month_summary = QLabel()
        self.month_summary.setObjectName("monthSummary")
        left_layout.addWidget(self.month_summary)

        self.records_caption = QLabel()
        self.records_caption.setObjectName("recordsCaption")
        left_layout.addWidget(self.records_caption)

        self.records = QTreeWidget()
        self.records.setHeaderLabels([tr("log.column.status_time"), tr("log.column.serial"), tr("log.column.profile"), tr("log.column.checks")])
        self.records.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.records.itemSelectionChanged.connect(self._record_selection_changed)
        self.records.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.records.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.records.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.records.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        left_layout.addWidget(self.records, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(7)
        self.record_title = QLabel()
        self.serial_label = QLabel()
        self.serial_label.setObjectName("serialText")
        self.serial_label.setTextFormat(Qt.TextFormat.RichText)
        self.serial_label.setWordWrap(True)
        self.record_meta = QLabel()
        self.record_meta.setObjectName("meta")
        self.record_meta.setWordWrap(True)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        right_layout.addWidget(self.record_title)
        right_layout.addWidget(self.serial_label)
        right_layout.addWidget(self.record_meta)
        right_layout.addWidget(self.detail, 1)
        splitter.addWidget(right)
        splitter.setSizes([430, 360])
        self.resize_handle = QWidget(self)
        self.resize_handle.setObjectName("resizeHandle")
        self.resize_handle.setFixedSize(22, 22)
        self.resize_handle.setCursor(QCursor(Qt.CursorShape.SizeFDiagCursor))
        self.resize_handle.setStyleSheet(
            """
            QWidget#resizeHandle {
                background: rgba(148, 163, 184, 38);
                border-top: 1px solid rgba(226, 232, 240, 64);
                border-left: 1px solid rgba(226, 232, 240, 64);
                border-bottom-right-radius: 7px;
            }
            """
        )
        self._move_handles = {self.header_bar, self.title, self.meta}
        for handle in self._move_handles:
            handle.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            handle.installEventFilter(self)
        self.resize_handle.installEventFilter(self)
        self.retranslate()

    def settings(self) -> dict[str, Any]:
        geometry = self.geometry()
        return {"geometry": [geometry.x(), geometry.y(), geometry.width(), geometry.height()]}

    def apply_settings(self, settings: dict[str, Any] | None) -> None:
        geometry = (settings or {}).get("geometry", [72, 120, 820, 560])
        if isinstance(geometry, list | tuple) and len(geometry) == 4:
            self.setGeometry(self._bounded_geometry(QRect(*(int(value) for value in geometry))))

    def configure(self, profile_id: str, db_path: str | Path | None = None) -> None:
        self._profile_id = profile_id
        self._db_path = resolve_log_db_path(str(db_path or "").strip())
        if self.isVisible():
            self.refresh()
        else:
            self._update_meta()

    def retranslate(self) -> None:
        self.title.setText(tr("log.title"))
        self.refresh_button.setText(tr("log.refresh"))
        self.records_caption.setText(tr("log.daily_records"))
        self.records.setHeaderLabels([tr("log.column.status_time"), tr("log.column.serial"), tr("log.column.profile"), tr("log.column.checks")])
        self._update_meta()
        self._update_empty_detail()

    def refresh(self) -> None:
        self._refresh_month_marks()
        self._load_day(self.calendar.selectedDate())

    def clamp_to_parent(self) -> None:
        self.setGeometry(self._bounded_geometry(self.geometry()))

    def avoid_rect(self, forbidden: QRect, margin: int = 12) -> bool:
        geometry = QRect(self.geometry())
        if not geometry.intersects(forbidden):
            return False
        parent = self.parentWidget()
        bounds = parent.rect().adjusted(8, 8, -8, -8) if parent is not None else QRect()
        target_top = forbidden.bottom() + max(1, int(margin))
        if bounds.isValid():
            available_height = bounds.bottom() - target_top + 1
            if available_height >= self.minimumHeight():
                geometry.setHeight(min(geometry.height(), available_height))
                geometry.moveTop(target_top)
            else:
                target_left = forbidden.right() + max(1, int(margin))
                available_width = bounds.right() - target_left + 1
                if available_width >= self.minimumWidth():
                    geometry.setWidth(min(geometry.width(), available_width))
                    geometry.moveLeft(target_left)
                else:
                    geometry.moveTop(target_top)
        else:
            geometry.moveTop(target_top)
        geometry = self._bounded_geometry(geometry)
        if geometry == self.geometry():
            return False
        self.setGeometry(geometry)
        self.settings_changed.emit(self.settings())
        return True

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(226, 232, 240, 42), 1.0))
        painter.setBrush(QColor(0, 0, 0, 70))
        painter.drawRoundedRect(self.rect().adjusted(4, 4, -4, -4), 9, 9)
        painter.end()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if hasattr(self, "resize_handle"):
            self._position_resize_handle()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        mode = self._hit_test(event.position().toPoint())
        if mode is None:
            super().mousePressEvent(event)
            return
        self._drag_mode = mode
        self._begin_drag(event.globalPosition().toPoint(), mode)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        position = event.position().toPoint()
        if self._drag_mode is None:
            self._update_cursor(self._hit_test(position))
            super().mouseMoveEvent(event)
            return
        self._drag_to(event.globalPosition().toPoint())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drag_mode is not None:
            self._finish_drag()
            self._update_cursor(self._hit_test(event.position().toPoint()))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drag_mode is None:
            self.unsetCursor()
        super().leaveEvent(event)

    @Slot()
    def _calendar_selection_changed(self) -> None:
        self._load_day(self.calendar.selectedDate())

    @Slot(int, int)
    def _calendar_page_changed(self, year: int, month: int) -> None:
        self._refresh_month_marks(year, month)

    def _refresh_month_marks(self, year: int | None = None, month: int | None = None) -> None:
        if year is None or month is None:
            year = self.calendar.yearShown()
            month = self.calendar.monthShown()
        default_format = QTextCharFormat()
        first = QDate(year, month, 1)
        for day in range(1, first.daysInMonth() + 1):
            self.calendar.setDateTextFormat(QDate(year, month, day), default_format)

        try:
            counts = month_day_counts(self._db_path, self._profile_id, year, month)
        except Exception as exc:
            self.month_summary.setText(str(exc))
            return

        total = sum(item["total"] for item in counts.values())
        ok = sum(item["ok"] for item in counts.values())
        ng = sum(item["ng"] for item in counts.values())
        self.month_summary.setText(f"{year:04d}-{month:02d}: {total} {tr('log.records')} | {tr('state.ok')} {ok} | {tr('state.ng')} {ng}")

        for date_text, info in counts.items():
            y, m, d = [int(part) for part in date_text.split("-")]
            fmt = QTextCharFormat()
            fmt.setFontWeight(700)
            if info["ng"]:
                fmt.setBackground(QColor(122, 36, 36))
                fmt.setForeground(QColor(255, 230, 230))
            else:
                fmt.setBackground(QColor(34, 92, 54))
                fmt.setForeground(QColor(230, 255, 235))
            self.calendar.setDateTextFormat(QDate(y, m, d), fmt)

    def _load_day(self, qdate: QDate) -> None:
        date_text = qdate.toString("yyyy-MM-dd")
        try:
            self._rows = query_rows_for_day(self._db_path, self._profile_id, date_text, self.limit_spin.value())
        except Exception as exc:
            self._rows = []
            self.records.clear()
            self.detail.setPlainText(str(exc))
            self._update_meta()
            return
        self._populate_records(date_text)
        self._update_meta()

    def _populate_records(self, date_text: str) -> None:
        self._row_by_id = {row.row_id: row for row in self._rows}
        self.records.clear()
        groups = {
            True: QTreeWidgetItem([tr("state.ok"), "", "", "0"]),
            False: QTreeWidgetItem([tr("state.ng"), "", "", "0"]),
        }
        for group in groups.values():
            group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.records.addTopLevelItem(group)

        for row in self._rows:
            time_text = row.created_at.replace("T", " ").split(" ")[-1].split("+")[0]
            item = QTreeWidgetItem([time_text, row.serial_text, row.profile_name or row.profile_id, str(self._check_count(row.checks_json))])
            item.setData(0, Qt.ItemDataRole.UserRole, row.row_id)
            item.setToolTip(0, f"#{row.row_id} {row.decision}")
            if row.passed:
                item.setForeground(0, QColor("#7dff91"))
            else:
                item.setForeground(0, QColor("#ff6b7f"))
            groups[row.passed].addChild(item)
            if row.serial_text:
                label = QLabel(self._serial_markup(row, compact=True))
                label.setTextFormat(Qt.TextFormat.RichText)
                label.setStyleSheet("background: transparent; padding-left: 3px;")
                self.records.setItemWidget(item, 1, label)

        for passed, group in groups.items():
            group.setExpanded(True)
            group.setText(3, str(group.childCount()))
            group.setForeground(0, QColor("#7dff91") if passed else QColor("#ff6b7f"))

        if self._rows:
            first_group = groups[False] if groups[False].childCount() else groups[True]
            self.records.setCurrentItem(first_group.child(0))
        else:
            self._update_empty_detail(date_text)

    def _record_selection_changed(self) -> None:
        selected = self.records.selectedItems()
        if not selected:
            return
        row_id = selected[0].data(0, Qt.ItemDataRole.UserRole)
        if row_id is None:
            return
        row = self._row_by_id.get(int(row_id))
        if row is None:
            return
        self._show_row_detail(row)

    def _show_row_detail(self, row: ProductionLogRow) -> None:
        status = tr("state.ok") if row.passed else tr("state.ng")
        checks = self._check_count(row.checks_json)
        frame = "" if row.frame_id is None else f" | {tr('log.column.frame')} {row.frame_id}"
        self.record_title.setText(f"{status} | {row.created_at.replace('T', ' ')}")
        self.serial_label.setText(self._serial_html(row))
        self.record_meta.setText(
            f"#{row.row_id} | {row.profile_name or row.profile_id} | {row.source_node} | "
            f"{checks} {tr('log.column.checks')}{frame}"
        )
        try:
            payload = json.loads(row.result_json)
            text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        except json.JSONDecodeError:
            text = row.result_json
        self.detail.setPlainText(text)

    def _update_empty_detail(self, date_text: str = "") -> None:
        self.record_title.setText(tr("log.empty"))
        self.serial_label.setText("")
        self.record_meta.setText(date_text)
        self.detail.setPlainText("")

    def _update_meta(self) -> None:
        self.meta.setText(f"{self._db_path}  {len(self._rows)}")

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if watched in self._move_handles:
            return self._handle_float_mouse_event(event, "move")
        if watched is self.resize_handle:
            return self._handle_float_mouse_event(event, "resize")
        return super().eventFilter(watched, event)

    def _handle_float_mouse_event(self, event: QEvent, mode: str) -> bool:
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._begin_drag(event.globalPosition().toPoint(), mode)
            event.accept()
            return True
        if event.type() == QEvent.Type.MouseMove and self._drag_mode == mode:
            self._drag_to(event.globalPosition().toPoint())
            event.accept()
            return True
        if event.type() == QEvent.Type.MouseButtonRelease and self._drag_mode == mode:
            self._finish_drag()
            event.accept()
            return True
        return False

    def _begin_drag(self, global_position: QPoint, mode: str) -> None:
        self._drag_mode = mode
        self._drag_start = global_position
        self._start_geometry = self.geometry()
        self.raise_()

    def _drag_to(self, global_position: QPoint) -> None:
        if self._drag_mode is None:
            return
        delta = global_position - self._drag_start
        geometry = QRect(self._start_geometry)
        if self._drag_mode == "move":
            geometry.moveTopLeft(self._start_geometry.topLeft() + delta)
        else:
            geometry.setWidth(max(self.minimumWidth(), self._start_geometry.width() + delta.x()))
            geometry.setHeight(max(self.minimumHeight(), self._start_geometry.height() + delta.y()))
        self.setGeometry(self._bounded_geometry(geometry))

    def _finish_drag(self) -> None:
        self._drag_mode = None
        self.settings_changed.emit(self.settings())

    def _position_resize_handle(self) -> None:
        self.resize_handle.move(self.width() - self.resize_handle.width() - 8, self.height() - self.resize_handle.height() - 8)
        self.resize_handle.raise_()

    def _hit_test(self, position: QPoint) -> str | None:
        if position.x() >= self.width() - self._resize_margin and position.y() >= self.height() - self._resize_margin:
            return "resize"
        if position.y() <= 44:
            child = self.childAt(position)
            if child in {self.refresh_button, self.limit_spin, self.close_button}:
                return None
            return "move"
        return None

    def _update_cursor(self, mode: str | None) -> None:
        if mode == "resize":
            self.setCursor(QCursor(Qt.CursorShape.SizeFDiagCursor))
        elif mode == "move":
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        else:
            self.unsetCursor()

    def _bounded_geometry(self, geometry: QRect) -> QRect:
        parent = self.parentWidget()
        if parent is None:
            return geometry
        bounds = parent.rect().adjusted(8, 8, -8, -8)
        geometry.setWidth(max(self._minimum_float_size[0], min(geometry.width(), bounds.width())))
        geometry.setHeight(max(self._minimum_float_size[1], min(geometry.height(), bounds.height())))
        if geometry.right() > bounds.right():
            geometry.moveRight(bounds.right())
        if geometry.bottom() > bounds.bottom():
            geometry.moveBottom(bounds.bottom())
        if geometry.left() < bounds.left():
            geometry.moveLeft(bounds.left())
        if geometry.top() < bounds.top():
            geometry.moveTop(bounds.top())
        return geometry

    def _check_count(self, checks_json: str) -> int:
        try:
            checks = json.loads(checks_json)
        except json.JSONDecodeError:
            return 0
        return len(checks) if isinstance(checks, list) else 0

    def _serial_html(self, row: ProductionLogRow) -> str:
        if not row.serial_text:
            return f"<span style='color:#94a3b8'>{html.escape(tr('log.serial'))}: -</span>"
        return f"<span style='color:#94a3b8'>{html.escape(tr('log.serial'))}: </span>{self._serial_markup(row, compact=False)}"

    def _serial_markup(self, row: ProductionLogRow, compact: bool) -> str:
        text = row.serial_text
        start = max(0, min(int(row.serial_start), len(text)))
        end = max(start, min(int(row.serial_end), len(text)))
        before = html.escape(text[:start])
        effective = html.escape(text[start:end])
        after = html.escape(text[end:])
        value = "" if compact or row.serial_value is None else f" / {row.serial_value}"
        return f"<span style='color:#dbeafe'>{before}</span><span style='background-color:#f59e0b;color:#111827;font-weight:700;padding:1px 3px'>{effective}</span><span style='color:#dbeafe'>{after}</span><span style='color:#94a3b8'>{html.escape(value)}</span>"
