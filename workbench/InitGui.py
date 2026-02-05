import builtins
import contextlib
import os

import FreeCAD as App
import FreeCADGui as Gui

# Global panel reference
mcp_panel = None


# SimpleMCPPanel defined in Activated method to avoid FreeCAD module reloading issues


class ConjureShowCommand:
    """Command to show the Conjure panel"""

    def GetResources(self):
        user_dir = App.getUserAppDataDir()
        icon_path = os.path.join(user_dir, "Mod", "conjure", "assets", "conjure_icon.png")
        return {
            "Pixmap": icon_path,
            "MenuText": "Show Conjure Panel",
            "ToolTip": "Show the Conjure - AI CAD Control panel",
        }

    def IsActive(self):
        return True

    def Activated(self):
        global mcp_panel
        print("INFO:Conjure_Gui:Opening Conjure Dashboard...")

        try:
            import json
            import queue
            import socket
            import threading
            import time
            import uuid as _uuid

            from PySide2 import QtCore, QtGui, QtWidgets

            class _NetworkSignals(QtCore.QObject):
                """Signal bridge for delivering results from worker thread to main thread."""

                result_ready = QtCore.Signal(str, object)  # (request_id, result_dict)

            class _NetworkWorker(QtCore.QThread):
                """Worker thread for all network I/O (socket commands, HTTP requests).

                All socket.connect/sendall/recv and urllib.request.urlopen calls
                run here, keeping the Qt main thread responsive.
                """

                def __init__(self):
                    super().__init__()
                    self.signals = _NetworkSignals()
                    self._queue = queue.Queue()
                    self._running = True

                def run(self):
                    while self._running:
                        try:
                            request_id, func, args, kwargs = self._queue.get(timeout=0.5)
                            try:
                                result = func(*args, **kwargs)
                            except Exception as e:
                                result = {"status": "error", "message": str(e)}
                            self.signals.result_ready.emit(request_id, result)
                        except queue.Empty:
                            continue
                        except Exception:
                            continue

                def submit(self, request_id, func, *args, **kwargs):
                    """Submit work to the network thread."""
                    self._queue.put((request_id, func, args, kwargs))

                def stop(self):
                    self._running = False
                    self.wait(2000)

            class MCPDashboard:
                """Comprehensive MCP Server Dashboard with monitoring and control"""

                # Brand color scheme — cyan / apricot on midnight
                COLORS = {
                    "primary": "#00CED1",  # Dark Turquoise (cyan)
                    "primary_dark": "#008B8B",  # Dark Cyan
                    "primary_warm": "#FBCEB1",  # Apricot
                    "success": "#10b981",  # Green
                    "warning": "#f59e0b",  # Amber
                    "danger": "#ef4444",  # Red
                    "bg_dark": "#0a0f14",  # Midnight
                    "bg_card": "#111a22",  # Dark card
                    "bg_light": "#f8fafc",  # Slate 50
                    "text_primary": "#f8fafc",  # Light text
                    "text_secondary": "#94a3b8",  # Muted text
                    "border": "#1c2b3a",  # Dark border
                }

                # Signal emitter for thread-safe UI updates
                class UpdateSignal:
                    def __init__(self):
                        self.callbacks = {}

                    def connect(self, name, callback):
                        self.callbacks[name] = callback

                    def emit(self, name, *args):
                        if name in self.callbacks:
                            self.callbacks[name](*args)

                def __init__(self):
                    # Create main widget with tabs
                    self.form = QtWidgets.QWidget()
                    self.form.setWindowTitle("Conjure Dashboard")
                    self.form.setMinimumSize(400, 300)
                    self.form.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

                    # Fix Windows redraw issues - ensure proper background painting
                    self.form.setAutoFillBackground(True)
                    self.form.setAttribute(QtCore.Qt.WA_StyledBackground, True)

                    # Apply modern stylesheet
                    self.apply_modern_style()

                    # Main layout
                    main_layout = QtWidgets.QVBoxLayout()
                    main_layout.setContentsMargins(0, 0, 0, 0)
                    main_layout.setSpacing(0)

                    # Header
                    header = self.create_header()
                    main_layout.addWidget(header)

                    # Content area with left sidebar navigation
                    content_widget = QtWidgets.QWidget()
                    content_layout = QtWidgets.QHBoxLayout()
                    content_layout.setSpacing(0)
                    content_layout.setContentsMargins(0, 0, 0, 0)

                    # Sidebar container with collapse button
                    self.sidebar_widget = QtWidgets.QWidget()
                    self.sidebar_collapsed = False
                    self.sidebar_expanded_width = 140
                    self.sidebar_collapsed_width = 50
                    sidebar_layout = QtWidgets.QVBoxLayout()
                    sidebar_layout.setContentsMargins(0, 0, 0, 0)
                    sidebar_layout.setSpacing(0)

                    # Collapse toggle button
                    self.collapse_btn = QtWidgets.QPushButton("◀")
                    self.collapse_btn.setFixedHeight(28)
                    self.collapse_btn.setToolTip("Collapse sidebar")
                    self.collapse_btn.clicked.connect(self.toggle_sidebar)
                    self.collapse_btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {self.COLORS["bg_card"]};
                            border: none;
                            border-bottom: 1px solid {self.COLORS["border"]};
                            color: {self.COLORS["text_secondary"]};
                            font-size: 12px;
                            padding: 4px;
                            text-align: center;
                            min-width: 0px;
                        }}
                        QPushButton:hover {{
                            background-color: {self.COLORS["border"]};
                        }}
                    """)
                    sidebar_layout.addWidget(self.collapse_btn)

                    # Left sidebar navigation
                    self.nav_list = QtWidgets.QListWidget()
                    self.nav_list.setSpacing(2)
                    self.nav_list.setIconSize(QtCore.QSize(20, 20))
                    self.nav_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
                    self.nav_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

                    # Navigation items with simple Unicode symbols (emojis don't render in Qt)
                    self.nav_items = [
                        ("◆", "Design", "Selected object & actions"),
                        ("◷", "History", "Timeline & snapshots"),
                        ("▤", "Analysis", "Geometry & validation"),
                        ("●", "Status", "Connections & feedback"),
                    ]
                    for icon, name, tooltip in self.nav_items:
                        item = QtWidgets.QListWidgetItem(f"{icon}  {name}")
                        item.setToolTip(tooltip)
                        item.setData(QtCore.Qt.UserRole, name)  # Store name for collapsed mode
                        item.setData(QtCore.Qt.UserRole + 1, icon)  # Store icon
                        self.nav_list.addItem(item)

                    # Select first item by default
                    self.nav_list.setCurrentRow(0)
                    sidebar_layout.addWidget(self.nav_list)

                    self.sidebar_widget.setLayout(sidebar_layout)
                    # Use min/max width instead of fixed to allow window resizing
                    self.sidebar_widget.setMinimumWidth(self.sidebar_collapsed_width)
                    self.sidebar_widget.setMaximumWidth(self.sidebar_expanded_width)
                    self.sidebar_widget.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

                    # Stacked widget for content panels
                    self.content_stack = QtWidgets.QStackedWidget()

                    # Create tab content widgets
                    self.design_tab = self.create_design_tab()
                    self.history_tab = self.create_history_tab()
                    self.analysis_tab = self.create_analysis_tab()
                    self.status_tab = self.create_status_tab()

                    # Add content widgets to stack
                    self.content_stack.addWidget(self.design_tab)
                    self.content_stack.addWidget(self.history_tab)
                    self.content_stack.addWidget(self.analysis_tab)
                    self.content_stack.addWidget(self.status_tab)

                    # Connect navigation to content switching
                    self.nav_list.currentRowChanged.connect(self.content_stack.setCurrentIndex)

                    # Add widgets with stretch factors for proper resizing
                    content_layout.addWidget(self.sidebar_widget, 0)  # No stretch - fixed width
                    content_layout.addWidget(self.content_stack, 1)  # Stretch to fill remaining space
                    self.content_stack.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                    content_widget.setLayout(content_layout)

                    main_layout.addWidget(content_widget, 1)  # Stretch content area
                    self.form.setLayout(main_layout)

                    # Initialize FreeCAD socket server for health checks
                    # Import ConjureServer from conjure module
                    self.socket_server = None
                    try:
                        import conjure

                        self.socket_server = conjure.ConjureServer()
                        self.socket_server.start()
                        print("INFO:Conjure_Gui:FreeCAD socket server started on localhost:9876")
                    except Exception as e:
                        print(f"WARNING:Conjure_Gui:Failed to start socket server: {e}")

                    # Network worker — all socket/HTTP I/O runs off the main thread
                    self._net_worker = _NetworkWorker()
                    self._net_callbacks = {}  # request_id -> callback
                    self._net_worker.signals.result_ready.connect(self._on_net_result)
                    self._net_worker.start()

                    # Timers for auto-refresh
                    self.timers = {}
                    self.setup_timers()

                    # Initial data load
                    self.refresh_all()

                    # Auto-connect to cloud if API key is configured
                    QtCore.QTimer.singleShot(500, self.auto_connect_cloud)

                    # Selection observer with 300ms debounce to batch rapid changes
                    class _SelectionObserver:
                        def __init__(self, dashboard):
                            self.dashboard = dashboard
                            self._debounce = QtCore.QTimer()
                            self._debounce.setSingleShot(True)
                            self._debounce.setInterval(300)
                            self._debounce.timeout.connect(self.dashboard._on_selection_changed)

                        def addSelection(self, doc, obj, sub, pos):
                            self._debounce.start()

                        def removeSelection(self, doc, obj, sub):
                            self._debounce.start()

                        def clearSelection(self, doc):
                            self._debounce.start()

                    self._selection_observer = _SelectionObserver(self)
                    Gui.Selection.addObserver(self._selection_observer)

                # ===== STYLING METHODS =====

                def apply_modern_style(self):
                    """Apply modern dark theme stylesheet"""
                    c = self.COLORS
                    style = f"""
                        QWidget {{
                            background-color: {c["bg_dark"]};
                            color: {c["text_primary"]};
                            font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
                            font-size: 13px;
                        }}
                        QListWidget {{
                            background-color: {c["bg_card"]};
                            border: none;
                            border-right: 1px solid {c["border"]};
                            outline: none;
                            padding: 8px 0;
                        }}
                        QListWidget::item {{
                            padding: 12px 16px;
                            border-radius: 0;
                            margin: 0;
                            color: {c["text_secondary"]};
                        }}
                        QListWidget::item:selected {{
                            background-color: {c["primary_dark"]};
                            color: {c["text_primary"]};
                            border-left: 3px solid {c["primary_warm"]};
                        }}
                        QListWidget::item:hover:!selected {{
                            background-color: {c["border"]};
                            color: {c["text_primary"]};
                        }}
                        QStackedWidget {{
                            background-color: {c["bg_dark"]};
                        }}
                        QGroupBox {{
                            background-color: {c["bg_card"]};
                            border: 1px solid {c["border"]};
                            border-radius: 10px;
                            margin-top: 14px;
                            padding: 6px 8px 8px 8px;
                            padding-top: 24px;
                            font-weight: 600;
                            color: {c["text_primary"]};
                        }}
                        QGroupBox::title {{
                            subcontrol-origin: margin;
                            subcontrol-position: top left;
                            left: 14px;
                            top: 6px;
                            padding: 2px 8px;
                            color: {c["primary"]};
                            font-size: 12px;
                            font-weight: 700;
                            letter-spacing: 0.5px;
                            text-transform: uppercase;
                        }}
                        QPushButton {{
                            background-color: {c["primary"]};
                            color: {c["text_primary"]};
                            border: none;
                            border-radius: 6px;
                            padding: 7px 20px;
                            font-weight: 600;
                            font-size: 12px;
                            min-width: 70px;
                            max-width: 180px;
                        }}
                        QPushButton:hover {{
                            background-color: {c["primary_warm"]};
                            color: {c["bg_dark"]};
                        }}
                        QPushButton:disabled {{
                            background-color: {c["border"]};
                            color: {c["text_secondary"]};
                        }}
                        QPushButton#dangerBtn {{
                            background-color: {c["danger"]};
                        }}
                        QPushButton#dangerBtn:hover {{
                            background-color: #dc2626;
                        }}
                        QPushButton#successBtn {{
                            background-color: {c["success"]};
                        }}
                        QPushButton#successBtn:hover {{
                            background-color: #059669;
                        }}
                        QLabel {{
                            color: {c["text_primary"]};
                        }}
                        QLabel#subtitle {{
                            color: {c["text_secondary"]};
                            font-size: 12px;
                        }}
                        QGroupBox QListWidget {{
                            background-color: rgba(255, 255, 255, 0.03);
                            border: 1px solid {c["border"]};
                            border-radius: 6px;
                            padding: 4px;
                        }}
                        QGroupBox QListWidget::item {{
                            padding: 7px 12px;
                            margin: 1px 0;
                            border-radius: 4px;
                            color: {c["text_primary"]};
                            font-size: 13px;
                        }}
                        QGroupBox QListWidget::item:selected {{
                            background-color: rgba(0, 206, 209, 0.3);
                            color: {c["text_primary"]};
                        }}
                        QGroupBox QListWidget::item:hover:!selected {{
                            background-color: rgba(255, 255, 255, 0.05);
                        }}
                        QProgressBar {{
                            background-color: rgba(255, 255, 255, 0.06);
                            border: none;
                            border-radius: 5px;
                            height: 10px;
                            text-align: center;
                        }}
                        QProgressBar::chunk {{
                            background-color: {c["primary"]};
                            border-radius: 5px;
                        }}
                        QTextEdit {{
                            background-color: {c["bg_card"]};
                            border: 1px solid {c["border"]};
                            border-radius: 6px;
                            color: {c["text_primary"]};
                            padding: 8px;
                        }}
                        QTableWidget {{
                            background-color: {c["bg_card"]};
                            border: 1px solid {c["border"]};
                            border-radius: 6px;
                            color: {c["text_primary"]};
                            gridline-color: {c["border"]};
                            alternate-background-color: rgba(255, 255, 255, 0.02);
                        }}
                        QTableWidget::item {{
                            padding: 10px 12px;
                            border-bottom: 1px solid {c["border"]};
                        }}
                        QTableWidget::item:selected {{
                            background-color: {c["primary"]};
                        }}
                        QHeaderView::section {{
                            background-color: {c["primary_dark"]};
                            color: {c["text_primary"]};
                            padding: 10px 12px;
                            border: none;
                            border-right: 1px solid {c["primary"]};
                            font-weight: 600;
                        }}
                        QHeaderView::section:last {{
                            border-right: none;
                        }}
                        QTreeWidget {{
                            background-color: {c["bg_card"]};
                            border: 1px solid {c["border"]};
                            border-radius: 6px;
                            color: {c["text_primary"]};
                            outline: none;
                        }}
                        QTreeWidget::item {{
                            padding: 6px 4px;
                            border-radius: 4px;
                        }}
                        QTreeWidget::item:selected {{
                            background-color: {c["primary"]};
                        }}
                        QTreeWidget::item:hover:!selected {{
                            background-color: {c["border"]};
                        }}
                        QScrollBar:vertical {{
                            background-color: {c["bg_dark"]};
                            width: 10px;
                            border-radius: 5px;
                            margin: 0;
                        }}
                        QScrollBar::handle:vertical {{
                            background-color: {c["border"]};
                            border-radius: 5px;
                            min-height: 30px;
                        }}
                        QScrollBar::handle:vertical:hover {{
                            background-color: {c["text_secondary"]};
                        }}
                        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                            height: 0px;
                        }}
                        QScrollBar:horizontal {{
                            background-color: {c["bg_dark"]};
                            height: 10px;
                            border-radius: 5px;
                            margin: 0;
                        }}
                        QScrollBar::handle:horizontal {{
                            background-color: {c["border"]};
                            border-radius: 5px;
                            min-width: 30px;
                        }}
                        QScrollBar::handle:horizontal:hover {{
                            background-color: {c["text_secondary"]};
                        }}
                        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                            width: 0px;
                        }}
                        QLineEdit {{
                            background-color: {c["bg_card"]};
                            border: 1px solid {c["border"]};
                            border-radius: 6px;
                            padding: 8px 12px;
                            color: {c["text_primary"]};
                        }}
                        QLineEdit:focus {{
                            border-color: {c["primary"]};
                        }}
                    """
                    self.form.setStyleSheet(style)

                def create_header(self):
                    """Create header with logo and title"""
                    header = QtWidgets.QWidget()
                    header.setMinimumHeight(50)
                    header.setMaximumHeight(70)
                    header.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                    header.setStyleSheet(f"""
                        background-color: {self.COLORS["bg_card"]};
                        border-bottom: 2px solid {self.COLORS["primary"]};
                    """)

                    layout = QtWidgets.QHBoxLayout()
                    layout.setContentsMargins(16, 0, 16, 0)

                    # Logo image
                    logo_label = QtWidgets.QLabel()
                    user_dir = App.getUserAppDataDir()
                    logo_path = os.path.join(user_dir, "Mod", "conjure", "assets", "conjure_logo.png")
                    if os.path.exists(logo_path):
                        pixmap = QtGui.QPixmap(logo_path)
                        if not pixmap.isNull():
                            logo_label.setPixmap(pixmap.scaledToHeight(40, QtCore.Qt.SmoothTransformation))
                            logo_label.setStyleSheet("background: transparent;")
                            layout.addWidget(logo_label)

                    # Title
                    title = QtWidgets.QLabel("Conjure")
                    title.setStyleSheet("""
                        font-size: 18px;
                        font-weight: 700;
                        color: #f8fafc;
                    """)
                    layout.addWidget(title)

                    # Subtitle
                    subtitle = QtWidgets.QLabel("AI-Powered CAD Assistant")
                    subtitle.setObjectName("subtitle")
                    subtitle.setStyleSheet("font-size: 12px; color: #94a3b8;")
                    layout.addWidget(subtitle)

                    layout.addStretch()

                    # Version badge
                    version = QtWidgets.QLabel("v0.1.0")
                    version.setStyleSheet(f"""
                        background-color: {self.COLORS["primary"]};
                        color: {self.COLORS["bg_dark"]};
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 11px;
                        font-weight: 600;
                    """)
                    layout.addWidget(version)

                    # Connection status dots with spacing
                    dot_spacer = QtWidgets.QWidget()
                    dot_spacer.setFixedWidth(8)
                    dot_spacer.setStyleSheet("background: transparent;")
                    layout.addWidget(dot_spacer)

                    self.header_bridge_dot = QtWidgets.QLabel("●")
                    self.header_bridge_dot.setToolTip("FreeCAD Bridge")
                    self.header_bridge_dot.setStyleSheet(
                        f"color: {self.COLORS['danger']}; font-size: 16px; padding: 0 3px; background: transparent;"
                    )
                    layout.addWidget(self.header_bridge_dot)

                    self.header_cloud_dot = QtWidgets.QLabel("●")
                    self.header_cloud_dot.setToolTip("Cloud Server")
                    self.header_cloud_dot.setStyleSheet(
                        f"color: {self.COLORS['danger']}; font-size: 16px; padding: 0 3px; background: transparent;"
                    )
                    layout.addWidget(self.header_cloud_dot)

                    header.setLayout(layout)
                    return header

                # ===== TAB CREATION METHODS =====

                def make_scrollable(self, content_widget):
                    """Wrap a widget in a scroll area"""
                    scroll = QtWidgets.QScrollArea()
                    scroll.setWidgetResizable(True)
                    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
                    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                    scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                    scroll.setWidget(content_widget)
                    return scroll

                def create_design_tab(self):
                    """Create Design tab - driven by selection observer."""
                    content = QtWidgets.QWidget()
                    layout = QtWidgets.QVBoxLayout()
                    layout.setContentsMargins(16, 12, 16, 16)
                    layout.setSpacing(8)

                    # Selected Object group
                    obj_group = QtWidgets.QGroupBox("Selected Object")
                    obj_layout = QtWidgets.QVBoxLayout()
                    obj_layout.setSpacing(12)
                    obj_layout.setContentsMargins(10, 24, 10, 10)

                    self.design_obj_name = QtWidgets.QLabel("No selection")
                    self.design_obj_name.setStyleSheet("font-size: 17px; font-weight: 700; padding: 4px 0;")
                    obj_layout.addWidget(self.design_obj_name)

                    self.design_obj_type = QtWidgets.QLabel("")
                    self.design_obj_type.setStyleSheet(
                        f"color: {self.COLORS['text_secondary']}; font-size: 12px; padding: 2px 0 6px 0;"
                    )
                    obj_layout.addWidget(self.design_obj_type)

                    # Editable properties table
                    self.design_props_table = QtWidgets.QTableWidget()
                    self.design_props_table.setColumnCount(2)
                    self.design_props_table.setHorizontalHeaderLabels(["Property", "Value"])
                    self.design_props_table.horizontalHeader().setStretchLastSection(True)
                    self.design_props_table.setMaximumHeight(200)
                    obj_layout.addWidget(self.design_props_table)

                    obj_group.setLayout(obj_layout)
                    layout.addWidget(obj_group)

                    # Quick Actions group
                    actions_group = QtWidgets.QGroupBox("Quick Actions")
                    self.design_actions_layout = QtWidgets.QGridLayout()
                    self.design_actions_layout.setSpacing(10)
                    self.design_actions_layout.setContentsMargins(10, 24, 10, 10)
                    actions_group.setLayout(self.design_actions_layout)
                    layout.addWidget(actions_group)

                    # Suggested Next Steps
                    suggest_group = QtWidgets.QGroupBox("Suggested Next Steps")
                    suggest_layout = QtWidgets.QVBoxLayout()
                    suggest_layout.setContentsMargins(10, 24, 10, 10)
                    self.design_suggestions = QtWidgets.QListWidget()
                    self.design_suggestions.setMaximumHeight(120)
                    self.design_suggestions.addItem("Select an object to see suggestions")
                    suggest_layout.addWidget(self.design_suggestions)
                    suggest_group.setLayout(suggest_layout)
                    layout.addWidget(suggest_group)

                    layout.addStretch()
                    content.setLayout(layout)
                    return self.make_scrollable(content)

                def create_history_tab(self):
                    """Create History tab with undo/redo and snapshots."""
                    content = QtWidgets.QWidget()
                    layout = QtWidgets.QVBoxLayout()
                    layout.setContentsMargins(16, 12, 16, 16)
                    layout.setSpacing(8)

                    # Undo / Redo group
                    undo_group = QtWidgets.QGroupBox("Undo / Redo")
                    undo_layout = QtWidgets.QHBoxLayout()
                    undo_layout.setContentsMargins(10, 24, 10, 10)
                    undo_layout.setSpacing(12)

                    undo_btn = QtWidgets.QPushButton("  Undo")
                    undo_btn.setFixedWidth(90)
                    undo_btn.setFixedHeight(32)
                    undo_btn.clicked.connect(lambda: self._do_undo_redo("undo"))
                    undo_layout.addWidget(undo_btn)

                    redo_btn = QtWidgets.QPushButton("  Redo")
                    redo_btn.setFixedWidth(90)
                    redo_btn.setFixedHeight(32)
                    redo_btn.clicked.connect(lambda: self._do_undo_redo("redo"))
                    undo_layout.addWidget(redo_btn)

                    self.history_undo_label = QtWidgets.QLabel("Undo: 0 | Redo: 0")
                    self.history_undo_label.setStyleSheet(f"color: {self.COLORS['text_secondary']};")
                    undo_layout.addWidget(self.history_undo_label)
                    undo_layout.addStretch()

                    undo_group.setLayout(undo_layout)
                    layout.addWidget(undo_group)

                    # Operation Timeline group
                    timeline_group = QtWidgets.QGroupBox("Operation Timeline")
                    timeline_layout = QtWidgets.QVBoxLayout()
                    timeline_layout.setContentsMargins(10, 24, 10, 10)
                    timeline_layout.setSpacing(12)

                    self.history_timeline = QtWidgets.QListWidget()
                    self.history_timeline.setMaximumHeight(200)
                    timeline_layout.addWidget(self.history_timeline)

                    timeline_btn_row = QtWidgets.QHBoxLayout()
                    refresh_btn = QtWidgets.QPushButton("Refresh")
                    refresh_btn.setFixedWidth(100)
                    refresh_btn.setFixedHeight(30)
                    refresh_btn.clicked.connect(self.refresh_history)
                    timeline_btn_row.addWidget(refresh_btn)
                    timeline_btn_row.addStretch()
                    timeline_layout.addLayout(timeline_btn_row)

                    timeline_group.setLayout(timeline_layout)
                    layout.addWidget(timeline_group)

                    # Snapshots group
                    snap_group = QtWidgets.QGroupBox("Snapshots")
                    snap_layout = QtWidgets.QVBoxLayout()
                    snap_layout.setContentsMargins(10, 24, 10, 10)
                    snap_layout.setSpacing(12)

                    self.history_snapshots = QtWidgets.QListWidget()
                    self.history_snapshots.setMaximumHeight(150)
                    snap_layout.addWidget(self.history_snapshots)

                    snap_btn_layout = QtWidgets.QHBoxLayout()
                    snap_btn_layout.setSpacing(10)
                    save_snap_btn = QtWidgets.QPushButton("Save Snapshot")
                    save_snap_btn.setObjectName("successBtn")
                    save_snap_btn.setFixedWidth(130)
                    save_snap_btn.setFixedHeight(32)
                    save_snap_btn.clicked.connect(self._save_snapshot)
                    snap_btn_layout.addWidget(save_snap_btn)

                    restore_snap_btn = QtWidgets.QPushButton("Restore")
                    restore_snap_btn.setFixedWidth(100)
                    restore_snap_btn.setFixedHeight(32)
                    restore_snap_btn.clicked.connect(self._restore_snapshot)
                    snap_btn_layout.addWidget(restore_snap_btn)
                    snap_btn_layout.addStretch()
                    snap_layout.addLayout(snap_btn_layout)

                    snap_group.setLayout(snap_layout)
                    layout.addWidget(snap_group)

                    layout.addStretch()
                    content.setLayout(layout)
                    return self.make_scrollable(content)

                def create_analysis_tab(self):
                    """Create Analysis tab - updates on selection change."""
                    content = QtWidgets.QWidget()
                    layout = QtWidgets.QVBoxLayout()
                    layout.setContentsMargins(16, 12, 16, 16)
                    layout.setSpacing(8)

                    # Mass Properties group
                    mass_group = QtWidgets.QGroupBox("Mass Properties")
                    mass_layout = QtWidgets.QVBoxLayout()
                    mass_layout.setSpacing(12)
                    mass_layout.setContentsMargins(10, 24, 10, 10)

                    prop_style = f"font-size: 13px; padding: 2px 0; color: {self.COLORS['text_primary']};"
                    self.analysis_volume = QtWidgets.QLabel("Volume: --")
                    self.analysis_volume.setStyleSheet(prop_style)
                    mass_layout.addWidget(self.analysis_volume)
                    self.analysis_area = QtWidgets.QLabel("Surface Area: --")
                    self.analysis_area.setStyleSheet(prop_style)
                    mass_layout.addWidget(self.analysis_area)
                    self.analysis_com = QtWidgets.QLabel("Center of Mass: --")
                    self.analysis_com.setStyleSheet(prop_style)
                    mass_layout.addWidget(self.analysis_com)
                    self.analysis_bbox = QtWidgets.QLabel("Bounding Box: --")
                    self.analysis_bbox.setStyleSheet(prop_style)
                    mass_layout.addWidget(self.analysis_bbox)

                    mass_group.setLayout(mass_layout)
                    layout.addWidget(mass_group)

                    # Material group
                    mat_group = QtWidgets.QGroupBox("Material")
                    mat_layout = QtWidgets.QVBoxLayout()
                    mat_layout.setContentsMargins(10, 24, 10, 10)
                    self.analysis_material = QtWidgets.QLabel("Material: --")
                    mat_layout.addWidget(self.analysis_material)
                    mat_group.setLayout(mat_layout)
                    layout.addWidget(mat_group)

                    # Geometry Validation group
                    valid_group = QtWidgets.QGroupBox("Geometry Validation")
                    valid_layout = QtWidgets.QVBoxLayout()
                    valid_layout.setSpacing(12)
                    valid_layout.setContentsMargins(10, 24, 10, 10)

                    self.analysis_status = QtWidgets.QLabel("Status: Not checked")
                    valid_layout.addWidget(self.analysis_status)

                    self.analysis_warnings = QtWidgets.QListWidget()
                    self.analysis_warnings.setMaximumHeight(120)
                    valid_layout.addWidget(self.analysis_warnings)

                    valid_btn_row = QtWidgets.QHBoxLayout()
                    validate_btn = QtWidgets.QPushButton("Run Validation")
                    validate_btn.setFixedWidth(130)
                    validate_btn.setFixedHeight(32)
                    validate_btn.clicked.connect(self._run_validation)
                    valid_btn_row.addWidget(validate_btn)
                    valid_btn_row.addStretch()
                    valid_layout.addLayout(valid_btn_row)

                    valid_group.setLayout(valid_layout)
                    layout.addWidget(valid_group)

                    layout.addStretch()
                    content.setLayout(layout)
                    return self.make_scrollable(content)

                def create_status_tab(self):
                    """Create Status tab - condensed connections, usage, recent ops, feedback."""
                    content = QtWidgets.QWidget()
                    layout = QtWidgets.QVBoxLayout()
                    layout.setContentsMargins(16, 12, 16, 16)
                    layout.setSpacing(8)

                    # Connections group
                    conn_group = QtWidgets.QGroupBox("Connections")
                    conn_layout = QtWidgets.QVBoxLayout()
                    conn_layout.setSpacing(12)
                    conn_layout.setContentsMargins(10, 24, 10, 10)

                    self.status_bridge = QtWidgets.QLabel("Bridge: ● Checking...")
                    self.status_bridge.setStyleSheet(
                        f"color: {self.COLORS['warning']}; font-size: 14px; font-weight: 600; padding: 4px 0;"
                    )
                    conn_layout.addWidget(self.status_bridge)

                    self.status_cloud = QtWidgets.QLabel("Cloud: ● Checking...")
                    self.status_cloud.setStyleSheet(
                        f"color: {self.COLORS['warning']}; font-size: 14px; font-weight: 600; padding: 4px 0;"
                    )
                    conn_layout.addWidget(self.status_cloud)

                    btn_row = QtWidgets.QHBoxLayout()
                    btn_row.setSpacing(10)
                    test_btn = QtWidgets.QPushButton("Test")
                    test_btn.setFixedWidth(90)
                    test_btn.setFixedHeight(30)
                    test_btn.clicked.connect(self.test_connection)
                    btn_row.addWidget(test_btn)
                    reload_btn = QtWidgets.QPushButton("Reload")
                    reload_btn.setFixedWidth(90)
                    reload_btn.setFixedHeight(30)
                    reload_btn.clicked.connect(self.reload_bridge)
                    btn_row.addWidget(reload_btn)
                    btn_row.addStretch()
                    conn_layout.addLayout(btn_row)

                    conn_group.setLayout(conn_layout)
                    layout.addWidget(conn_group)

                    # Usage group
                    usage_group = QtWidgets.QGroupBox("Usage")
                    usage_layout = QtWidgets.QVBoxLayout()
                    usage_layout.setContentsMargins(10, 24, 10, 10)
                    usage_layout.setSpacing(8)

                    self.status_tier = QtWidgets.QLabel("Tier: Unknown")
                    self.status_tier.setStyleSheet("font-weight: 600; font-size: 13px; padding: 2px 0;")
                    usage_layout.addWidget(self.status_tier)

                    self.status_usage_bar = QtWidgets.QProgressBar()
                    self.status_usage_bar.setRange(0, 100)
                    usage_layout.addWidget(self.status_usage_bar)

                    self.status_usage_label = QtWidgets.QLabel("Used: 0 / 5000")
                    self.status_usage_label.setStyleSheet(f"color: {self.COLORS['text_secondary']};")
                    usage_layout.addWidget(self.status_usage_label)

                    usage_group.setLayout(usage_layout)
                    layout.addWidget(usage_group)

                    # Recent Operations group
                    recent_group = QtWidgets.QGroupBox("Recent Operations")
                    recent_layout = QtWidgets.QVBoxLayout()
                    recent_layout.setContentsMargins(10, 24, 10, 10)
                    self.status_recent_ops = QtWidgets.QListWidget()
                    self.status_recent_ops.setMaximumHeight(150)
                    recent_layout.addWidget(self.status_recent_ops)
                    recent_group.setLayout(recent_layout)
                    layout.addWidget(recent_group)

                    # Feedback group
                    feedback_group = QtWidgets.QGroupBox("Feedback")
                    feedback_layout = QtWidgets.QVBoxLayout()
                    feedback_layout.setContentsMargins(10, 24, 10, 10)
                    feedback_layout.setSpacing(12)

                    self.status_last_op = QtWidgets.QLabel("Last operation: --")
                    feedback_layout.addWidget(self.status_last_op)

                    fb_btn_row = QtWidgets.QHBoxLayout()
                    fb_btn_row.setSpacing(10)
                    worked_btn = QtWidgets.QPushButton("Worked")
                    worked_btn.setObjectName("successBtn")
                    worked_btn.setFixedWidth(100)
                    worked_btn.setFixedHeight(30)
                    worked_btn.clicked.connect(lambda: self._send_feedback(True))
                    fb_btn_row.addWidget(worked_btn)
                    failed_btn = QtWidgets.QPushButton("Failed")
                    failed_btn.setObjectName("dangerBtn")
                    failed_btn.setFixedWidth(100)
                    failed_btn.setFixedHeight(30)
                    failed_btn.clicked.connect(lambda: self._send_feedback(False))
                    fb_btn_row.addWidget(failed_btn)
                    fb_btn_row.addStretch()
                    feedback_layout.addLayout(fb_btn_row)

                    feedback_group.setLayout(feedback_layout)
                    layout.addWidget(feedback_group)

                    layout.addStretch()
                    content.setLayout(layout)
                    return self.make_scrollable(content)

                # ===== DATA COLLECTION METHODS =====

                # ===== ASYNC NETWORK I/O =====

                def _on_net_result(self, request_id, result):
                    """Dispatch worker-thread result to the registered callback (main thread)."""
                    callback = self._net_callbacks.pop(request_id, None)
                    if callback:
                        try:
                            callback(result)
                        except Exception as e:
                            print(f"ERROR:Conjure_Gui:Network callback error: {e}")

                def _send_async(self, command_dict, callback=None, timeout=3.0):
                    """Send a socket command without blocking the main thread.

                    Args:
                        command_dict: JSON-serialisable command.
                        callback:    Optional ``callback(result_dict)`` invoked on
                                     the main thread when the response arrives.
                        timeout:     Socket timeout in seconds.
                    """
                    rid = _uuid.uuid4().hex
                    if callback:
                        self._net_callbacks[rid] = callback
                    self._net_worker.submit(rid, self._do_socket_io, command_dict, timeout)

                @staticmethod
                def _do_socket_io(command_dict, timeout=3.0):
                    """Execute a socket command (runs on the worker thread)."""
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(timeout)
                        sock.connect(("localhost", 9876))
                        sock.sendall(json.dumps(command_dict).encode("utf-8") + b"\n")
                        response = sock.recv(8192).decode("utf-8")
                        sock.close()
                        return json.loads(response)
                    except socket.timeout:
                        return {"status": "error", "message": "Socket timeout"}
                    except Exception as e:
                        return {"status": "error", "message": str(e)}

                def _http_async(self, url, callback=None, timeout=2):
                    """Execute an HTTP GET without blocking the main thread."""
                    rid = _uuid.uuid4().hex
                    if callback:
                        self._net_callbacks[rid] = callback
                    self._net_worker.submit(rid, self._do_http_io, url, timeout)

                @staticmethod
                def _do_http_io(url, timeout=2):
                    """Execute an HTTP GET (runs on the worker thread)."""
                    import urllib.request

                    try:
                        req = urllib.request.Request(url, method="GET")
                        req.add_header("User-Agent", "Conjure-FreeCAD/0.1.0")
                        with urllib.request.urlopen(req, timeout=timeout) as resp:
                            if resp.status == 200:
                                return {"ok": True, "status": resp.status}
                            return {"ok": False, "status": resp.status}
                    except Exception as e:
                        return {"ok": False, "error": str(e)}

                # ===== SELECTION OBSERVER HANDLER =====

                def _on_selection_changed(self):
                    """Called when FreeCAD selection changes - update Design and Analysis tabs."""
                    try:
                        self._update_design_tab()
                        self._update_analysis_tab()
                    except Exception as e:
                        print(f"ERROR:Conjure_Gui:Selection update error: {e}")

                def _update_design_tab(self):
                    """Update the Design tab with selected object info."""
                    selection = Gui.Selection.getSelection()
                    if not selection:
                        self.design_obj_name.setText("No selection")
                        self.design_obj_type.setText("")
                        self.design_props_table.setRowCount(0)
                        # Clear action buttons
                        while self.design_actions_layout.count():
                            child = self.design_actions_layout.takeAt(0)
                            if child.widget():
                                child.widget().deleteLater()
                        self.design_suggestions.clear()
                        self.design_suggestions.addItem("Select an object to see suggestions")
                        return

                    obj = selection[0]
                    self.design_obj_name.setText(obj.Label)
                    self.design_obj_type.setText(f"Type: {obj.TypeId}  |  Name: {obj.Name}")

                    # Populate properties table
                    skip_props = {"Shape", "Placement", "Label", "ExpressionEngine", "Visibility", "Label2"}
                    props = []
                    for prop in obj.PropertiesList:
                        if prop in skip_props:
                            continue
                        try:
                            val = getattr(obj, prop)
                            if isinstance(val, (int, float, str, bool)):
                                props.append((prop, str(val)))
                            elif hasattr(val, "Value"):
                                props.append((prop, str(val.Value)))
                        except Exception:
                            continue

                    self.design_props_table.setRowCount(len(props))
                    for row, (pname, pval) in enumerate(props):
                        self.design_props_table.setItem(row, 0, QtWidgets.QTableWidgetItem(pname))
                        self.design_props_table.setItem(row, 1, QtWidgets.QTableWidgetItem(pval))

                    # Populate quick action buttons
                    while self.design_actions_layout.count():
                        child = self.design_actions_layout.takeAt(0)
                        if child.widget():
                            child.widget().deleteLater()

                    has_shape = hasattr(obj, "Shape")
                    actions = []
                    if has_shape:
                        actions = [
                            ("Fillet", "create_fillet", {"object_name": obj.Name, "radius": 1}),
                            ("Chamfer", "create_chamfer", {"object_name": obj.Name, "size": 1}),
                            ("Shell", "shell_object", {"object_name": obj.Name, "thickness": 1}),
                            ("Copy", "copy_object", {"source": obj.Name}),
                            ("Delete", "delete_object", {"object_name": obj.Name}),
                        ]
                    for idx, (label, cmd, params) in enumerate(actions):
                        btn = QtWidgets.QPushButton(label)
                        btn.setMinimumHeight(28)
                        if label == "Delete":
                            btn.setObjectName("dangerBtn")
                        btn.clicked.connect(
                            lambda checked=False, c=cmd, p=params: self._send_async({"type": c, "params": p})
                        )
                        self.design_actions_layout.addWidget(btn, idx // 3, idx % 3)

                    # Suggestions placeholder
                    self.design_suggestions.clear()
                    if has_shape:
                        self.design_suggestions.addItem("Try adding fillets to smooth edges")
                        self.design_suggestions.addItem("Consider shelling for lighter parts")
                    else:
                        self.design_suggestions.addItem("Object has no shape data")

                def _update_analysis_tab(self):
                    """Update Analysis tab with selected object geometry info."""
                    selection = Gui.Selection.getSelection()
                    if not selection:
                        self.analysis_volume.setText("Volume: --")
                        self.analysis_area.setText("Surface Area: --")
                        self.analysis_com.setText("Center of Mass: --")
                        self.analysis_bbox.setText("Bounding Box: --")
                        self.analysis_material.setText("Material: --")
                        self.analysis_status.setText("Status: No selection")
                        self.analysis_warnings.clear()
                        return

                    obj = selection[0]
                    if hasattr(obj, "Shape"):
                        shape = obj.Shape
                        try:
                            self.analysis_volume.setText(f"Volume: {shape.Volume:.2f} mm³")
                        except Exception:
                            self.analysis_volume.setText("Volume: N/A")
                        try:
                            self.analysis_area.setText(f"Surface Area: {shape.Area:.2f} mm²")
                        except Exception:
                            self.analysis_area.setText("Surface Area: N/A")
                        try:
                            com = shape.CenterOfMass
                            self.analysis_com.setText(f"Center of Mass: ({com.x:.1f}, {com.y:.1f}, {com.z:.1f})")
                        except Exception:
                            self.analysis_com.setText("Center of Mass: N/A")
                        try:
                            bb = shape.BoundBox
                            self.analysis_bbox.setText(
                                f"Bounding Box: {bb.XLength:.1f} x {bb.YLength:.1f} x {bb.ZLength:.1f} mm"
                            )
                        except Exception:
                            self.analysis_bbox.setText("Bounding Box: N/A")
                    else:
                        self.analysis_volume.setText("Volume: No shape")
                        self.analysis_area.setText("Surface Area: No shape")
                        self.analysis_com.setText("Center of Mass: No shape")
                        self.analysis_bbox.setText("Bounding Box: No shape")

                    # Material info via socket (async — doesn't block selection)
                    self._send_async(
                        {"type": "get_object_engineering_material", "params": {"object_name": obj.Name}},
                        callback=self._on_material_result,
                    )

                def _on_material_result(self, result):
                    """Handle material lookup result (main thread)."""
                    try:
                        if result.get("status") == "success":
                            self.analysis_material.setText(f"Material: {result.get('material_name', 'None')}")
                        else:
                            self.analysis_material.setText("Material: None")
                    except Exception:
                        self.analysis_material.setText("Material: --")

                def _run_validation(self):
                    """Run geometry validation on selected object (async)."""
                    selection = Gui.Selection.getSelection()
                    if not selection:
                        self.analysis_status.setText("Status: No object selected")
                        return
                    obj = selection[0]
                    obj_name = obj.Name
                    self._send_async(
                        {"type": "validate_geometry", "params": {"object_name": obj_name}},
                        callback=lambda r, n=obj_name: self._on_validation_result(r, n),
                    )

                def _on_validation_result(self, result, obj_name):
                    """Handle validation result (main thread)."""
                    self.analysis_warnings.clear()
                    if result.get("status") == "success":
                        is_valid = result.get("valid", True)
                        if is_valid:
                            self.analysis_status.setText(f"Status: {obj_name} - Valid")
                            self.analysis_status.setStyleSheet(f"color: {self.COLORS['success']}; font-weight: 600;")
                        else:
                            self.analysis_status.setText(f"Status: {obj_name} - Issues found")
                            self.analysis_status.setStyleSheet(f"color: {self.COLORS['warning']}; font-weight: 600;")
                        for w in result.get("warnings", []):
                            self.analysis_warnings.addItem(f"⚠ {w}")
                        for e in result.get("errors", []):
                            self.analysis_warnings.addItem(f"✖ {e}")
                    else:
                        self.analysis_status.setText(f"Status: Error - {result.get('error', 'unknown')}")
                        self.analysis_status.setStyleSheet(f"color: {self.COLORS['danger']};")

                # ===== HISTORY TAB METHODS =====

                def _do_undo_redo(self, action):
                    """Execute undo or redo via socket (async), then refresh history."""
                    self._send_async({"type": action}, callback=lambda _r: self.refresh_history())

                def refresh_history(self):
                    """Refresh the History tab data (async)."""
                    self._send_async({"type": "get_history"}, callback=self._on_history_result)

                def _on_history_result(self, result):
                    """Handle get_history result, then chain snapshot fetch."""
                    try:
                        if result.get("status") == "success":
                            undo_c = result.get("undo_count", 0)
                            redo_c = result.get("redo_count", 0)
                            self.history_undo_label.setText(f"Undo: {undo_c} | Redo: {redo_c}")
                            self.history_timeline.clear()
                            for name in result.get("undo_names", []):
                                self.history_timeline.addItem(f"• {name}")
                            if not result.get("undo_names"):
                                self.history_timeline.addItem("  (no operations)")
                    except Exception as e:
                        print(f"ERROR:Conjure_Gui:Error refreshing history: {e}")
                    # Chain: fetch snapshots
                    self._send_async({"type": "list_snapshots"}, callback=self._on_snapshots_result)

                def _on_snapshots_result(self, result):
                    """Handle list_snapshots result (main thread)."""
                    try:
                        if result.get("status") == "success":
                            self.history_snapshots.clear()
                            for s in result.get("snapshots", []):
                                name = s.get("name", "unnamed")
                                created = s.get("created_at", "")
                                self.history_snapshots.addItem(f"{name}  ({created})")
                            if not result.get("snapshots"):
                                self.history_snapshots.addItem("  (no snapshots)")
                    except Exception as e:
                        print(f"ERROR:Conjure_Gui:Error refreshing snapshots: {e}")

                def _save_snapshot(self):
                    """Save a new snapshot with user-provided name (async)."""
                    name, ok = QtWidgets.QInputDialog.getText(self.form, "Save Snapshot", "Snapshot name:")
                    if ok and name.strip():
                        sname = name.strip()

                        def _on_saved(result):
                            if result.get("status") == "success":
                                self.log(f"Snapshot '{sname}' saved")
                            else:
                                self.log(f"Failed to save snapshot: {result.get('error', 'unknown')}")
                            self.refresh_history()

                        self._send_async(
                            {"type": "save_snapshot", "params": {"name": sname}},
                            callback=_on_saved,
                        )

                def _restore_snapshot(self):
                    """Restore the selected snapshot (async)."""
                    current = self.history_snapshots.currentItem()
                    if not current:
                        self.log("No snapshot selected")
                        return
                    name = current.text().strip().split("  (")[0]
                    if name.startswith("("):
                        return  # placeholder text

                    def _on_restored(result):
                        msg = result.get("message", result.get("error", "unknown"))
                        self.log(f"Restore: {msg}")
                        self.refresh_history()

                    self._send_async(
                        {"type": "restore_snapshot", "params": {"name": name}},
                        callback=_on_restored,
                    )

                # ===== STATUS TAB METHODS =====

                def _send_feedback(self, success):
                    """Send explicit feedback for the last operation (fire-and-forget)."""
                    self._send_async({"type": "record_feedback", "params": {"success": success}})
                    label = "positive" if success else "negative"
                    self.log(f"Feedback sent: {label}")

                def refresh_status(self):
                    """Refresh the Status tab — all I/O dispatched to worker thread."""
                    # Bridge health (socket)
                    self._send_async({"type": "health_check"}, callback=self._on_bridge_health)
                    # Usage info (socket)
                    self._send_async({"type": "get_usage"}, callback=self._on_usage_result)
                    # Cloud reachability
                    self._check_cloud_async()

                def _on_bridge_health(self, result):
                    """Handle bridge health check (main thread)."""
                    try:
                        if result.get("status") == "success":
                            self.status_bridge.setText("Bridge: ● Running")
                            self.status_bridge.setStyleSheet(f"color: {self.COLORS['success']}; font-weight: 600;")
                            self.header_bridge_dot.setStyleSheet(
                                f"color: {self.COLORS['success']}; font-size: 16px; padding: 0 3px; background: transparent;"
                            )
                            recent = result.get("recent_operations", [])
                            if recent:
                                last = recent[0]
                                status_str = "OK" if last["success"] else "FAIL"
                                self.status_last_op.setText(f"Last operation: {last['type']} ({status_str})")
                                self.status_recent_ops.clear()
                                for op in recent[:10]:
                                    icon = "✓" if op["success"] else "✗"
                                    self.status_recent_ops.addItem(f" {icon} {op['type']} ({op['duration_ms']:.0f}ms)")
                        else:
                            self.status_bridge.setText("Bridge: ● Not Running")
                            self.status_bridge.setStyleSheet(f"color: {self.COLORS['danger']}; font-weight: 600;")
                            self.header_bridge_dot.setStyleSheet(
                                f"color: {self.COLORS['danger']}; font-size: 16px; padding: 0 3px; background: transparent;"
                            )
                    except Exception as e:
                        print(f"ERROR:Conjure_Gui:Bridge health error: {e}")

                def _check_cloud_async(self):
                    """Check cloud connection — only the HTTP fallback is async."""
                    from conjure import get_cloud_bridge

                    bridge = get_cloud_bridge()
                    if bridge and bridge.running and bridge.adapter_id:
                        self.status_cloud.setText("Cloud: ● Connected")
                        self.status_cloud.setStyleSheet(f"color: {self.COLORS['success']}; font-weight: 600;")
                        self.header_cloud_dot.setStyleSheet(
                            f"color: {self.COLORS['success']}; font-size: 16px; padding: 0 3px; background: transparent;"
                        )
                    elif bridge and bridge.running:
                        self.status_cloud.setText("Cloud: ● Connecting...")
                        self.status_cloud.setStyleSheet(f"color: {self.COLORS['warning']}; font-weight: 600;")
                        self.header_cloud_dot.setStyleSheet(
                            f"color: {self.COLORS['warning']}; font-size: 16px; padding: 0 3px; background: transparent;"
                        )
                    else:
                        # HTTP reachability probe — runs on worker thread
                        self._http_async(
                            "https://conjure.lautrek.com/health",
                            callback=self._on_cloud_probe,
                            timeout=2,
                        )

                def _on_cloud_probe(self, result):
                    """Handle first cloud endpoint probe; fall back to localhost."""
                    if result.get("ok"):
                        self.status_cloud.setText("Cloud: ● Reachable (not bridged)")
                        self.status_cloud.setStyleSheet(f"color: {self.COLORS['warning']}; font-weight: 600;")
                        self.header_cloud_dot.setStyleSheet(
                            f"color: {self.COLORS['warning']}; font-size: 16px; padding: 0 3px; background: transparent;"
                        )
                    else:
                        self._http_async(
                            "http://localhost:8080/health",
                            callback=self._on_cloud_probe_localhost,
                            timeout=2,
                        )

                def _on_cloud_probe_localhost(self, result):
                    """Handle localhost cloud fallback probe."""
                    if result.get("ok"):
                        self.status_cloud.setText("Cloud: ● Reachable (not bridged)")
                        self.status_cloud.setStyleSheet(f"color: {self.COLORS['warning']}; font-weight: 600;")
                        self.header_cloud_dot.setStyleSheet(
                            f"color: {self.COLORS['warning']}; font-size: 16px; padding: 0 3px; background: transparent;"
                        )
                    else:
                        self.status_cloud.setText("Cloud: ● Unreachable")
                        self.status_cloud.setStyleSheet(f"color: {self.COLORS['danger']}; font-weight: 600;")
                        self.header_cloud_dot.setStyleSheet(
                            f"color: {self.COLORS['danger']}; font-size: 16px; padding: 0 3px; background: transparent;"
                        )

                def _on_usage_result(self, result):
                    """Handle get_usage result (main thread)."""
                    try:
                        if result.get("status") == "success":
                            tier = result.get("tier", "unknown").capitalize()
                            self.status_tier.setText(f"Tier: {tier}")
                            usage_data = result.get("usage", {})
                            used = usage_data.get("operations_recent", 0)
                            limit = usage_data.get("limit", 5000)
                            percent = int(used / limit * 100) if limit > 0 else 0
                            self.status_usage_bar.setValue(percent)
                            self.status_usage_label.setText(f"Used: {used} / {limit}")
                        else:
                            self.status_tier.setText("Tier: Unavailable")
                            self.status_usage_bar.setValue(0)
                            self.status_usage_label.setText("Usage data unavailable")
                    except Exception as e:
                        print(f"ERROR:Conjure_Gui:Usage update error: {e}")

                def load_api_key(self):
                    """Load API key from environment or config file."""
                    import os

                    # Check environment first
                    api_key = os.environ.get("CONJURE_API_KEY", "")
                    if api_key:
                        return api_key

                    # Check config file
                    try:
                        import yaml

                        config_paths = [
                            os.path.expanduser("~/.conjure/config.yaml"),
                            os.path.expanduser("~/.config/conjure/config.yaml"),
                        ]
                        for path in config_paths:
                            if os.path.exists(path):
                                with open(path) as f:
                                    config = yaml.safe_load(f)
                                    if config and config.get("hosted_server", {}).get("api_key"):
                                        return config["hosted_server"]["api_key"]
                    except Exception:
                        pass

                    return ""

                def load_server_url(self):
                    """Load server URL from env var, config file, or default."""
                    import os

                    # 1. Check environment variable first
                    env_url = os.environ.get("CONJURE_SERVER_URL", "")
                    if env_url:
                        # Convert HTTP URL to WebSocket URL
                        ws_url = env_url.replace("https://", "wss://").replace("http://", "ws://")
                        # Add WebSocket path if not present
                        if "/api/v1/adapter/ws" not in ws_url:
                            ws_url = ws_url.rstrip("/") + "/api/v1/adapter/ws"
                        return ws_url

                    # 2. Check config file
                    try:
                        import yaml

                        config_path = os.path.expanduser("~/.conjure/config.yaml")
                        if os.path.exists(config_path):
                            with open(config_path) as f:
                                config = yaml.safe_load(f)
                                if config and config.get("hosted_server", {}).get("ws_url"):
                                    return config["hosted_server"]["ws_url"]
                    except Exception:
                        pass

                    # 3. Default to production
                    return "wss://conjure.lautrek.com/api/v1/adapter/ws"

                def save_api_key(self, api_key, server_url=None):
                    """Save API key to config file."""
                    import os

                    config_dir = os.path.expanduser("~/.conjure")
                    config_path = os.path.join(config_dir, "config.yaml")

                    try:
                        import yaml

                        os.makedirs(config_dir, exist_ok=True)

                        # Load existing config or create new
                        config = {}
                        if os.path.exists(config_path):
                            with open(config_path) as f:
                                config = yaml.safe_load(f) or {}

                        # Update API key and server URL
                        if "hosted_server" not in config:
                            config["hosted_server"] = {}
                        config["hosted_server"]["api_key"] = api_key
                        if server_url:
                            config["hosted_server"]["ws_url"] = server_url
                            # Derive HTTP URL from WS URL
                            http_url = server_url.replace("wss://", "https://").replace("ws://", "http://")
                            http_url = http_url.split("/api/")[0]  # Strip path
                            config["hosted_server"]["url"] = http_url
                        config["hosted_server"]["enabled"] = True

                        # Save
                        with open(config_path, "w") as f:
                            yaml.dump(config, f, default_flow_style=False)

                        self.log(f"API key saved to {config_path}")
                    except Exception as e:
                        self.log(f"Failed to save API key: {e}")

                def toggle_cloud_bridge(self, force_connect=False):
                    """Connect or disconnect the cloud bridge."""
                    from conjure import get_cloud_bridge, start_cloud_bridge, stop_cloud_bridge

                    bridge = get_cloud_bridge()

                    if bridge and bridge.running and not force_connect:
                        # Disconnect
                        stop_cloud_bridge()
                        self.status_cloud.setText("Cloud: ● Disconnected")
                        self.status_cloud.setStyleSheet(f"color: {self.COLORS['danger']}; font-weight: 600;")
                        self.header_cloud_dot.setStyleSheet(
                            f"color: {self.COLORS['danger']}; font-size: 16px; padding: 0 3px; background: transparent;"
                        )
                        self.log("Cloud bridge disconnected")
                    else:
                        # Connect using env vars / config
                        api_key = self.load_api_key()
                        if not api_key:
                            self.status_cloud.setText("Cloud: ● No API Key")
                            self.status_cloud.setStyleSheet(f"color: {self.COLORS['danger']}; font-weight: 600;")
                            self.log("No API key configured. Set CONJURE_API_KEY env var or ~/.conjure/config.yaml")
                            return

                        server_url = self.load_server_url()

                        # Extract host for display
                        try:
                            from urllib.parse import urlparse

                            parsed = urlparse(server_url.replace("wss://", "https://").replace("ws://", "http://"))
                            display_host = parsed.netloc
                        except Exception:
                            display_host = server_url

                        try:
                            start_cloud_bridge(api_key, server_url)
                            self.status_cloud.setText(f"Cloud: ● Connecting to {display_host}...")
                            self.status_cloud.setStyleSheet(f"color: {self.COLORS['warning']}; font-weight: 600;")
                            self.header_cloud_dot.setStyleSheet(
                                f"color: {self.COLORS['warning']}; font-size: 16px; padding: 0 3px; background: transparent;"
                            )
                            self.log(f"Cloud bridge connecting to {display_host}...")
                        except Exception as e:
                            self.log(f"Error starting cloud bridge: {e}")

                def auto_connect_cloud(self):
                    """Auto-connect to cloud if API key is configured."""
                    api_key = self.load_api_key()
                    if api_key:
                        self.log("Auto-connecting to Conjure Cloud...")
                        self.toggle_cloud_bridge(force_connect=True)
                    else:
                        self.log("No API key found. Set CONJURE_API_KEY env var to auto-connect.")

                def refresh_all(self):
                    """Refresh all data"""
                    self.refresh_status()
                    self.refresh_history()

                def toggle_sidebar(self):
                    """Toggle sidebar between collapsed (icons only) and expanded (icons + text)"""
                    self.sidebar_collapsed = not self.sidebar_collapsed

                    if self.sidebar_collapsed:
                        # Collapse: show only icons, centered
                        self.sidebar_widget.setMinimumWidth(self.sidebar_collapsed_width)
                        self.sidebar_widget.setMaximumWidth(self.sidebar_collapsed_width)
                        self.collapse_btn.setText("▶")
                        self.collapse_btn.setToolTip("Expand sidebar")
                        # Update items to show only icons (centered)
                        for i in range(self.nav_list.count()):
                            item = self.nav_list.item(i)
                            icon = item.data(QtCore.Qt.UserRole + 1)
                            item.setText(icon)
                            item.setTextAlignment(QtCore.Qt.AlignCenter)
                    else:
                        # Expand: show icons + text, left aligned
                        self.sidebar_widget.setMinimumWidth(self.sidebar_collapsed_width)
                        self.sidebar_widget.setMaximumWidth(self.sidebar_expanded_width)
                        self.collapse_btn.setText("◀")
                        self.collapse_btn.setToolTip("Collapse sidebar")
                        # Update items to show icons + text (left aligned)
                        for i in range(self.nav_list.count()):
                            item = self.nav_list.item(i)
                            icon = item.data(QtCore.Qt.UserRole + 1)
                            name = item.data(QtCore.Qt.UserRole)
                            item.setText(f"{icon}  {name}")
                            item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

                def setup_timers(self):
                    """Setup auto-refresh timers (async I/O — safe at any interval)."""
                    # Health / connections — 30s (was 5s; I/O is now off-thread)
                    self.timers["health"] = QtCore.QTimer()
                    self.timers["health"].timeout.connect(self.refresh_status)
                    self.timers["health"].start(30000)

                    # History — 60s (was 10s; I/O is now off-thread)
                    self.timers["history"] = QtCore.QTimer()
                    self.timers["history"].timeout.connect(self.refresh_history)
                    self.timers["history"].start(60000)

                # ===== CONTROL ACTIONS =====

                def test_connection(self):
                    """Test connection to cloud server and bridge (async)."""
                    self.log("Testing connections...")
                    self._test_cloud_endpoint(0, time.time())
                    self._send_async({"type": "health_check"}, callback=self._on_test_bridge)

                def _test_cloud_endpoint(self, index, start):
                    """Walk through cloud endpoints sequentially via async HTTP."""
                    endpoints = [
                        ("https://conjure.lautrek.com/health", "conjure.lautrek.com"),
                        ("http://localhost:8080/health", "localhost:8080"),
                    ]
                    if index >= len(endpoints):
                        self.log("✗ Could not connect to cloud server")
                        return
                    url, endpoint = endpoints[index]
                    t0 = time.time()

                    def _on_result(result):
                        if result.get("ok"):
                            latency = int((time.time() - t0) * 1000)
                            self.log(f"✓ Cloud server OK: {endpoint} ({latency}ms)")
                        else:
                            self.log(f"✗ {endpoint}: {result.get('error', 'unknown')}")
                            self._test_cloud_endpoint(index + 1, start)

                    self._http_async(url, callback=_on_result, timeout=5)

                def _on_test_bridge(self, result):
                    """Handle bridge test result."""
                    if result.get("status") == "success":
                        self.log("✓ FreeCAD bridge OK")
                    else:
                        self.log(f"✗ FreeCAD bridge: {result.get('message', 'not responding')}")
                    self.refresh_status()

                def reload_bridge(self):
                    """Restart the internal socket server with module reload"""
                    import importlib
                    import socket

                    self.log("Restarting socket server...")

                    # Stop existing socket server
                    if self.socket_server:
                        try:
                            self.socket_server.stop()
                            # Wait for the server thread to finish
                            if hasattr(self.socket_server, "thread") and self.socket_server.thread:
                                self.socket_server.thread.join(timeout=2.0)
                            self.log("✓ Socket server stopped")
                        except Exception as e:
                            self.log(f"WARNING: Error stopping server: {e}")
                        self.socket_server = None

                    # Wait for port to be released and verify it's free
                    port_free = False
                    for _attempt in range(10):  # Try for up to 2 seconds
                        time.sleep(0.2)
                        try:
                            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            test_sock.bind(("localhost", 9876))
                            test_sock.close()
                            port_free = True
                            break
                        except OSError:
                            pass  # Port still in use

                    if not port_free:
                        self.log("WARNING: Port 9876 still in use, attempting restart anyway...")

                    # Reload the conjure module to pick up code changes
                    try:
                        import conjure

                        # Also reload dependent modules in src/
                        modules_to_reload = ["rendering", "validation", "geometry", "history"]
                        for mod_name in modules_to_reload:
                            try:
                                mod = __import__(mod_name)
                                importlib.reload(mod)
                                self.log(f"✓ Reloaded {mod_name}")
                            except Exception:
                                pass  # Module may not exist or not be loaded

                        # Reload main conjure module
                        importlib.reload(conjure)
                        self.log("✓ Reloaded conjure module")

                        # Create and start new server
                        self.socket_server = conjure.ConjureServer()
                        self.socket_server.start()
                        self.log("✓ Socket server restarted on localhost:9876")

                    except Exception as e:
                        self.log(f"ERROR: Failed to restart server: {e}")
                        import traceback

                        self.log(traceback.format_exc())

                    time.sleep(0.3)
                    self.refresh_status()

                def log(self, message):
                    """Log message to console"""
                    print(f"INFO:Conjure_Gui:{message}")

                # ===== HELPERS =====

                def escape_html(self, text):
                    """Escape HTML special characters"""
                    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                # ===== CLEANUP =====

                def getStandardButtons(self):
                    """Return standard buttons for the task panel"""
                    return int(QtWidgets.QDialogButtonBox.Close)

                def _cleanup(self):
                    """Shared cleanup for accept/reject."""
                    for timer in self.timers.values():
                        timer.stop()
                    if hasattr(self, "_selection_observer"):
                        Gui.Selection.removeObserver(self._selection_observer)
                    # Stop network worker thread
                    if hasattr(self, "_net_worker"):
                        self._net_worker.stop()
                    # Stop socket server if it's running
                    if self.socket_server:
                        with contextlib.suppress(builtins.BaseException):
                            self.socket_server.stop()

                def accept(self):
                    """Close the panel and cleanup"""
                    self._cleanup()
                    Gui.Control.closeDialog()

                def reject(self):
                    """Close the panel and cleanup"""
                    self._cleanup()
                    Gui.Control.closeDialog()

            # Create dashboard as a dockable widget instead of Task Panel
            mw = Gui.getMainWindow()
            if mw:
                # Check if dock already exists
                existing_dock = mw.findChild(QtWidgets.QDockWidget, "ConjureDashboard")
                if existing_dock:
                    existing_dock.show()
                    existing_dock.raise_()
                    print("INFO:Conjure_Gui:✓ Dashboard shown (existing)")
                else:
                    # Create new dock widget
                    mcp_panel = MCPDashboard()
                    dock = QtWidgets.QDockWidget("Conjure Dashboard", mw)
                    dock.setObjectName("ConjureDashboard")
                    dock.setWidget(mcp_panel.form)
                    dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
                    dock.setFeatures(
                        QtWidgets.QDockWidget.DockWidgetClosable
                        | QtWidgets.QDockWidget.DockWidgetMovable
                        | QtWidgets.QDockWidget.DockWidgetFloatable
                    )
                    mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
                    dock.show()
                    dock.raise_()
                    # Store reference to prevent garbage collection
                    mw._conjure_panel = mcp_panel
                    mw._conjure_dock = dock
                    print("INFO:Conjure_Gui:✓ Dashboard opened (dock widget)")
        except Exception as e:
            print(f"ERROR:Conjure_Gui:Failed to open dashboard: {e}")
            import traceback

            traceback.print_exc()


# Register the command
if not hasattr(Gui, "conjure_command"):
    Gui.conjure_command = ConjureShowCommand()
    Gui.addCommand("Conjure_Show", Gui.conjure_command)
    print("INFO:Conjure_Gui:✓ Registered Conjure command")


class ConjureWorkbench(Gui.Workbench):
    MenuText = "Conjure"
    ToolTip = "Conjure - AI CAD Control"

    Icon = os.path.join(App.getUserAppDataDir(), "Mod", "conjure", "assets", "conjure_icon.svg")

    def Initialize(self):
        """This function is called at workbench creation."""
        print("INFO:Conjure_Gui:Initializing Conjure Workbench")

        # Set up command list
        self.command_list = ["Conjure_Show"]

        try:
            self.appendMenu("&Conjure", self.command_list)
            print("INFO:Conjure_Gui:✓ Menu created")
        except:
            print("INFO:Conjure_Gui:Menu skipped (not available)")

        print("INFO:Conjure_Gui:✓ Workbench initialized")

    def Activated(self):
        """This function is called when the workbench is activated."""
        print("INFO:Conjure_Gui:✓ Conjure Workbench activated")
        print("INFO:Conjure_Gui:  MCP Agent is running as a background service")
        print("INFO:Conjure_Gui:  Ready to receive commands from Claude Code")
        from PySide2 import QtCore

        QtCore.QTimer.singleShot(200, lambda: Gui.runCommand("Conjure_Show"))

    def Deactivated(self):
        """This function is called when the workbench is deactivated."""
        print("INFO:Conjure_Gui:Conjure Workbench deactivated")

    def GetClassName(self):
        """Return the name of the associated C++ class."""
        return "Gui::PythonWorkbench"


# Add the workbench if it hasn't been added already
if not hasattr(Gui, "conjure_workbench"):
    try:
        Gui.conjure_workbench = ConjureWorkbench()
        Gui.addWorkbench(Gui.conjure_workbench)
        print("INFO:Conjure_Gui:✓ Registered Conjure Workbench")
    except Exception as e:
        print(f"ERROR:Conjure_Gui:Failed to register Conjure Workbench: {e}")
