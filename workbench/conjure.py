"""
Conjure FreeCAD Server - Thin Client

Socket server that runs inside FreeCAD and executes primitive CAD commands.
All orchestration and business logic happens on the hosted server.
This client only receives and executes low-level commands.

Supported command types:
- Primitives: create_box, create_cylinder, create_sphere, create_cone
- Booleans: boolean_fuse, boolean_cut, boolean_intersect
- Transforms: move_object, rotate_object, copy_object, delete_object
- Modifiers: create_fillet, create_chamfer
- Queries: get_state, get_object_details, list_objects, get_bounding_box
- Export: export_stl, export_step
- View: set_view, capture_view
- Script: run_script (escape hatch)
"""

import builtins
import contextlib
import json
import math
import os
import queue
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import FreeCAD as App

# Import logger for persistent file logging
try:
    from logger import (
        get_log_file_path,
        log_command,
        log_connection,
        log_crash,
        log_debug,
        log_error,
        log_exception,
        log_info,
        log_shutdown,
        log_signal_crash,
        log_startup,
        log_warning,
    )

    HAS_LOGGER = True
except ImportError:
    HAS_LOGGER = False

    # Fallback no-op functions
    def log_startup(*args, **kwargs):
        pass

    def log_shutdown(*args, **kwargs):
        pass

    def log_command(*args, **kwargs):
        pass

    def log_exception(*args, **kwargs):
        pass

    def log_crash(*args, **kwargs):
        pass

    def log_connection(*args, **kwargs):
        pass

    def log_info(*args, **kwargs):
        pass

    def log_warning(*args, **kwargs):
        pass

    def log_error(*args, **kwargs):
        pass

    def log_debug(*args, **kwargs):
        pass

    def get_log_file_path():
        return "N/A"


try:
    import FreeCADGui as Gui

    HAS_GUI = True
except ImportError:
    HAS_GUI = False

try:
    from PySide2 import QtCore

    HAS_QT = True
except ImportError:
    HAS_QT = False

# Add shared module to path for material client
_shared_path = Path(__file__).parent / "shared"
if str(_shared_path) not in sys.path:
    sys.path.insert(0, str(_shared_path))

try:
    from materials import MaterialsClient
except ImportError:
    MaterialsClient = None  # Will use fallback

try:
    from shared.standards import get_standards_library

    HAS_STANDARDS = True
except ImportError:
    HAS_STANDARDS = False

    def get_standards_library():
        return None


# Feedback client for orchestration learning
try:
    from shared.feedback import (
        FeedbackClient,
        get_feedback_client,
        init_feedback_client,
        stop_feedback_client,
    )

    HAS_FEEDBACK = True
except ImportError:
    HAS_FEEDBACK = False
    FeedbackClient = None

    def get_feedback_client():
        return None

    def init_feedback_client(*args, **kwargs):
        return None

    def stop_feedback_client():
        pass


# =============================================================================
# PLACEMENT OBSERVER - Notifies server when objects move
# =============================================================================


class PlacementObserver:
    """
    FreeCAD document observer that watches for Placement changes.

    When an object's placement changes, notifies the cloud server to propagate
    constraints through the dependency graph.
    """

    def __init__(self):
        self.enabled = False
        self._cloud_bridge = None
        self._last_positions: dict[str, tuple] = {}  # object_name -> (x, y, z)
        self._debounce_timer = None

    def set_cloud_bridge(self, bridge):
        """Set the cloud bridge for sending notifications."""
        self._cloud_bridge = bridge

    def slotChangedObject(self, obj, prop):
        """Called when any object property changes."""
        if not self.enabled or prop != "Placement":
            return

        if not hasattr(obj, "Placement"):
            return

        name = obj.Name
        new_pos = obj.Placement.Base
        new_tuple = (new_pos.x, new_pos.y, new_pos.z)

        old_tuple = self._last_positions.get(name)
        self._last_positions[name] = new_tuple

        # Skip if position didn't actually change (or first observation)
        if old_tuple is None or old_tuple == new_tuple:
            return

        # Notify server of placement change (debounced)
        self._notify_server(name, old_tuple, new_tuple)

    def _notify_server(self, object_name: str, old_pos: tuple, new_pos: tuple):
        """Send placement change notification to server."""
        if not self._cloud_bridge or not self._cloud_bridge.ws:
            return

        try:
            # Send propagate_changes command
            import json

            message = json.dumps(
                {
                    "type": "command",
                    "command": {
                        "type": "propagate_changes",
                        "params": {
                            "object_name": object_name,
                            "old_position": list(old_pos),
                            "new_position": list(new_pos),
                        },
                    },
                }
            )
            with self._cloud_bridge._ws_lock:
                self._cloud_bridge.ws.send(message)
            App.Console.PrintMessage(f"[Conjure] Propagating constraints for {object_name}\n")
        except Exception as e:
            App.Console.PrintWarning(f"[Conjure] Failed to propagate: {e}\n")

    def slotDeletedObject(self, obj):
        """Called when an object is deleted."""
        name = getattr(obj, "Name", None)
        if name and name in self._last_positions:
            del self._last_positions[name]


# Global placement observer instance
_placement_observer: PlacementObserver | None = None


def get_placement_observer() -> PlacementObserver:
    """Get or create the global placement observer."""
    global _placement_observer
    if _placement_observer is None:
        _placement_observer = PlacementObserver()
    return _placement_observer


def enable_constraint_propagation(enable: bool = True):
    """Enable or disable automatic constraint propagation on object moves."""
    observer = get_placement_observer()
    observer.enabled = enable

    if enable and App.ActiveDocument:
        # Register observer with active document
        App.ActiveDocument.addDocumentObserver(observer)
        App.Console.PrintMessage("[Conjure] Constraint propagation enabled\n")
    elif not enable and App.ActiveDocument:
        try:
            App.ActiveDocument.removeDocumentObserver(observer)
        except Exception:
            pass  # Observer may not be registered
        App.Console.PrintMessage("[Conjure] Constraint propagation disabled\n")


class ConjureServer:
    """Thin socket server for FreeCAD command execution."""

    def __init__(self, host="0.0.0.0", port=9876, server_url="https://conjure.lautrek.com"):
        self.host = host
        self.port = port
        self.server_url = server_url
        self.running = False
        self.socket = None
        self.client = None
        self.thread = None

        # Thread-safe operation queue
        self.operation_queue = queue.Queue()
        self.result_map = {}

        # Operation activity tracking
        self._current_op = None  # {"type": str, "start_time": float}
        self._recent_ops = []  # last 20 ops: [{type, duration_ms, success, timestamp}]

        # Materials client for engineering materials
        self.materials_client = None
        if MaterialsClient:
            try:
                self.materials_client = MaterialsClient(server_url)
            except Exception as e:
                App.Console.PrintWarning(f"Conjure: Failed to initialize materials client: {e}\n")

        # Feedback client for orchestration learning
        self.feedback_client = None
        self.feedback_enabled = True  # Can be disabled via config

        # In-memory assembly relationships store
        self._relationships = []

        # Start queue processor if Qt available
        self.process_timer = None
        if HAS_QT:
            try:
                self.process_timer = QtCore.QTimer()
                self.process_timer.timeout.connect(self._process_queue)
                self.process_timer.start(200)  # 5x/sec is sufficient
            except Exception as e:
                App.Console.PrintWarning(f"Queue processor failed: {e}\n")

    def start(self):
        """Start the server in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()
        App.Console.PrintMessage(f"Conjure server started on {self.host}:{self.port}\n")
        log_startup(self.port, self.server_url)
        if HAS_LOGGER:
            App.Console.PrintMessage(f"Conjure logs: {get_log_file_path()}\n")

        # Register fatal signal handlers for crash logging
        self._register_signal_handlers()

        # Start feedback client for orchestration learning
        if HAS_FEEDBACK and self.feedback_enabled:
            try:
                import uuid

                session_id = f"freecad_{uuid.uuid4().hex[:8]}"
                self.feedback_client = init_feedback_client(
                    server_url=self.server_url,
                    session_id=session_id,
                    enabled=True,
                )
                App.Console.PrintMessage("[Conjure] Feedback collection enabled\n")
            except Exception as e:
                App.Console.PrintWarning(f"[Conjure] Feedback client failed: {e}\n")
                self.feedback_client = None

    def _register_signal_handlers(self):
        """Register handlers for fatal signals to log crashes before process dies."""

        def _fatal_signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
            log_signal_crash(signum, sig_name)
            # Best-effort socket cleanup so port is released
            try:
                if self.client:
                    self.client.close()
                if self.socket:
                    self.socket.close()
            except Exception:
                pass
            # Re-raise with default handler so the OS produces a core dump if configured
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for sig in (signal.SIGSEGV, signal.SIGABRT):
            signal.signal(sig, _fatal_signal_handler)
        # SIGBUS doesn't exist on all platforms
        if hasattr(signal, "SIGBUS"):
            signal.signal(signal.SIGBUS, _fatal_signal_handler)
        log_info("Fatal signal handlers registered (SIGSEGV, SIGABRT, SIGBUS)")

    def stop(self):
        """Stop the server."""
        self.running = False
        if self.process_timer:
            self.process_timer.stop()
        if self.client:
            with contextlib.suppress(builtins.BaseException):
                self.client.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(builtins.BaseException):
                self.client.close()
        if self.socket:
            with contextlib.suppress(builtins.BaseException):
                self.socket.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(builtins.BaseException):
                self.socket.close()

        # Stop feedback client and flush remaining data
        if self.feedback_client:
            try:
                stats = self.feedback_client.get_stats()
                stop_feedback_client()
                App.Console.PrintMessage(f"[Conjure] Feedback: sent={stats['sent']}, buffered={stats['buffered']}\n")
            except Exception:
                pass
            self.feedback_client = None

        App.Console.PrintMessage("Conjure server stopped\n")
        log_shutdown()

    def _server_loop(self):
        """Main server loop - accepts connections and handles commands."""
        import struct

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Force immediate port release on close (no TIME_WAIT)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))

        # Retry bind in case of stale TIME_WAIT from a previous crash
        for attempt in range(3):
            try:
                self.socket.bind((self.host, self.port))
                break
            except OSError as e:
                if attempt < 2:
                    App.Console.PrintWarning(f"[Conjure] Port {self.port} busy, retrying in 2s... ({e})\n")
                    time.sleep(2)
                else:
                    App.Console.PrintError(f"[Conjure] Could not bind port {self.port}: {e}\n")
                    return

        self.socket.listen(1)
        self.socket.settimeout(1.0)

        while self.running:
            try:
                self.client, addr = self.socket.accept()
                App.Console.PrintMessage(f"Client connected from {addr}\n")
                log_connection("ACCEPT", f"from {addr}")
                self._handle_client()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    App.Console.PrintWarning(f"Server error: {e}\n")
                    log_exception("server_loop", e)

    # Read-only commands that don't modify FreeCAD documents - safe for direct execution
    READ_ONLY_COMMANDS = {
        "health_check",
        "get_state",
        "get_metrics",
        "get_usage",
        "get_config",
        "get_object_details",
        "list_objects",
        "get_bounding_box",
        "get_topology",
        "get_edges",
        "get_faces",
        "get_face_info",
        "find_face",
        "list_faces",
        "list_edges",
        "validate_geometry",
        "check_interference",
        "measure_distance",
        "measure_face_distance",
        "check_face_alignment",
        "get_relationships",
        "check_relationships",
        "detect_relationships",
        "suggest_relationships",
        "get_help",
        "list_standards",
        "search_standards",
        "get_standard",
        "get_gear",
        "list_gears",
        "get_gear_formulas",
        "get_material_design_guidelines",
        "calculate_thread_geometry",
        "calculate_polygon_geometry",
        "calculate_clearance_fit",
        "calculate_shrinkage_compensation",
        "get_arm_endpoint",
        "get_radial_positions",
        "get_socket_placement",
        "position_on_scalp",
        "get_all_electrode_positions",
        "reload_module",
        "check_all",
        "suggest_fixes",
        "measure_gap",
        "find_contacts",
        "search_properties",
        "capture_design_state",
        "sample_geometry",
        "sample_mesh_surface",
        "get_view_info",
    }

    def _handle_client(self):
        """Handle commands from connected client."""
        buffer = ""
        while self.running and self.client:
            try:
                data = self.client.recv(4096).decode("utf-8")
                if not data:
                    break

                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        # Parse command to check if it's read-only
                        try:
                            cmd = json.loads(line.strip())
                            cmd_type = cmd.get("type", "")
                            cmd_type = self.COMMAND_ALIASES.get(cmd_type, cmd_type)

                            if cmd_type in self.READ_ONLY_COMMANDS:
                                # Direct execution for read-only commands (faster)
                                response = self._execute_command(line.strip())
                            else:
                                # Queue write commands for main thread (thread-safe)
                                response = self._execute_command_queued(line.strip())
                        except json.JSONDecodeError:
                            response = self._execute_command_queued(line.strip())

                        self.client.send((json.dumps(response) + "\n").encode("utf-8"))
            except Exception as e:
                App.Console.PrintWarning(f"Client error: {e}\n")
                log_exception("handle_client", e)
                break

        if self.client:
            self.client.close()
            self.client = None
            log_connection("DISCONNECT", "client disconnected")

    # Command aliases for better discoverability
    COMMAND_ALIASES = {
        # Boolean operation aliases
        "boolean_union": "boolean_fuse",
        "union": "boolean_fuse",
        "fuse": "boolean_fuse",
        "boolean_subtract": "boolean_cut",
        "subtract": "boolean_cut",
        "cut": "boolean_cut",
        "boolean_difference": "boolean_cut",
        "difference": "boolean_cut",
        "intersect": "boolean_intersect",
        # Transform aliases
        "move": "move_object",
        "translate": "move_object",
        "rotate": "rotate_object",
        "copy": "copy_object",
        "delete": "delete_object",
        "remove": "delete_object",
        # Primitive aliases
        "box": "create_box",
        "cylinder": "create_cylinder",
        "sphere": "create_sphere",
        "cone": "create_cone",
        "torus": "create_torus",
        "wedge": "create_wedge",
        "prism": "create_prism",
        # Modifier aliases
        "fillet": "create_fillet",
        "chamfer": "create_chamfer",
        # High-level feature operation aliases
        "round_face": "fillet_face",
        "smooth_face": "fillet_face",
        "bevel_face": "chamfer_face",
        "hole": "create_hole",
        "drill": "create_hole",
        "pocket": "create_pocket",
        "recess": "create_pocket",
        # Smart placement aliases
        "align": "align_to_face",
        "snap_to": "align_to_face",
        "place_on": "place_on_face",
        "center": "center_on",
        # Analysis aliases
        "interference": "check_interference",
        "collides": "check_interference",
        "fit_check": "analyze_fit",
        "fits": "analyze_fit",
        # Query aliases
        "state": "get_state",
        "get_enhanced_state": "get_state",  # MCP bridge compatibility
        "bounding_box": "get_bounding_box",
        "bbox": "get_bounding_box",
        # Geometry reference aliases
        "topology": "get_topology",
        "edges": "get_edges",
        "faces": "get_faces",
        "face_info": "get_face_info",
        "find": "find_face",
        # Import/Export aliases
        "import_svg": "import_svg",
        # Document management aliases
        "clear": "clear_document",
        "reset": "clear_document",
        "new": "new_document",
        # Reference object aliases (list_references/get_reference_info are now SERVER-SIDE)
        "place_ref": "place_reference",
        # MCP bridge compatibility aliases (passthrough names -> existing handlers)
        "center_object": "center_on",
        "create_polygon_prism": "create_prism",
        "get_face": "find_face",
        "align_objects": "align_to_face",
    }

    # Commands that run frequently and should not spam the console
    _QUIET_COMMANDS = {"health_check", "get_state", "get_metrics", "get_usage", "get_config"}

    def _record_feedback(self, cmd_type: str, success: bool, duration_ms: float, error: str = None):
        """Record feedback for a command execution (non-blocking)."""
        if not self.feedback_client:
            return

        try:
            failure_category = None
            if not success and error:
                failure_category = self.feedback_client.categorize_failure(error)

            self.feedback_client.record(
                command_type=cmd_type,
                success=success,
                execution_time_ms=duration_ms,
                failure_category=failure_category,
            )
        except Exception:
            pass  # Don't let feedback recording break command execution

    def _record_activity(self, cmd_type, duration_ms, success):
        """Record operation in recent activity list for dashboard display."""
        self._recent_ops.insert(
            0,
            {
                "type": cmd_type,
                "duration_ms": round(duration_ms, 1),
                "success": success,
                "timestamp": time.time(),
            },
        )
        self._recent_ops = self._recent_ops[:20]

    def _execute_command(self, command_str):
        """Parse and execute a command."""
        start_time = time.time()
        cmd_type = "unknown"
        params = {}

        try:
            cmd = json.loads(command_str)
            cmd_type = cmd.get("type", "")
            params = cmd.get("params", {})

            # Resolve command aliases
            cmd_type = self.COMMAND_ALIASES.get(cmd_type, cmd_type)

            verbose = cmd_type not in self._QUIET_COMMANDS
            if verbose:
                App.Console.PrintMessage(f"[Conjure] >> {cmd_type}\n")
            self._current_op = {"type": cmd_type, "start_time": time.time()}

            # Route to handler
            handler = getattr(self, f"_cmd_{cmd_type}", None)
            if handler:
                # Wrap mutating commands in FreeCAD transactions for undo support
                _READ_ONLY = {
                    "get_state",
                    "get_object_details",
                    "get_bounding_box",
                    "find_objects",
                    "list_faces",
                    "list_edges",
                    "list_objects",
                    "get_topology",
                    "get_edges",
                    "get_faces",
                    "get_face_info",
                    "get_face",
                    "find_face",
                    "get_help",
                    "measure_distance",
                    "measure_face_distance",
                    "measure_gap",
                    "check_interference",
                    "check_face_alignment",
                    "check_relationships",
                    "validate_geometry",
                    "check_all",
                    "suggest_fixes",
                    "detect_relationships",
                    "suggest_relationships",
                    "get_relationships",
                    "list_standards",
                    "search_standards",
                    "get_standard",
                    "get_gear",
                    "list_gears",
                    "get_gear_formulas",
                    "get_material_design_guidelines",
                    "calculate_thread_geometry",
                    "calculate_polygon_geometry",
                    "calculate_clearance_fit",
                    "calculate_shrinkage_compensation",
                    "capture_view",
                    "set_view",
                    "get_history",
                    "undo",
                    "redo",
                    "search_properties",
                    "find_contacts",
                    "capture_design_state",
                    "sample_geometry",
                    "sample_mesh_surface",
                    "get_view_info",
                    "get_feature_tree",
                    "list_snapshots",
                    "eval_expression",
                }
                use_transaction = cmd_type not in _READ_ONLY
                doc = App.ActiveDocument
                if use_transaction and doc:
                    doc.openTransaction(cmd_type)
                try:
                    result = handler(params)
                except Exception:
                    if use_transaction and doc:
                        doc.abortTransaction()
                    raise
                if use_transaction and doc:
                    if result.get("status") == "success":
                        doc.commitTransaction()
                    else:
                        doc.abortTransaction()
                duration_ms = (time.time() - start_time) * 1000
                success = result.get("status") == "success"
                log_command(cmd_type, params, duration_ms, success, result.get("error"))
                # Record feedback for orchestration learning
                self._record_feedback(cmd_type, success, duration_ms, result.get("error"))
                if verbose:
                    if success:
                        App.Console.PrintMessage(f"[Conjure] << {cmd_type} OK ({duration_ms:.0f}ms)\n")
                    else:
                        App.Console.PrintWarning(f"[Conjure] << {cmd_type} FAIL ({duration_ms:.0f}ms)\n")
                self._record_activity(cmd_type, duration_ms, success)
                self._current_op = None
                return result
            else:
                duration_ms = (time.time() - start_time) * 1000
                error = f"Unknown command: {cmd_type}"
                log_command(cmd_type, params, duration_ms, False, error)
                self._record_feedback(cmd_type, False, duration_ms, error)
                if verbose:
                    App.Console.PrintWarning(f"[Conjure] << {cmd_type} FAIL ({duration_ms:.0f}ms)\n")
                self._record_activity(cmd_type, duration_ms, False)
                self._current_op = None
                return {"status": "error", "error": error}
        except json.JSONDecodeError as e:
            duration_ms = (time.time() - start_time) * 1000
            error = f"Invalid JSON: {e}"
            log_command(cmd_type, params, duration_ms, False, error)
            self._record_feedback(cmd_type, False, duration_ms, error)
            App.Console.PrintWarning(f"[Conjure] << {cmd_type} FAIL ({duration_ms:.0f}ms)\n")
            self._record_activity(cmd_type, duration_ms, False)
            self._current_op = None
            return {"status": "error", "error": error}
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_exception(f"command {cmd_type}", e)
            log_crash(f"command {cmd_type}", e, self._get_doc_state_for_logging())
            self._record_feedback(cmd_type, False, duration_ms, str(e))
            App.Console.PrintWarning(f"[Conjure] << {cmd_type} FAIL ({duration_ms:.0f}ms)\n")
            self._record_activity(cmd_type, duration_ms, False)
            self._current_op = None
            return {"status": "error", "error": str(e)}

    def _execute_command_threadsafe(self, command_str, timeout=30.0):
        """
        Execute command on main GUI thread via Qt.

        FreeCAD operations must run on the main thread to avoid crashes.
        Uses QTimer.singleShot for reliable cross-thread invocation.

        Args:
            command_str: JSON command string
            timeout: Maximum time to wait for result (seconds)

        Returns:
            Command result dict
        """
        import threading
        import uuid as uuid_mod

        op_id = str(uuid_mod.uuid4())
        result_event = threading.Event()

        def run_on_main_thread():
            try:
                result = self._execute_command(command_str)
                self.result_map[op_id] = {"result": result, "done": True}
            except Exception as e:
                self.result_map[op_id] = {"result": {"status": "error", "error": str(e)}, "done": True}
            result_event.set()

        # Use QTimer.singleShot to invoke on main thread
        if HAS_QT:
            QtCore.QTimer.singleShot(0, run_on_main_thread)
        else:
            # Fallback to direct execution if Qt not available
            run_on_main_thread()

        # Wait for result
        if result_event.wait(timeout):
            result = self.result_map.pop(op_id, {}).get("result", {"status": "error", "error": "No result"})
            return result
        else:
            self.result_map.pop(op_id, None)
            return {"status": "error", "error": f"Command timed out after {timeout}s"}

    def _get_doc_state_for_logging(self):
        """Get minimal document state for crash logging."""
        try:
            doc = App.ActiveDocument
            if doc:
                return {
                    "document": doc.Name,
                    "object_count": len(doc.Objects),
                }
        except Exception:
            pass
        return None

    def _execute_command_queued(self, command_str, timeout=30.0):
        """
        Execute command on main thread via operation queue.

        This is the preferred thread-safe method. Commands are queued and
        processed by _process_queue which runs on a QTimer on the main thread.

        Args:
            command_str: JSON command string
            timeout: Maximum time to wait for result (seconds)

        Returns:
            Command result dict
        """
        import uuid as uuid_mod

        op_id = str(uuid_mod.uuid4())
        log_debug(f"_execute_command_queued: queuing op {op_id[:8]} - {command_str[:80]}")

        # Queue the operation for main thread execution
        self.operation_queue.put(
            {
                "id": op_id,
                "func": self._execute_command,
                "args": [command_str],
                "kwargs": {},
            }
        )

        log_debug(f"_execute_command_queued: waiting for op {op_id[:8]}")

        # Wait for result with polling
        start_time = time.time()
        while time.time() - start_time < timeout:
            if op_id in self.result_map and self.result_map[op_id].get("done"):
                result = self.result_map.pop(op_id)["result"]
                log_debug(f"_execute_command_queued: got result for op {op_id[:8]}")
                return result
            time.sleep(0.01)  # 10ms polling interval

        # Timeout - clean up and return error
        log_error(f"_execute_command_queued: TIMEOUT for op {op_id[:8]}")
        self.result_map.pop(op_id, None)
        return {"status": "error", "error": f"Command timed out after {timeout}s"}

    def _process_queue(self):
        """Process operations on main thread (called by QTimer)."""
        try:
            while not self.operation_queue.empty():
                op = self.operation_queue.get_nowait()
                op_id = op["id"]
                log_debug(f"_process_queue: processing op {op_id[:8]}")
                try:
                    result = op["func"](*op.get("args", []), **op.get("kwargs", {}))
                    self.result_map[op_id] = {"result": result, "done": True}
                    log_debug(f"_process_queue: completed op {op_id[:8]}")
                except Exception as e:
                    log_error(f"_process_queue: error in op {op_id[:8]}: {e}")
                    self.result_map[op_id] = {"result": {"status": "error", "error": str(e)}, "done": True}
        except Exception as e:
            log_error(f"_process_queue: outer error: {e}")

    def execute_threadsafe(self, command_type, params, timeout=30.0):
        """
        Execute a command in a thread-safe manner.

        Queues the operation for main thread execution and waits for result.
        This must be used when calling from non-GUI threads (cloud bridge, etc).

        Args:
            command_type: Command type string (e.g., "create_box")
            params: Command parameters dict
            timeout: Maximum time to wait for result (seconds)

        Returns:
            Command result dict
        """
        import time
        import uuid

        op_id = str(uuid.uuid4())

        # Create operation for queue
        def execute_cmd():
            return self._execute_command(json.dumps({"type": command_type, "params": params}))

        self.operation_queue.put(
            {
                "id": op_id,
                "func": execute_cmd,
                "args": [],
                "kwargs": {},
            }
        )

        # Wait for result with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            if op_id in self.result_map and self.result_map[op_id].get("done"):
                result = self.result_map.pop(op_id)["result"]
                return result
            time.sleep(0.01)  # 10ms polling interval

        # Timeout - clean up and return error
        self.result_map.pop(op_id, None)
        return {"status": "error", "error": f"Command timed out after {timeout}s"}

    # ==================== Parameter Helpers ====================

    def _get_object_param(self, params, required=True):
        """
        Get object name from params with standardized naming.

        Supports: object_name (preferred), object, name (legacy)
        Returns tuple of (object_name, error_response or None)
        """
        # Preferred: object_name
        obj_name = params.get("object_name")

        # Fallback: object (used by some commands)
        if not obj_name:
            obj_name = params.get("object")

        # Legacy fallback: name (deprecated for input objects)
        if not obj_name:
            obj_name = params.get("name")

        if required and not obj_name:
            return None, {
                "status": "error",
                "error": "Missing required parameter: 'object_name'. Specify the object to operate on.",
            }

        return obj_name, None

    def _get_object(self, params, required=True):
        """
        Get FreeCAD object from params.

        Returns tuple of (object, error_response or None)
        """
        obj_name, error = self._get_object_param(params, required)
        if error:
            return None, error

        doc = self._get_doc()
        obj = doc.getObject(obj_name)

        if required and not obj:
            return None, {
                "status": "error",
                "error": f"Object '{obj_name}' not found in document.",
            }

        return obj, None

    def _require_params(self, params, *required_keys):
        """
        Check that required parameters are present.

        Returns error response dict if missing, None if all present.
        """
        missing = [k for k in required_keys if k not in params or params[k] is None]
        if missing:
            return {
                "status": "error",
                "error": f"Missing required parameter(s): {', '.join(missing)}",
            }
        return None

    # ==================== Status Commands ====================

    def _cmd_health_check(self, params):
        """Return health status with operation activity."""
        result = {
            "status": "success",
            "server": "conjure",
            "version": "0.1.0",
            "operations_count": len(self.result_map),
            "recent_operations": self._recent_ops[:5],
        }
        if self._current_op:
            result["current_operation"] = {
                "type": self._current_op["type"],
                "elapsed_s": round(time.time() - self._current_op["start_time"], 1),
            }
        return result

    def _cmd_get_metrics(self, params):
        """Return metrics data."""
        return {"status": "success", "metrics": {}}

    def _cmd_get_config(self, params):
        """Return config data."""
        return {"status": "success", "config": {}}

    def _cmd_get_usage(self, params):
        """Return usage stats."""
        return {"status": "success", "tier": "Free", "used": 0, "limit": 5000}

    def _cmd_reload_module(self, params):
        """Hot-reload the conjure module without FreeCAD restart.

        This is useful during development to pick up code changes without
        restarting FreeCAD. Note that some changes (global state, class
        instances) may not fully reload.

        Usage: Send {"type": "reload_module"} to the socket.
        """
        import importlib
        import sys

        reloaded_modules = []
        errors = []

        # List of modules to reload (in dependency order)
        modules_to_reload = [
            "standards",
            "materials",
            "conjure",
        ]

        for mod_name in modules_to_reload:
            if mod_name in sys.modules:
                try:
                    importlib.reload(sys.modules[mod_name])
                    reloaded_modules.append(mod_name)
                except Exception as e:
                    errors.append(f"{mod_name}: {str(e)}")

        if errors:
            return {
                "status": "partial",
                "message": f"Reloaded {len(reloaded_modules)} modules with {len(errors)} errors",
                "reloaded": reloaded_modules,
                "errors": errors,
            }

        return {
            "status": "success",
            "message": f"Reloaded {len(reloaded_modules)} modules",
            "reloaded": reloaded_modules,
        }

    def _cmd_clear_document(self, params):
        """Clear all objects from document, optionally keeping specified ones.

        Parameters:
            keep: list of object names to keep (optional)
            confirm: must be True to execute (safety check)
        """
        if not params.get("confirm", False):
            return {
                "status": "error",
                "error": "Safety check: set confirm=true to clear document",
            }

        doc = App.ActiveDocument
        if not doc:
            return {"status": "success", "deleted": 0, "message": "No active document"}

        keep_names = set(params.get("keep", []))
        deleted = []

        # Get all object names first (avoid modifying while iterating)
        all_names = [obj.Name for obj in doc.Objects]

        for name in all_names:
            if name not in keep_names:
                try:
                    doc.removeObject(name)
                    deleted.append(name)
                except Exception:
                    pass  # Object may have dependencies

        self._safe_recompute(doc)

        return {
            "status": "success",
            "deleted": len(deleted),
            "deleted_objects": deleted,
            "kept": list(keep_names) if keep_names else [],
        }

    def _cmd_new_document(self, params):
        """Create a new document, optionally closing the current one.

        Parameters:
            name: document name (default: "Untitled")
            close_current: close current document first (default: False)
        """
        name = params.get("name", "Untitled")
        close_current = params.get("close_current", False)

        if close_current and App.ActiveDocument:
            App.closeDocument(App.ActiveDocument.Name)

        doc = App.newDocument(name)
        return {"status": "success", "document": doc.Name}

    # ==================== Spreadsheet / Parametric ====================

    def _cmd_create_spreadsheet(self, params):
        """Create a Spreadsheet object for parametric design.

        Parameters:
            name: spreadsheet name (default: "Spreadsheet")
        """
        doc = self._get_doc()
        name = params.get("name", "Spreadsheet")
        sheet = doc.addObject("Spreadsheet::Sheet", name)
        self._safe_recompute(doc)
        return {"status": "success", "object": sheet.Name}

    def _cmd_set_cell(self, params):
        """Set a cell value in a Spreadsheet.

        Parameters:
            spreadsheet: name of the Spreadsheet object
            cell: cell address (e.g. "A1", "B2")
            value: value to set (number or string)
            alias: optional alias name for the cell
        """
        doc = self._get_doc()
        ss_name = params.get("spreadsheet")
        if not ss_name:
            return {"status": "error", "error": "spreadsheet parameter is required"}
        cell = params.get("cell")
        if not cell:
            return {"status": "error", "error": "cell parameter is required"}
        value = params.get("value")
        if value is None:
            return {"status": "error", "error": "value parameter is required"}

        obj = doc.getObject(ss_name)
        if obj is None:
            return {"status": "error", "error": f"Object '{ss_name}' not found"}
        if not hasattr(obj, "set"):
            return {"status": "error", "error": f"Object '{ss_name}' is not a Spreadsheet"}

        obj.set(cell, str(value))

        alias = params.get("alias")
        if alias:
            obj.setAlias(cell, alias)

        self._safe_recompute(doc)
        return {"status": "success", "spreadsheet": ss_name, "cell": cell, "value": value}

    def _cmd_set_expression(self, params):
        """Bind an object property to a spreadsheet expression.

        Parameters:
            object: name of the target object
            property: property name (e.g. "Height", "Radius")
            expression: expression string (e.g. "Params.A1", "Params.A1 * 2")
        """
        doc = self._get_doc()
        obj_name = params.get("object")
        if not obj_name:
            return {"status": "error", "error": "object parameter is required"}
        prop = params.get("property")
        if not prop:
            return {"status": "error", "error": "property parameter is required"}
        expression = params.get("expression")
        if not expression:
            return {"status": "error", "error": "expression parameter is required"}

        obj = doc.getObject(obj_name)
        if obj is None:
            return {"status": "error", "error": f"Object '{obj_name}' not found"}
        if not hasattr(obj, prop):
            return {"status": "error", "error": f"Object '{obj_name}' has no property '{prop}'"}

        obj.setExpression(prop, expression)
        self._safe_recompute(doc)

        return {
            "status": "success",
            "object": obj_name,
            "property": prop,
            "expression": expression,
        }

    def _cmd_get_expression(self, params):
        """Get the expression bound to an object property.

        Parameters:
            object: name of the target object
            property: property name (e.g. "Height", "Radius"); omit to list all
        """
        doc = self._get_doc()
        obj_name = params.get("object")
        if not obj_name:
            return {"status": "error", "error": "object parameter is required"}

        obj = doc.getObject(obj_name)
        if obj is None:
            return {"status": "error", "error": f"Object '{obj_name}' not found"}

        prop = params.get("property")
        if prop:
            # Return expression for a single property
            expressions = {p: e for p, e in obj.ExpressionEngine}
            expr = expressions.get(prop)
            return {
                "status": "success",
                "object": obj_name,
                "property": prop,
                "expression": expr,
                "value": getattr(obj, prop, None),
            }
        else:
            # Return all expressions on this object
            all_exprs = [{"property": p, "expression": e} for p, e in obj.ExpressionEngine]
            return {
                "status": "success",
                "object": obj_name,
                "expressions": all_exprs,
            }

    def _cmd_enable_propagation(self, params):
        """Enable or disable automatic constraint propagation.

        When enabled, moving an object automatically updates all constrained
        objects to maintain their relationships.

        Parameters:
            enable: True to enable, False to disable (default: True)
        """
        enable = params.get("enable", True)
        enable_constraint_propagation(enable)
        return {
            "status": "success",
            "propagation_enabled": enable,
            "message": f"Constraint propagation {'enabled' if enable else 'disabled'}",
        }

    # ==================== Reference Objects ====================
    # NOTE: Reference dimensions are stored on the SERVER (COMMON_ITEMS in ai/intent.py)
    # Client only handles FreeCAD-specific operations (creating geometry, setting visual props)
    # Workflow: AI calls server get_common_items → then calls client place_reference with dimensions

    def _cmd_place_reference(self, params):
        """Place a reference object in the workspace for designing around.

        This is a thin adapter - dimensions come from server's COMMON_ITEMS.
        AI should call server's get_common_items first to get dimensions.

        Parameters:
            name: object name (e.g., "REF_phone")
            width: width in mm (required)
            height: height in mm (required)
            thickness: thickness/depth in mm (required)
            position: [x, y, z] placement (default [0,0,0])
            angle: viewing angle in degrees (default 70)
            orientation: "portrait" or "landscape" (default portrait)
        """
        # Require dimensions - should come from server's COMMON_ITEMS
        width = params.get("width")
        height = params.get("height")
        thickness = params.get("thickness") or params.get("depth")

        if not all([width, height, thickness]):
            return {
                "status": "error",
                "error": "Missing dimensions. Use server's get_common_items to get device dimensions first.",
                "required": ["width", "height", "thickness"],
            }

        obj_name = params.get("name", "REF_object")
        orientation = params.get("orientation", "portrait")
        angle = params.get("angle", 70)  # Default viewing angle
        pos = params.get("position") or [0, 0, 0]

        # Swap dimensions for landscape
        if orientation == "landscape":
            width, height = height, width

        doc = self._get_doc()

        # Create the reference object as a box
        ref = doc.addObject("Part::Box", obj_name)
        ref.Length = width
        ref.Width = thickness
        ref.Height = height

        # Position and rotate to viewing angle
        ref.Placement.Base = App.Vector(pos[0], pos[1], pos[2])
        # Rotate around X axis for viewing angle (leaning back)
        # Negative angle so phone leans BACK (top away from viewer)
        ref.Placement.Rotation = App.Rotation(App.Vector(1, 0, 0), -(90 - angle))

        # Set visual properties to distinguish from design geometry
        if hasattr(ref, "ViewObject") and ref.ViewObject:
            ref.ViewObject.Transparency = 60
            ref.ViewObject.ShapeColor = (0.2, 0.6, 1.0)  # Light blue

        self._safe_recompute(doc)

        return {
            "status": "success",
            "object": obj_name,
            "dimensions": {"width": width, "height": height, "thickness": thickness},
            "angle": angle,
            "orientation": orientation,
            "tip": f"Design your stand around this {width}x{height}x{thickness}mm reference",
        }

    # ==================== Primitive Commands ====================

    def _cmd_create_box(self, params):
        """Create a box primitive."""
        doc = self._get_doc()
        name = params.get("name", "Box")
        box = doc.addObject("Part::Box", name)
        box.Length = params.get("length", 10)
        box.Width = params.get("width", 10)
        box.Height = params.get("height", 10)
        pos = params.get("position") or [0, 0, 0]
        box.Placement.Base = App.Vector(pos[0], pos[1], pos[2])
        self._safe_recompute(doc)
        return {"status": "success", "object": name}

    def _cmd_create_cylinder(self, params):
        """Create a cylinder."""
        doc = self._get_doc()
        name = params.get("name", "Cylinder")
        cyl = doc.addObject("Part::Cylinder", name)
        cyl.Radius = params.get("radius", 5)
        cyl.Height = params.get("height", 10)
        pos = params.get("position") or [0, 0, 0]
        cyl.Placement.Base = App.Vector(pos[0], pos[1], pos[2])
        self._safe_recompute(doc)
        return {"status": "success", "object": name}

    def _cmd_create_sphere(self, params):
        """Create a sphere."""
        doc = self._get_doc()
        name = params.get("name", "Sphere")
        sphere = doc.addObject("Part::Sphere", name)
        sphere.Radius = params.get("radius", 5)
        pos = params.get("position") or [0, 0, 0]
        sphere.Placement.Base = App.Vector(pos[0], pos[1], pos[2])
        self._safe_recompute(doc)
        return {"status": "success", "object": name}

    def _cmd_create_cone(self, params):
        """Create a cone."""
        doc = self._get_doc()
        name = params.get("name", "Cone")
        cone = doc.addObject("Part::Cone", name)
        cone.Radius1 = params.get("radius1", 5)
        cone.Radius2 = params.get("radius2", 0)
        cone.Height = params.get("height", 10)
        pos = params.get("position") or [0, 0, 0]
        cone.Placement.Base = App.Vector(pos[0], pos[1], pos[2])
        self._safe_recompute(doc)
        return {"status": "success", "object": name}

    def _cmd_create_torus(self, params):
        """Create a torus."""
        doc = self._get_doc()
        name = params.get("name", "Torus")
        torus = doc.addObject("Part::Torus", name)
        torus.Radius1 = params.get("radius1", 10)
        torus.Radius2 = params.get("radius2", 2)
        pos = params.get("position") or [0, 0, 0]
        torus.Placement.Base = App.Vector(pos[0], pos[1], pos[2])
        self._safe_recompute(doc)
        return {"status": "success", "object": name}

    def _cmd_create_wedge(self, params):
        """
        Create a wedge (angled block) - useful for sloped/angled surfaces.

        Parameters:
            name: Object name
            xmin, xmax: X range at base (default 0, 10)
            ymin, ymax: Y range (depth, default 0, 10)
            zmin, zmax: Z range (height, default 0, 10)
            x2min, x2max: X range at top (default same as base for rectangular wedge)
            position: [x, y, z] placement

        For a simple angled ramp, set x2min=x2max to create a triangular cross-section.
        """
        doc = self._get_doc()
        name = params.get("name", "Wedge")
        wedge = doc.addObject("Part::Wedge", name)

        # Base dimensions
        wedge.Xmin = params.get("xmin", 0)
        wedge.Xmax = params.get("xmax", 10)
        wedge.Ymin = params.get("ymin", 0)
        wedge.Ymax = params.get("ymax", 10)
        wedge.Zmin = params.get("zmin", 0)
        wedge.Zmax = params.get("zmax", 10)

        # Top dimensions (for angled/sloped top)
        wedge.X2min = params.get("x2min", params.get("xmin", 0))
        wedge.X2max = params.get("x2max", params.get("xmax", 10))
        wedge.Z2min = params.get("z2min", params.get("zmax", 10))
        wedge.Z2max = params.get("z2max", params.get("zmax", 10))

        pos = params.get("position") or [0, 0, 0]
        wedge.Placement.Base = App.Vector(pos[0], pos[1], pos[2])
        self._safe_recompute(doc)
        return {"status": "success", "object": name}

    def _cmd_create_prism(self, params):
        """
        Create a prism with specified polygon base and height.

        Parameters:
            name: Object name
            polygon: Number of sides (default 6 for hexagon)
            circumradius: Radius of circumscribed circle (default 5)
            height: Height of prism (default 10)
            position: [x, y, z] placement
        """
        doc = self._get_doc()
        name = params.get("name", "Prism")
        prism = doc.addObject("Part::Prism", name)
        prism.Polygon = params.get("polygon", 6)
        prism.Circumradius = params.get("circumradius", params.get("radius", 5))
        prism.Height = params.get("height", 10)
        pos = params.get("position") or [0, 0, 0]
        prism.Placement.Base = App.Vector(pos[0], pos[1], pos[2])
        self._safe_recompute(doc)
        return {"status": "success", "object": name}

    # ==================== Gear Commands ====================
    # NOTE: All gear geometry is calculated on the SERVER (cad/gear_generator.py)
    # Client only receives pre-computed points and creates the shape

    def _cmd_create_gear_from_profile(self, params):
        """Create spur gear from server-computed profile points.

        Server calculates involute geometry - client creates smooth B-spline profile.
        Uses B-spline interpolation for smooth tooth curves instead of polygon facets.
        """
        import Part

        doc = self._get_doc()
        name = params.get("name", "Gear")
        points = params.get("points", [])
        height = params.get("height", 10)
        bore_d = params.get("bore_diameter", 0)
        pos = params.get("position") or [0, 0, 0]
        use_bspline = params.get("smooth", False)  # Disabled - use high-res polygon

        if not points:
            return {"status": "error", "error": "No profile points provided"}

        try:
            # Convert 2D points to FreeCAD vectors
            vectors = [App.Vector(p[0], p[1], 0) for p in points]

            # Create profile wire - high-resolution polygon with proper involute points
            if use_bspline and len(vectors) > 4:
                # B-spline interpolation (experimental - currently disabled)
                closed_vectors = vectors + [vectors[0]]
                bspline = Part.BSplineCurve()
                bspline.interpolate(closed_vectors)
                wire = Part.Wire([bspline.toShape()])
            else:
                # High-resolution polygon from server-computed involute points
                vectors.append(vectors[0])  # Close profile
                wire = Part.makePolygon(vectors)

            face = Part.Face(wire)
            gear_shape = face.extrude(App.Vector(0, 0, height))

            # Add center bore if specified
            if bore_d > 0:
                bore = Part.makeCylinder(bore_d / 2, height + 2, App.Vector(0, 0, -1))
                gear_shape = gear_shape.cut(bore)

            gear_obj = doc.addObject("Part::Feature", name)
            gear_obj.Shape = gear_shape
            gear_obj.Placement.Base = App.Vector(pos[0], pos[1], pos[2])
            self._safe_recompute(doc)

            return {"status": "success", "object": name, "profile": "bspline" if use_bspline else "polygon"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_create_internal_gear_from_profile(self, params):
        """Create internal (ring) gear from server-computed profile points.

        Server calculates involute tooth space profile - client creates smooth B-spline
        profile and cuts it from cylinder for smooth internal gear teeth.
        """
        import Part

        doc = self._get_doc()
        name = params.get("name", "RingGear")
        points = params.get("points", [])
        height = params.get("height", 10)
        outer_d = params.get("outer_diameter", 100)
        pos = params.get("position") or [0, 0, 0]
        use_bspline = params.get("smooth", False)  # Disabled - use high-res polygon

        if not points:
            return {"status": "error", "error": "No profile points provided"}

        try:
            # Create outer cylinder
            outer_cyl = Part.makeCylinder(outer_d / 2, height)

            # Convert 2D points to FreeCAD vectors
            vectors = [App.Vector(p[0], p[1], 0) for p in points]

            # Create tooth space profile - high-resolution polygon
            if use_bspline and len(vectors) > 4:
                closed_vectors = vectors + [vectors[0]]
                bspline = Part.BSplineCurve()
                bspline.interpolate(closed_vectors)
                wire = Part.Wire([bspline.toShape()])
            else:
                vectors.append(vectors[0])  # Close profile
                wire = Part.makePolygon(vectors)

            face = Part.Face(wire)
            tooth_cut = face.extrude(App.Vector(0, 0, height + 2))
            tooth_cut.translate(App.Vector(0, 0, -1))

            # Cut teeth from cylinder
            ring_gear = outer_cyl.cut(tooth_cut)

            gear_obj = doc.addObject("Part::Feature", name)
            gear_obj.Shape = ring_gear
            gear_obj.Placement.Base = App.Vector(pos[0], pos[1], pos[2])
            self._safe_recompute(doc)

            return {"status": "success", "object": name, "profile": "bspline" if use_bspline else "polygon"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # Legacy gear commands - delegate to profile-based commands
    # These exist for backward compatibility until MCP bridge is updated

    def _cmd_create_gear(self, params):
        """Legacy: Create gear (generates profile locally for now)."""
        # TODO: Remove once MCP bridge uses server-side generator
        import Part

        self._get_doc()
        name = params.get("name", "Gear")
        m = params.get("module", 1.0)
        z = params.get("teeth", 20)
        height = params.get("face_width", 10)
        bore_d = params.get("bore_diameter", 0)
        pos = params.get("position") or [0, 0, 0]

        # Generate profile locally (temporary until server handles this)
        pitch_r = m * z / 2
        tip_r = pitch_r + 0.7 * m
        root_r = pitch_r - 0.9 * m
        angular_pitch = 2 * math.pi / z
        tooth_angle = angular_pitch * 0.48

        points = []
        for tooth in range(z):
            theta = tooth * angular_pitch
            points.extend(
                [
                    (root_r * math.cos(theta - angular_pitch * 0.02), root_r * math.sin(theta - angular_pitch * 0.02)),
                    (root_r * math.cos(theta + angular_pitch * 0.02), root_r * math.sin(theta + angular_pitch * 0.02)),
                    (tip_r * math.cos(theta + tooth_angle * 0.35), tip_r * math.sin(theta + tooth_angle * 0.35)),
                    (tip_r * math.cos(theta + angular_pitch * 0.25), tip_r * math.sin(theta + angular_pitch * 0.25)),
                    (
                        tip_r * math.cos(theta + angular_pitch * 0.5 - tooth_angle * 0.35),
                        tip_r * math.sin(theta + angular_pitch * 0.5 - tooth_angle * 0.35),
                    ),
                    (root_r * math.cos(theta + angular_pitch * 0.48), root_r * math.sin(theta + angular_pitch * 0.48)),
                ]
            )

        return self._cmd_create_gear_from_profile(
            {"name": name, "points": points, "height": height, "bore_diameter": bore_d, "position": pos}
        )

    def _cmd_create_internal_gear(self, params):
        """Legacy: Create internal gear (generates profile locally for now)."""
        # TODO: Remove once MCP bridge uses server-side generator
        import Part

        self._get_doc()
        name = params.get("name", "RingGear")
        m = params.get("module", 1.0)
        z = params.get("teeth", 48)
        height = params.get("face_width", 10)
        wall = params.get("wall_thickness", 5)
        pos = params.get("position") or [0, 0, 0]

        pitch_r = m * z / 2
        inner_tip_r = pitch_r - 0.7 * m
        inner_root_r = pitch_r + 0.9 * m
        outer_r = inner_root_r + wall
        angular_pitch = 2 * math.pi / z
        tooth_angle = angular_pitch * 0.48

        points = []
        for tooth in range(z):
            theta = tooth * angular_pitch
            points.extend(
                [
                    (
                        inner_root_r * math.cos(theta - angular_pitch * 0.02),
                        inner_root_r * math.sin(theta - angular_pitch * 0.02),
                    ),
                    (
                        inner_root_r * math.cos(theta + angular_pitch * 0.02),
                        inner_root_r * math.sin(theta + angular_pitch * 0.02),
                    ),
                    (
                        inner_tip_r * math.cos(theta + tooth_angle * 0.35),
                        inner_tip_r * math.sin(theta + tooth_angle * 0.35),
                    ),
                    (
                        inner_tip_r * math.cos(theta + angular_pitch * 0.25),
                        inner_tip_r * math.sin(theta + angular_pitch * 0.25),
                    ),
                    (
                        inner_tip_r * math.cos(theta + angular_pitch * 0.5 - tooth_angle * 0.35),
                        inner_tip_r * math.sin(theta + angular_pitch * 0.5 - tooth_angle * 0.35),
                    ),
                    (
                        inner_root_r * math.cos(theta + angular_pitch * 0.48),
                        inner_root_r * math.sin(theta + angular_pitch * 0.48),
                    ),
                ]
            )

        return self._cmd_create_internal_gear_from_profile(
            {"name": name, "points": points, "height": height, "outer_diameter": outer_r * 2, "position": pos}
        )

    # ==================== Boolean Commands ====================

    def _cmd_boolean_fuse(self, params):
        """Fuse objects."""
        doc = self._get_doc()
        name = params.get("name", "Fused")
        # Support both formats: objects array or object_a/object_b
        objects = params.get("objects") or []
        if not objects:
            obj_a = params.get("object_a")
            obj_b = params.get("object_b")
            if obj_a and obj_b:
                objects = [obj_a, obj_b]

        shapes = []
        for obj_name in objects:
            obj = doc.getObject(obj_name)
            if obj and hasattr(obj, "Shape"):
                shapes.append(obj.Shape)

        if len(shapes) >= 2:
            result = shapes[0]
            for shape in shapes[1:]:
                result = result.fuse(shape)
            fused = doc.addObject("Part::Feature", name)
            fused.Shape = result
            self._safe_recompute(doc)
            return {"status": "success", "object": name}
        return {"status": "error", "error": "Need at least 2 objects to fuse"}

    def _cmd_boolean_cut(self, params):
        """Cut one object from another."""
        doc = self._get_doc()
        name = params.get("name", "Cut")

        # Support both param formats: base/tool and object_a/object_b
        base_name = params.get("base") or params.get("object_a")
        tool_name = params.get("tool") or params.get("object_b")

        if not base_name or not tool_name:
            return {"status": "error", "error": "Both 'base' and 'tool' parameters required"}

        base = doc.getObject(base_name)
        tool = doc.getObject(tool_name)

        if not base or not hasattr(base, "Shape"):
            return {"status": "error", "error": f"Base object '{base_name}' not found or has no shape"}
        if not tool or not hasattr(tool, "Shape"):
            return {"status": "error", "error": f"Tool object '{tool_name}' not found or has no shape"}

        result = base.Shape.cut(tool.Shape)
        cut = doc.addObject("Part::Feature", name)
        cut.Shape = result
        if not params.get("keep_tool", False):
            doc.removeObject(tool.Name)
        self._safe_recompute(doc)
        return {"status": "success", "object": name}

    def _cmd_boolean_intersect(self, params):
        """Intersect objects."""
        doc = self._get_doc()
        name = params.get("name", "Intersected")
        # Support both formats: objects array or object_a/object_b
        objects = params.get("objects") or []
        if not objects:
            obj_a = params.get("object_a")
            obj_b = params.get("object_b")
            if obj_a and obj_b:
                objects = [obj_a, obj_b]

        shapes = []
        for obj_name in objects:
            obj = doc.getObject(obj_name)
            if obj and hasattr(obj, "Shape"):
                shapes.append(obj.Shape)

        if len(shapes) >= 2:
            result = shapes[0]
            for shape in shapes[1:]:
                result = result.common(shape)
            intersected = doc.addObject("Part::Feature", name)
            intersected.Shape = result
            self._safe_recompute(doc)
            return {"status": "success", "object": name}
        return {"status": "error", "error": "Need at least 2 objects to intersect"}

    # ==================== Profile-Based Operations ====================
    # These operations enable organic modeling beyond primitive shapes

    def _cmd_extrude(self, params):
        """
        Extrude a 2D profile into a 3D solid.

        Parameters:
            profile: Profile definition (type, parameters, or reference)
            distance: Extrusion distance in mm
            direction: "normal", "custom", or vector [x, y, z]
            taper_angle: Draft angle in degrees (default 0)
            is_cut: If True, creates a pocket/cut instead of boss
            name: Result object name

        Profile types:
            - rectangle: {width, height}
            - circle: {radius}
            - polygon: {sides, radius}
            - face: {object_name, face} - extrude existing face
        """
        doc = self._get_doc()
        name = params.get("name", "Extruded")
        profile = params.get("profile", {})
        distance = params.get("distance", 10)
        direction = params.get("direction", "normal")
        taper = params.get("taper_angle", 0)
        is_cut = params.get("is_cut", False)
        base_object = params.get("base_object")  # For cut operations

        try:
            import Part

            # Create the profile wire/face
            profile_type = profile.get("type", "rectangle")
            profile_params = profile.get("parameters", {})
            position = profile.get("position", [0, 0, 0])
            normal = profile.get("normal", [0, 0, 1])

            # Build the profile shape
            if profile_type == "rectangle":
                w = profile_params.get("width", 20)
                h = profile_params.get("height", 20)
                # Create rectangle centered at origin
                pts = [
                    App.Vector(-w / 2, -h / 2, 0),
                    App.Vector(w / 2, -h / 2, 0),
                    App.Vector(w / 2, h / 2, 0),
                    App.Vector(-w / 2, h / 2, 0),
                ]
                wire = Part.makePolygon(pts + [pts[0]])
                face = Part.Face(wire)

            elif profile_type == "circle":
                r = profile_params.get("radius", 10)
                circle = Part.makeCircle(r)
                wire = Part.Wire(circle)
                face = Part.Face(wire)

            elif profile_type == "polygon":
                sides = profile_params.get("sides", 6)
                r = profile_params.get("radius", 10)
                import math

                pts = []
                for i in range(sides):
                    angle = 2 * math.pi * i / sides
                    pts.append(App.Vector(r * math.cos(angle), r * math.sin(angle), 0))
                wire = Part.makePolygon(pts + [pts[0]])
                face = Part.Face(wire)

            elif profile_type == "face":
                # Use existing face from object
                ref_obj = doc.getObject(profile.get("reference_object"))
                face_ref = profile.get("reference_face", "Face1")
                if ref_obj and hasattr(ref_obj, "Shape"):
                    face_idx = int(face_ref.replace("Face", "")) - 1
                    face = ref_obj.Shape.Faces[face_idx]
                else:
                    return {"status": "error", "error": "Cannot find face reference"}

            else:
                return {"status": "error", "error": f"Unknown profile type: {profile_type}"}

            # Position the profile
            pos_vec = App.Vector(*position)
            normal_vec = App.Vector(*normal).normalize()

            # Calculate rotation to align Z-up to normal
            z_axis = App.Vector(0, 0, 1)
            if not z_axis.isEqual(normal_vec, 1e-6):
                rot_axis = z_axis.cross(normal_vec)
                if rot_axis.Length > 1e-6:
                    import math

                    rot_angle = math.acos(z_axis.dot(normal_vec)) * 180 / math.pi
                    face.rotate(App.Vector(0, 0, 0), rot_axis, rot_angle)

            face.translate(pos_vec)

            # Determine extrusion direction
            if direction == "normal":
                ext_dir = normal_vec * distance
            elif isinstance(direction, list):
                ext_dir = App.Vector(*direction).normalize() * distance
            else:
                ext_dir = normal_vec * distance

            # Create the extrusion
            if taper != 0:
                # Tapered extrusion using Part.makeExtrudeBS or similar
                solid = face.extrude(ext_dir)  # Basic for now
            else:
                solid = face.extrude(ext_dir)

            # Handle cut vs boss
            if is_cut and base_object:
                base = doc.getObject(base_object)
                if base and hasattr(base, "Shape"):
                    result_shape = base.Shape.cut(solid)
                    result = doc.addObject("Part::Feature", name)
                    result.Shape = result_shape
                else:
                    return {"status": "error", "error": f"Base object '{base_object}' not found"}
            else:
                result = doc.addObject("Part::Feature", name)
                result.Shape = solid

            self._safe_recompute(doc)

            return {
                "status": "success",
                "object": name,
                "operation": "extrude",
                "profile_type": profile_type,
                "distance": distance,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_revolve(self, params):
        """
        Revolve a 2D profile around an axis to create a rotational solid.

        Parameters:
            profile: Profile definition (same as extrude)
            axis_point: [x, y, z] point on rotation axis
            axis_direction: [x, y, z] direction of rotation axis
            angle: Rotation angle in degrees (default 360 for full revolution)
            is_cut: If True, creates a cut instead of adding material
            name: Result object name
        """
        doc = self._get_doc()
        name = params.get("name", "Revolved")
        profile = params.get("profile", {})
        axis_point = params.get("axis_point", [0, 0, 0])
        axis_dir = params.get("axis_direction", [0, 0, 1])
        angle = params.get("angle", 360)
        is_cut = params.get("is_cut", False)
        base_object = params.get("base_object")

        try:
            import Part

            # Create profile (similar to extrude)
            profile_type = profile.get("type", "rectangle")
            profile_params = profile.get("parameters", {})
            position = profile.get("position", [0, 0, 0])

            if profile_type == "rectangle":
                w = profile_params.get("width", 20)
                h = profile_params.get("height", 20)
                pts = [
                    App.Vector(-w / 2 + position[0], -h / 2 + position[1], position[2]),
                    App.Vector(w / 2 + position[0], -h / 2 + position[1], position[2]),
                    App.Vector(w / 2 + position[0], h / 2 + position[1], position[2]),
                    App.Vector(-w / 2 + position[0], h / 2 + position[1], position[2]),
                ]
                wire = Part.makePolygon(pts + [pts[0]])
                face = Part.Face(wire)

            elif profile_type == "circle":
                r = profile_params.get("radius", 10)
                circle = Part.makeCircle(r, App.Vector(*position))
                wire = Part.Wire(circle)
                face = Part.Face(wire)

            else:
                # For other types, create a simple rectangle as fallback
                w = profile_params.get("width", 20)
                h = profile_params.get("height", 20)
                pts = [
                    App.Vector(-w / 2 + position[0], -h / 2 + position[1], position[2]),
                    App.Vector(w / 2 + position[0], -h / 2 + position[1], position[2]),
                    App.Vector(w / 2 + position[0], h / 2 + position[1], position[2]),
                    App.Vector(-w / 2 + position[0], h / 2 + position[1], position[2]),
                ]
                wire = Part.makePolygon(pts + [pts[0]])
                face = Part.Face(wire)

            # Create revolution
            axis_pt = App.Vector(*axis_point)
            axis_vec = App.Vector(*axis_dir)

            solid = face.revolve(axis_pt, axis_vec, angle)

            if is_cut and base_object:
                base = doc.getObject(base_object)
                if base and hasattr(base, "Shape"):
                    result_shape = base.Shape.cut(solid)
                    result = doc.addObject("Part::Feature", name)
                    result.Shape = result_shape
                else:
                    return {"status": "error", "error": "Base object not found"}
            else:
                result = doc.addObject("Part::Feature", name)
                result.Shape = solid

            self._safe_recompute(doc)

            return {
                "status": "success",
                "object": name,
                "operation": "revolve",
                "angle": angle,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_loft(self, params):
        """
        Create a solid by blending between multiple profiles.

        Parameters:
            profiles: List of profile definitions with positions
            ruled: If True, use straight connections (faster)
            closed: If True, connect last profile to first
            solid: If True, create solid; False for surface
            name: Result object name

        Example profiles:
            [
                {"type": "circle", "parameters": {"radius": 20}, "position": [0, 0, 0]},
                {"type": "rectangle", "parameters": {"width": 30, "height": 30}, "position": [0, 0, 50]},
                {"type": "circle", "parameters": {"radius": 15}, "position": [0, 0, 100]}
            ]
        """
        doc = self._get_doc()
        name = params.get("name", "Lofted")
        profiles = params.get("profiles", [])
        ruled = params.get("ruled", False)
        closed = params.get("closed", False)
        solid = params.get("solid", True)

        if len(profiles) < 2:
            return {"status": "error", "error": "Loft requires at least 2 profiles"}

        try:
            wires = []

            for prof in profiles:
                profile_type = prof.get("type", "circle")
                profile_params = prof.get("parameters", {})
                position = prof.get("position", [0, 0, 0])
                normal = prof.get("normal", [0, 0, 1])

                if profile_type == "circle":
                    r = profile_params.get("radius", 10)
                    circle = Part.makeCircle(r, App.Vector(*position), App.Vector(*normal))
                    wire = Part.Wire(circle)

                elif profile_type == "rectangle":
                    w = profile_params.get("width", 20)
                    h = profile_params.get("height", 20)
                    # Create rectangle in XY plane, then position
                    pts = [
                        App.Vector(-w / 2, -h / 2, 0),
                        App.Vector(w / 2, -h / 2, 0),
                        App.Vector(w / 2, h / 2, 0),
                        App.Vector(-w / 2, h / 2, 0),
                    ]
                    wire = Part.makePolygon(pts + [pts[0]])
                    # Position and orient
                    wire.translate(App.Vector(*position))

                elif profile_type == "polygon":
                    sides = profile_params.get("sides", 6)
                    r = profile_params.get("radius", 10)
                    import math

                    pts = []
                    for i in range(sides):
                        angle = 2 * math.pi * i / sides
                        pts.append(
                            App.Vector(
                                r * math.cos(angle) + position[0], r * math.sin(angle) + position[1], position[2]
                            )
                        )
                    wire = Part.makePolygon(pts + [pts[0]])

                elif profile_type == "ellipse":
                    major = profile_params.get("major", 20)
                    minor = profile_params.get("minor", 10)
                    ellipse = Part.Ellipse(App.Vector(*position), major, minor)
                    wire = Part.Wire(Part.Edge(ellipse))

                else:
                    return {"status": "error", "error": f"Unknown profile type: {profile_type}"}

                wires.append(wire)

            # Create loft
            loft = Part.makeLoft(wires, solid, ruled, closed)

            result = doc.addObject("Part::Feature", name)
            result.Shape = loft
            self._safe_recompute(doc)

            return {
                "status": "success",
                "object": name,
                "operation": "loft",
                "profile_count": len(profiles),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_sweep(self, params):
        """
        Sweep a profile along a path to create a solid.

        Parameters:
            profile: Profile definition
            path: Path definition (type, parameters, or points)
            solid: If True, create solid; False for surface
            name: Result object name

        Path types:
            - line: {start: [x,y,z], end: [x,y,z]}
            - arc: {center, radius, start_angle, end_angle}
            - spline: {points: [[x,y,z], ...]}
            - helix: {radius, pitch, height}
            - edge: {object_name, edge} - use existing edge
        """
        doc = self._get_doc()
        name = params.get("name", "Swept")
        profile = params.get("profile", {})
        path = params.get("path", {})
        params.get("solid", True)

        try:
            import Part

            # Create profile wire
            profile_type = profile.get("type", "circle")
            profile_params = profile.get("parameters", {})
            profile_pos = profile.get("position", [0, 0, 0])

            if profile_type == "circle":
                r = profile_params.get("radius", 5)
                circle = Part.makeCircle(r, App.Vector(*profile_pos))
                profile_wire = Part.Wire(circle)

            elif profile_type == "rectangle":
                w = profile_params.get("width", 10)
                h = profile_params.get("height", 10)
                pts = [
                    App.Vector(-w / 2 + profile_pos[0], -h / 2 + profile_pos[1], profile_pos[2]),
                    App.Vector(w / 2 + profile_pos[0], -h / 2 + profile_pos[1], profile_pos[2]),
                    App.Vector(w / 2 + profile_pos[0], h / 2 + profile_pos[1], profile_pos[2]),
                    App.Vector(-w / 2 + profile_pos[0], h / 2 + profile_pos[1], profile_pos[2]),
                ]
                profile_wire = Part.makePolygon(pts + [pts[0]])

            else:
                r = profile_params.get("radius", 5)
                circle = Part.makeCircle(r, App.Vector(*profile_pos))
                profile_wire = Part.Wire(circle)

            # Create path
            path_type = path.get("type", "line")
            path_params = path.get("parameters", {})

            if path_type == "line":
                start = path_params.get("start", [0, 0, 0])
                end = path_params.get("end", [0, 0, 100])
                line = Part.makeLine(App.Vector(*start), App.Vector(*end))
                path_wire = Part.Wire(line)

            elif path_type == "arc":
                center = path_params.get("center", [0, 0, 0])
                radius = path_params.get("radius", 50)
                start_angle = path_params.get("start_angle", 0)
                end_angle = path_params.get("end_angle", 90)
                axis = path_params.get("axis", [0, 0, 1])
                import math

                arc = Part.makeCircle(radius, App.Vector(*center), App.Vector(*axis), start_angle, end_angle)
                path_wire = Part.Wire(arc)

            elif path_type == "spline":
                points = path.get("points", [[0, 0, 0], [50, 50, 50], [100, 0, 100]])
                pts = [App.Vector(*p) for p in points]
                spline = Part.BSplineCurve()
                spline.interpolate(pts)
                path_wire = Part.Wire(Part.Edge(spline))

            elif path_type == "helix":
                radius = path_params.get("radius", 20)
                pitch = path_params.get("pitch", 10)
                height = path_params.get("height", 50)
                center = path_params.get("center", [0, 0, 0])
                helix = Part.makeHelix(pitch, height, radius)
                helix.translate(App.Vector(*center))
                path_wire = Part.Wire(helix)

            elif path_type == "edge":
                ref_obj = doc.getObject(path.get("reference_object"))
                edge_ref = path.get("reference_edge", "Edge1")
                if ref_obj and hasattr(ref_obj, "Shape"):
                    edge_idx = int(edge_ref.replace("Edge", "")) - 1
                    path_wire = Part.Wire(ref_obj.Shape.Edges[edge_idx])
                else:
                    return {"status": "error", "error": "Edge reference not found"}

            else:
                return {"status": "error", "error": f"Unknown path type: {path_type}"}

            # Create sweep
            sweep = Part.Wire(profile_wire).makePipeShell([path_wire], True, False)

            if sweep.isNull():
                # Fallback to simpler sweep
                sweep = profile_wire.makePipe(path_wire)

            result = doc.addObject("Part::Feature", name)
            result.Shape = sweep
            self._safe_recompute(doc)

            return {
                "status": "success",
                "object": name,
                "operation": "sweep",
                "path_type": path_type,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_shell(self, params):
        """
        Hollow out a solid with uniform wall thickness.

        Parameters:
            object_name: Object to shell
            thickness: Wall thickness in mm
            faces_to_remove: List of faces to open (semantic or Face#)
            direction: "inward", "outward", or "both"
            name: Result object name

        Example:
            {"object_name": "Box", "thickness": 2, "faces_to_remove": ["top"]}
        """
        doc = self._get_doc()
        name = params.get("name", "Shelled")
        obj_name = params.get("object_name")
        thickness = params.get("thickness", 2)
        faces_to_remove = params.get("faces_to_remove", [])
        direction = params.get("direction", "inward")

        obj = doc.getObject(obj_name)
        if not obj or not hasattr(obj, "Shape"):
            return {"status": "error", "error": f"Object '{obj_name}' not found"}

        try:
            shape = obj.Shape

            # Resolve face references
            remove_faces = []
            for face_ref in faces_to_remove:
                if isinstance(face_ref, str):
                    if face_ref.startswith("Face"):
                        idx = int(face_ref.replace("Face", "")) - 1
                        if 0 <= idx < len(shape.Faces):
                            remove_faces.append(shape.Faces[idx])
                    else:
                        # Semantic face name
                        for _i, f in enumerate(shape.Faces):
                            normal = self._get_face_normal(f)
                            semantic = None
                            for fname, dir_vec in self.SEMANTIC_DIRECTIONS.items():
                                dot = sum(a * b for a, b in zip(normal, dir_vec))
                                if dot > 0.9:
                                    semantic = fname
                                    break
                            if semantic == face_ref.lower():
                                remove_faces.append(f)
                                break

            # Determine shell direction
            if direction == "outward":
                shell_thickness = thickness
            elif direction == "both":
                shell_thickness = thickness / 2
            else:  # inward
                shell_thickness = -thickness

            # Create shell
            if remove_faces:
                shelled = shape.makeThickness(remove_faces, shell_thickness, 1e-3)
            else:
                # Shell without removing faces (hollow sphere-like)
                shelled = shape.makeThickness([], shell_thickness, 1e-3)

            result = doc.addObject("Part::Feature", name)
            result.Shape = shelled
            self._safe_recompute(doc)

            # Validate shell result
            if not result.Shape.isValid():
                doc.removeObject(name)
                return {
                    "status": "error",
                    "error": "Shell produced invalid geometry",
                    "suggestion": "Try reducing thickness or selecting different faces",
                }

            if result.Shape.Volume < 0.001:
                doc.removeObject(name)
                return {
                    "status": "error",
                    "error": "Shell produced near-zero volume",
                    "suggestion": "Geometry may be too thin for requested shell thickness",
                }

            return {
                "status": "success",
                "object": name,
                "operation": "shell",
                "thickness": thickness,
                "faces_removed": len(remove_faces),
                "volume": round(result.Shape.Volume, 2),
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "suggestion": "Complex geometry may not support shell operation. Try simpler faces.",
            }

    def _cmd_import_svg(self, params):
        """
        Import an SVG file as a shape.

        Parameters:
            filepath: Path to SVG file
            name: Result object name
        """
        doc = self._get_doc()
        filepath = params.get("filepath")
        name = params.get("name", "SVGImport")

        if not filepath or not Path(filepath).exists():
            return {"status": "error", "error": f"File not found: {filepath}"}

        try:
            import importSVG

            existing_objects = set(doc.Objects)
            importSVG.insert(filepath, doc.Name)
            new_objects = [obj for obj in doc.Objects if obj not in existing_objects]

            if not new_objects:
                return {"status": "error", "error": "No objects imported from SVG"}

            # If name provided, rename the first (or combined) object
            if name:
                new_objects[0].Label = name

            return {
                "status": "success",
                "objects": [obj.Name for obj in new_objects],
                "primary_object": new_objects[0].Name,
                "count": len(new_objects),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ==================== Transform Commands ====================

    def _cmd_move_object(self, params):
        """Move an object."""
        obj, error = self._get_object(params)
        if error:
            return error

        doc = self._get_doc()
        x = params.get("x", 0)
        y = params.get("y", 0)
        z = params.get("z", 0)

        if params.get("relative", True):
            current = obj.Placement.Base
            obj.Placement.Base = App.Vector(current.x + x, current.y + y, current.z + z)
        else:
            obj.Placement.Base = App.Vector(x, y, z)

        self._safe_recompute(doc)
        return {"status": "success", "object_name": obj.Name}

    def _cmd_rotate_object(self, params):
        """Rotate an object."""
        obj, error = self._get_object(params)
        if error:
            return error

        doc = self._get_doc()
        axis = params.get("axis", "z").lower()
        angle = params.get("angle", 0)
        center = params.get("center", [0, 0, 0])

        axis_map = {"x": App.Vector(1, 0, 0), "y": App.Vector(0, 1, 0), "z": App.Vector(0, 0, 1)}
        axis_vec = axis_map.get(axis, App.Vector(0, 0, 1))

        rotation = App.Rotation(axis_vec, angle)
        obj.Placement = App.Placement(
            obj.Placement.Base, rotation * obj.Placement.Rotation, App.Vector(center[0], center[1], center[2])
        )
        self._safe_recompute(doc)
        return {"status": "success", "object_name": obj.Name}

    def _cmd_copy_object(self, params):
        """Copy an object."""
        # For copy, we use 'source' as the object to copy (different from standard object_name)
        doc = self._get_doc()
        source_name = params.get("source") or params.get("object_name")
        if not source_name:
            return {"status": "error", "error": "Missing required parameter: 'source' (object to copy)"}

        source = doc.getObject(source_name)
        if not source:
            return {"status": "error", "error": f"Source object '{source_name}' not found"}

        new_name = params.get("new_name", params.get("name", source.Name + "_copy"))
        offset = params.get("offset", [0, 0, 0])

        copy = doc.addObject("Part::Feature", new_name)
        copy.Shape = source.Shape.copy()
        copy.Placement.Base = App.Vector(
            source.Placement.Base.x + offset[0],
            source.Placement.Base.y + offset[1],
            source.Placement.Base.z + offset[2],
        )
        self._safe_recompute(doc)
        return {"status": "success", "object_name": new_name}

    def _cmd_delete_object(self, params):
        """Delete an object."""
        obj_name, error = self._get_object_param(params)
        if error:
            return error

        doc = self._get_doc()
        if doc.getObject(obj_name):
            doc.removeObject(obj_name)
            self._safe_recompute(doc)
            return {"status": "success", "deleted": obj_name}
        return {"status": "error", "error": f"Object '{obj_name}' not found"}

    # ==================== Modifier Commands ====================

    def _validate_shape(self, obj):
        """Check if an object has a valid shape after operation."""
        if not hasattr(obj, "Shape") or obj.Shape.isNull():
            return False, "Operation produced null shape"

        # Check for invalid bounding box (infinity values indicate failure)
        try:
            bbox = obj.Shape.BoundBox
            if bbox.XLength <= 0 or bbox.YLength <= 0 or bbox.ZLength <= 0:
                return False, "Operation produced empty or invalid geometry"
            # Check for infinity (indicates failed operation)
            if bbox.XMax > 1e300 or bbox.XMin < -1e300:
                return False, "Operation produced invalid geometry (computation failed)"
        except Exception as e:
            return False, f"Shape validation failed: {e}"

        return True, None

    def _safe_recompute(self, doc=None):
        """Recompute document, catching C++ geometry errors.

        Returns True on success, False if recompute raised an exception.
        The document may be in a partially-updated state on failure.
        """
        if doc is None:
            doc = self._get_doc()
        try:
            doc.recompute()
            return True
        except Exception as e:
            App.Console.PrintWarning(f"[Conjure] recompute failed: {e}\n")
            log_exception("safe_recompute", e)
            return False

    def _cmd_create_fillet(self, params):
        """Create fillet on edges.

        Supports edge selection syntax:
          - List of indices: [1, 4, 7]
          - By face: {"face": "top"} or {"face": 3}
          - By type: {"type": "Circle"}
          - By length: {"length_gt": 10, "length_lt": 50}
          - Combined: {"face": "top", "type": "Line"}

        Options:
          - validate: (default True) Pre-validate edges, skip problematic ones
          - fallback: (default True) Try Part.makeFillet() if Part::Fillet fails
          - fallback_to_chamfer: (default True) Try chamfer if all fillet methods fail
        """
        obj, error = self._get_object(params)
        if error:
            return error

        doc = self._get_doc()
        radius = params.get("radius", 1)
        edges_param = params.get("edges", None)
        validate = params.get("validate", True)
        use_fallback = params.get("fallback", True)
        fallback_to_chamfer = params.get("fallback_to_chamfer", True)
        name = params.get("name", obj.Name + "_fillet")

        # Resolve edge selector (supports list, dict, or None for all)
        edges = self._resolve_edge_selector(obj, edges_param)

        if not edges:
            return {"status": "error", "error": "No edges match the specified criteria"}

        # Pre-validate edges if enabled
        skipped = []
        if validate:
            edges, skipped = self._validate_fillettable_edges(obj, edges, radius)
            if not edges:
                return {
                    "status": "error",
                    "error": "No valid edges after validation",
                    "skipped_edges": skipped,
                }

        # Try Part::Fillet feature first
        fillet = doc.addObject("Part::Fillet", name)
        fillet_name = fillet.Name  # Store name before potential deletion
        fillet.Base = obj
        fillet.Edges = [(i, radius, radius) for i in edges]
        self._safe_recompute(doc)

        valid, error_msg = self._validate_shape(fillet)

        # If Part::Fillet fails and fallback is enabled, try Part.makeFillet()
        if not valid and use_fallback:
            doc.removeObject(fillet_name)

            try:
                # Part.makeFillet works on edge objects directly
                edge_objects = [obj.Shape.Edges[i - 1] for i in edges]
                new_shape = obj.Shape.makeFillet(radius, edge_objects)

                if not new_shape.isNull():
                    result = doc.addObject("Part::Feature", name)
                    result.Shape = new_shape
                    self._safe_recompute(doc)

                    valid, error_msg = self._validate_shape(result)
                    if valid:
                        response = {
                            "status": "success",
                            "object_name": result.Name,
                            "method": "makeFillet",
                        }
                        if skipped:
                            response["skipped_edges"] = skipped
                        return response
                    else:
                        doc.removeObject(result.Name)
            except Exception as e:
                error_msg = f"Part::Fillet and makeFillet both failed: {e}"

        # If fillet failed and chamfer fallback is enabled, try chamfer
        if not valid and fallback_to_chamfer:
            try:
                chamfer = doc.addObject("Part::Chamfer", name)
                chamfer.Base = obj
                chamfer.Edges = [(i, radius, radius) for i in edges]
                self._safe_recompute(doc)

                chamfer_valid, chamfer_error = self._validate_shape(chamfer)
                if chamfer_valid:
                    response = {
                        "status": "success",
                        "object_name": chamfer.Name,
                        "method": "chamfer_fallback",
                        "message": "Fillet failed, applied chamfer instead",
                    }
                    if skipped:
                        response["skipped_edges"] = skipped
                    return response
                else:
                    doc.removeObject(chamfer.Name)
            except Exception:
                pass  # Continue to error response

        # Adaptive edge reduction: try progressively fewer edges sorted by length
        if not valid and use_fallback and len(edges) > 1:
            # Sort edges by length descending so we keep the most significant ones
            edge_lengths = []
            for i in edges:
                try:
                    edge_len = obj.Shape.Edges[i - 1].Length
                except Exception:
                    edge_len = 0
                edge_lengths.append((i, edge_len))
            edge_lengths.sort(key=lambda x: x[1], reverse=True)

            # Try subsets: top 50%, then top 25%
            for fraction_label, fraction in [("50%", 0.5), ("25%", 0.25)]:
                subset_count = max(1, int(len(edge_lengths) * fraction))
                subset_edges = [e[0] for e in edge_lengths[:subset_count]]
                reduced_skipped = [e[0] for e in edge_lengths[subset_count:]]

                try:
                    edge_objects = [obj.Shape.Edges[i - 1] for i in subset_edges]
                    new_shape = obj.Shape.makeFillet(radius, edge_objects)

                    if not new_shape.isNull():
                        result = doc.addObject("Part::Feature", name)
                        result.Shape = new_shape
                        self._safe_recompute(doc)

                        subset_valid, _ = self._validate_shape(result)
                        if subset_valid:
                            response = {
                                "status": "success",
                                "object_name": result.Name,
                                "method": f"adaptive_makeFillet_{fraction_label}",
                                "message": f"Filleted top {fraction_label} of edges ({subset_count}/{len(edge_lengths)})",
                                "filleted_edges": subset_edges,
                                "unfilleted_edges": reduced_skipped,
                            }
                            if skipped:
                                response["pre_validation_skipped"] = skipped
                            return response
                        else:
                            doc.removeObject(result.Name)
                except Exception:
                    continue  # Try next smaller subset

        if not valid:
            # Clean up fillet object if it still exists
            if fillet_name in [o.Name for o in doc.Objects]:
                doc.removeObject(fillet_name)
            return {
                "status": "error",
                "error": f"Fillet failed: {error_msg}. Try smaller radius or specify fewer edges.",
                "skipped_edges": skipped if skipped else None,
                "attempted_edges": edges,
                "chamfer_fallback_attempted": fallback_to_chamfer,
            }

        response = {"status": "success", "object_name": fillet_name, "method": "Part::Fillet"}
        if skipped:
            response["skipped_edges"] = skipped
        return response

    def _cmd_create_chamfer(self, params):
        """Create chamfer on edges.

        Supports edge selection syntax:
          - List of indices: [1, 4, 7]
          - By face: {"face": "top"} or {"face": 3}
          - By type: {"type": "Circle"}
          - By length: {"length_gt": 10, "length_lt": 50}
          - Combined: {"face": "top", "type": "Line"}

        Options:
          - validate: (default True) Pre-validate edges, skip problematic ones
        """
        obj, error = self._get_object(params)
        if error:
            return error

        doc = self._get_doc()
        size = params.get("size", 1)
        edges_param = params.get("edges", None)
        validate = params.get("validate", True)
        name = params.get("name", obj.Name + "_chamfer")

        # Resolve edge selector (supports list, dict, or None for all)
        edges = self._resolve_edge_selector(obj, edges_param)

        if not edges:
            return {"status": "error", "error": "No edges match the specified criteria"}

        # Pre-validate edges if enabled (use size as radius equivalent)
        skipped = []
        if validate:
            edges, skipped = self._validate_fillettable_edges(obj, edges, size)
            if not edges:
                return {
                    "status": "error",
                    "error": "No valid edges after validation",
                    "skipped_edges": skipped,
                }

        chamfer = doc.addObject("Part::Chamfer", name)
        chamfer.Base = obj
        chamfer.Edges = [(i, size, size) for i in edges]

        self._safe_recompute(doc)

        # Validate result
        valid, error_msg = self._validate_shape(chamfer)
        if not valid:
            doc.removeObject(chamfer.Name)
            return {
                "status": "error",
                "error": f"Chamfer failed: {error_msg}. Try smaller size or specify fewer edges.",
                "skipped_edges": skipped if skipped else None,
                "attempted_edges": edges,
            }

        response = {"status": "success", "object_name": chamfer.Name}
        if skipped:
            response["skipped_edges"] = skipped
        return response

    # ==================== Query Commands ====================

    def _cmd_get_state(self, params):
        """Get document state."""
        doc = App.ActiveDocument
        if not doc:
            return {"status": "success", "document": None, "objects": []}

        objects = []
        for obj in doc.Objects:
            obj_info = {
                "name": obj.Name,
                "label": obj.Label,
                "type": obj.TypeId,
            }
            if hasattr(obj, "Shape"):
                bbox = obj.Shape.BoundBox
                obj_info["bounds"] = {
                    "min": [bbox.XMin, bbox.YMin, bbox.ZMin],
                    "max": [bbox.XMax, bbox.YMax, bbox.ZMax],
                }
            if hasattr(obj, "Placement"):
                pos = obj.Placement.Base
                obj_info["position"] = [pos.x, pos.y, pos.z]
            objects.append(obj_info)

        return {"status": "success", "document": doc.Name, "objects": objects}

    def _cmd_get_bounding_box(self, params):
        """Get bounding box of object."""
        obj, error = self._get_object(params)
        if error:
            return error

        if not hasattr(obj, "Shape"):
            return {"status": "error", "error": f"Object '{obj.Name}' has no shape"}

        bbox = obj.Shape.BoundBox
        return {
            "status": "success",
            "object_name": obj.Name,
            "bounding_box": {
                "min": [bbox.XMin, bbox.YMin, bbox.ZMin],
                "max": [bbox.XMax, bbox.YMax, bbox.ZMax],
                "center": [bbox.Center.x, bbox.Center.y, bbox.Center.z],
                "size": [bbox.XLength, bbox.YLength, bbox.ZLength],
            },
        }

    def _cmd_list_objects(self, params):
        """List all objects."""
        doc = App.ActiveDocument
        if not doc:
            return {"status": "success", "objects": []}

        pattern = params.get("pattern", "*")
        import fnmatch

        objects = []
        for obj in doc.Objects:
            if fnmatch.fnmatch(obj.Name, pattern):
                objects.append({"name": obj.Name, "label": obj.Label, "type": obj.TypeId})

        return {"status": "success", "objects": objects, "count": len(objects)}

    def _cmd_measure_distance(self, params):
        """Measure distance between two objects."""
        doc = self._get_doc()
        obj_a = doc.getObject(params.get("object_a"))
        obj_b = doc.getObject(params.get("object_b"))

        if not obj_a or not obj_b:
            return {"status": "error", "error": "One or both objects not found"}

        if hasattr(obj_a, "Shape") and hasattr(obj_b, "Shape"):
            dist = obj_a.Shape.distToShape(obj_b.Shape)[0]
            return {"status": "success", "distance": dist}
        return {"status": "error", "error": "Objects must have shapes"}

    # ==================== Geometry Reference System ====================

    # Semantic direction mapping for face lookup
    SEMANTIC_DIRECTIONS = {
        "top": [0, 0, 1],
        "bottom": [0, 0, -1],
        "front": [0, -1, 0],
        "back": [0, 1, 0],
        "left": [-1, 0, 0],
        "right": [1, 0, 0],
    }

    def _get_face_normal(self, face):
        """Extract normal vector from a face."""
        try:
            # Try to get normal at center of face
            uv = face.Surface.parameter(face.CenterOfMass)
            normal = face.normalAt(uv[0], uv[1])
            return [normal.x, normal.y, normal.z]
        except Exception:
            # Fallback for simple planar faces
            if hasattr(face.Surface, "Axis"):
                axis = face.Surface.Axis
                return [axis.x, axis.y, axis.z]
            return [0, 0, 0]

    def _get_face_edge_indices(self, shape, face):
        """Get 1-based edge indices for edges bounding a face."""
        face_edges = face.Edges
        indices = []
        for i, edge in enumerate(shape.Edges):
            for fe in face_edges:
                if edge.isSame(fe):
                    indices.append(i + 1)
                    break
        return indices

    def _get_face_by_direction(self, obj, direction, tolerance=0.1):
        """Find face index most aligned with semantic direction."""
        target = self.SEMANTIC_DIRECTIONS.get(direction.lower())
        if not target:
            return None

        best_face = None
        best_dot = -1

        for i, face in enumerate(obj.Shape.Faces):
            normal = self._get_face_normal(face)
            dot = sum(a * b for a, b in zip(normal, target))
            if dot > best_dot:
                best_dot = dot
                best_face = i + 1

        return best_face if best_dot > (1 - tolerance) else None

    def _get_edges_of_face(self, obj, face_ref):
        """Get edge indices for a face (by index or semantic name)."""
        shape = obj.Shape

        # Resolve semantic name to index
        if isinstance(face_ref, str):
            face_idx = self._get_face_by_direction(obj, face_ref)
            if not face_idx:
                return []
        else:
            face_idx = face_ref

        # Get face and its edge indices
        if face_idx < 1 or face_idx > len(shape.Faces):
            return []

        face = shape.Faces[face_idx - 1]
        return self._get_face_edge_indices(shape, face)

    def _get_edges_by_type(self, obj, edge_type):
        """Get edge indices matching a curve type."""
        indices = []
        for i, edge in enumerate(obj.Shape.Edges):
            if edge.Curve.__class__.__name__ == edge_type:
                indices.append(i + 1)
        return indices

    def _get_edges_by_length(self, obj, min_length=None, max_length=None):
        """Get edge indices within length range."""
        indices = []
        for i, edge in enumerate(obj.Shape.Edges):
            length = edge.Length
            if min_length is not None and length <= min_length:
                continue
            if max_length is not None and length >= max_length:
                continue
            indices.append(i + 1)
        return indices

    def _resolve_edge_selector(self, obj, selector):
        """Resolve edge selector to list of 1-based indices."""
        if selector is None:
            return list(range(1, len(obj.Shape.Edges) + 1))

        if isinstance(selector, list):
            # Convert Edge# strings to integers if needed
            result = []
            for item in selector:
                if isinstance(item, int):
                    result.append(item)
                elif isinstance(item, str) and item.startswith("Edge"):
                    try:
                        idx = int(item.replace("Edge", ""))
                        result.append(idx)
                    except ValueError:
                        pass  # Skip invalid edge refs
                elif isinstance(item, str):
                    # Try parsing as integer string
                    try:
                        result.append(int(item))
                    except ValueError:
                        pass
            return result if result else list(range(1, len(obj.Shape.Edges) + 1))

        if isinstance(selector, dict):
            result = set(range(1, len(obj.Shape.Edges) + 1))

            if "face" in selector:
                face_edges = self._get_edges_of_face(obj, selector["face"])
                result &= set(face_edges)

            if "type" in selector:
                type_edges = self._get_edges_by_type(obj, selector["type"])
                result &= set(type_edges)

            if "length_gt" in selector or "length_lt" in selector:
                length_edges = self._get_edges_by_length(
                    obj,
                    min_length=selector.get("length_gt"),
                    max_length=selector.get("length_lt"),
                )
                result &= set(length_edges)

            if "z_min" in selector or "z_max" in selector:
                z_lo = selector.get("z_min", float("-inf"))
                z_hi = selector.get("z_max", float("inf"))
                z_edges = set()
                for idx in range(1, len(obj.Shape.Edges) + 1):
                    edge = obj.Shape.Edges[idx - 1]
                    verts = edge.Vertexes
                    if len(verts) >= 2:
                        if all(z_lo <= v.Z <= z_hi for v in verts):
                            z_edges.add(idx)
                    elif len(verts) == 1:  # Closed edge (circle)
                        if z_lo <= verts[0].Z <= z_hi:
                            z_edges.add(idx)
                result &= z_edges

            if "x_min" in selector or "x_max" in selector:
                x_lo = selector.get("x_min", float("-inf"))
                x_hi = selector.get("x_max", float("inf"))
                x_edges = set()
                for idx in range(1, len(obj.Shape.Edges) + 1):
                    edge = obj.Shape.Edges[idx - 1]
                    verts = edge.Vertexes
                    if len(verts) >= 2:
                        if all(x_lo <= v.X <= x_hi for v in verts):
                            x_edges.add(idx)
                    elif len(verts) == 1:
                        if x_lo <= verts[0].X <= x_hi:
                            x_edges.add(idx)
                result &= x_edges

            if "y_min" in selector or "y_max" in selector:
                y_lo = selector.get("y_min", float("-inf"))
                y_hi = selector.get("y_max", float("inf"))
                y_edges = set()
                for idx in range(1, len(obj.Shape.Edges) + 1):
                    edge = obj.Shape.Edges[idx - 1]
                    verts = edge.Vertexes
                    if len(verts) >= 2:
                        if all(y_lo <= v.Y <= y_hi for v in verts):
                            y_edges.add(idx)
                    elif len(verts) == 1:
                        if y_lo <= verts[0].Y <= y_hi:
                            y_edges.add(idx)
                result &= y_edges

            return sorted(result)

        return []

    def _validate_fillettable_edges(self, obj, edges, radius):
        """Pre-validate edges for fillet operation.

        Returns (valid_edges, skipped_edges) where skipped_edges contains
        details about why each edge was skipped.
        """
        valid = []
        skipped = []

        for idx in edges:
            if idx < 1 or idx > len(obj.Shape.Edges):
                skipped.append({"index": idx, "reason": "invalid_index"})
                continue

            edge = obj.Shape.Edges[idx - 1]

            # Check 1: Edge length must be > 2*radius to avoid self-intersection
            if edge.Length < 2 * radius:
                skipped.append(
                    {
                        "index": idx,
                        "reason": "too_short",
                        "length": round(edge.Length, 2),
                        "min_required": round(2 * radius, 2),
                    }
                )
                continue

            # Check 2: Skip BSpline edges (often problematic with fillets)
            curve_type = edge.Curve.__class__.__name__
            if curve_type == "BSplineCurve":
                skipped.append({"index": idx, "reason": "bspline_curve"})
                continue

            valid.append(idx)

        return valid, skipped

    def _cmd_get_topology(self, params):
        """Get complete topology breakdown of an object."""
        obj, error = self._get_object(params)
        if error:
            return error

        if not hasattr(obj, "Shape"):
            return {"status": "error", "error": f"Object '{obj.Name}' has no shape"}

        shape = obj.Shape

        vertices = []
        for i, v in enumerate(shape.Vertexes):
            vertices.append(
                {
                    "index": i + 1,
                    "position": [v.X, v.Y, v.Z],
                }
            )

        edges = []
        for i, e in enumerate(shape.Edges):
            edge_info = {
                "index": i + 1,
                "type": e.Curve.__class__.__name__,
                "length": round(e.Length, 4),
                "center": [
                    round(e.BoundBox.Center.x, 4),
                    round(e.BoundBox.Center.y, 4),
                    round(e.BoundBox.Center.z, 4),
                ],
            }
            if len(e.Vertexes) >= 2:
                edge_info["start"] = [e.Vertexes[0].X, e.Vertexes[0].Y, e.Vertexes[0].Z]
                edge_info["end"] = [e.Vertexes[-1].X, e.Vertexes[-1].Y, e.Vertexes[-1].Z]
            if hasattr(e.Curve, "Radius"):
                edge_info["radius"] = round(e.Curve.Radius, 4)
            edges.append(edge_info)

        faces = []
        for i, f in enumerate(shape.Faces):
            normal = self._get_face_normal(f)
            # Determine semantic direction
            semantic = None
            for name, direction in self.SEMANTIC_DIRECTIONS.items():
                dot = sum(a * b for a, b in zip(normal, direction))
                if dot > 0.9:
                    semantic = name
                    break

            faces.append(
                {
                    "index": i + 1,
                    "type": f.Surface.__class__.__name__,
                    "area": round(f.Area, 4),
                    "center": [
                        round(f.CenterOfMass.x, 4),
                        round(f.CenterOfMass.y, 4),
                        round(f.CenterOfMass.z, 4),
                    ],
                    "normal": [round(n, 4) for n in normal],
                    "semantic": semantic,
                    "edge_indices": self._get_face_edge_indices(shape, f),
                }
            )

        return {
            "status": "success",
            "object_name": obj.Name,
            "topology": {
                "vertex_count": len(vertices),
                "edge_count": len(edges),
                "face_count": len(faces),
                "vertices": vertices,
                "edges": edges,
                "faces": faces,
            },
        }

    def _cmd_get_edges(self, params):
        """Query edges by criteria."""
        obj, error = self._get_object(params)
        if error:
            return error

        if not hasattr(obj, "Shape"):
            return {"status": "error", "error": f"Object '{obj.Name}' has no shape"}

        # Build selector from params
        selector = {}
        if "face" in params:
            selector["face"] = params["face"]
        if "type" in params:
            selector["type"] = params["type"]
        if "length_gt" in params:
            selector["length_gt"] = params["length_gt"]
        if "length_lt" in params:
            selector["length_lt"] = params["length_lt"]

        if not selector:
            indices = list(range(1, len(obj.Shape.Edges) + 1))
        else:
            indices = self._resolve_edge_selector(obj, selector)

        # Return edge details
        edges = []
        for idx in indices:
            e = obj.Shape.Edges[idx - 1]
            edge_info = {
                "index": idx,
                "type": e.Curve.__class__.__name__,
                "length": round(e.Length, 4),
            }
            if hasattr(e.Curve, "Radius"):
                edge_info["radius"] = round(e.Curve.Radius, 4)
            edges.append(edge_info)

        return {
            "status": "success",
            "object_name": obj.Name,
            "edges": edges,
            "count": len(edges),
        }

    def _cmd_get_faces(self, params):
        """Query faces by criteria."""
        obj, error = self._get_object(params)
        if error:
            return error

        if not hasattr(obj, "Shape"):
            return {"status": "error", "error": f"Object '{obj.Name}' has no shape"}

        shape = obj.Shape

        # Filters
        face_type = params.get("type")
        direction = params.get("direction")  # Semantic: "top", "bottom", etc.
        min_area = params.get("area_gt")
        max_area = params.get("area_lt")

        faces = []
        for i, f in enumerate(shape.Faces):
            # Type filter
            if face_type and f.Surface.__class__.__name__ != face_type:
                continue

            # Area filter
            if min_area is not None and f.Area <= min_area:
                continue
            if max_area is not None and f.Area >= max_area:
                continue

            normal = self._get_face_normal(f)

            # Direction filter
            if direction:
                target = self.SEMANTIC_DIRECTIONS.get(direction.lower())
                if target:
                    dot = sum(a * b for a, b in zip(normal, target))
                    if dot < 0.9:
                        continue

            # Determine semantic direction
            semantic = None
            for name, dir_vec in self.SEMANTIC_DIRECTIONS.items():
                dot = sum(a * b for a, b in zip(normal, dir_vec))
                if dot > 0.9:
                    semantic = name
                    break

            faces.append(
                {
                    "index": i + 1,
                    "type": f.Surface.__class__.__name__,
                    "area": round(f.Area, 4),
                    "center": [
                        round(f.CenterOfMass.x, 4),
                        round(f.CenterOfMass.y, 4),
                        round(f.CenterOfMass.z, 4),
                    ],
                    "normal": [round(n, 4) for n in normal],
                    "semantic": semantic,
                    "edge_count": len(f.Edges),
                }
            )

        return {
            "status": "success",
            "object_name": obj.Name,
            "faces": faces,
            "count": len(faces),
        }

    # Aliases for MCP tool names (MCP uses list_*, socket server uses get_*)
    def _cmd_list_faces(self, params):
        """Alias for get_faces - MCP compatibility."""
        return self._cmd_get_faces(params)

    def _cmd_list_edges(self, params):
        """Alias for get_edges - MCP compatibility."""
        return self._cmd_get_edges(params)

    def _cmd_get_object_details(self, params):
        """Get detailed object information."""
        return self._cmd_get_topology(params)

    # ==================== High-Level Feature Operations ====================
    # These enable context-aware modeling: "fillet the top face" instead of "fillet edges 1,3,5,7"

    def _cmd_fillet_face(self, params):
        """
        Apply fillet to all edges of a semantic face.

        This is the key "quality modeling" command - instead of guessing edge indices,
        you simply specify "top", "front", etc. and the fillet is applied intelligently.

        Parameters:
            object_name: Target object
            face: Semantic face ("top", "bottom", "front", "back", "left", "right") or Face index
            radius: Fillet radius
            edge_filter: Optional - "Line" for straight edges only, "Circle" for arcs
            name: Optional result name

        Examples:
            {"command": "fillet_face", "parameters": {"object_name": "Box", "face": "top", "radius": 2}}
            {"command": "fillet_face", "parameters": {"object_name": "Box", "face": "top", "radius": 2, "edge_filter": "Line"}}
        """
        obj, error = self._get_object(params)
        if error:
            return error

        face = params.get("face")
        if not face:
            return {"status": "error", "error": "Missing required parameter: 'face'"}

        radius = params.get("radius", 1)
        edge_filter = params.get("edge_filter")
        name = params.get("name")

        # Get edges of the specified face
        edge_indices = self._get_edges_of_face(obj, face)
        if not edge_indices:
            return {"status": "error", "error": f"No edges found for face '{face}'"}

        # Apply edge type filter if specified
        if edge_filter:
            type_edges = self._get_edges_by_type(obj, edge_filter)
            edge_indices = [i for i in edge_indices if i in type_edges]
            if not edge_indices:
                return {"status": "error", "error": f"No '{edge_filter}' edges on face '{face}'"}

        # Delegate to fillet with the resolved edges
        fillet_params = {
            "object_name": obj.Name,
            "edges": edge_indices,
            "radius": radius,
        }
        if name:
            fillet_params["name"] = name

        result = self._cmd_create_fillet(fillet_params)

        # Enhance result with semantic info
        if result.get("status") == "success":
            result["face"] = face
            result["edge_count"] = len(edge_indices)
        return result

    def _cmd_chamfer_face(self, params):
        """
        Apply chamfer to all edges of a semantic face.

        Parameters:
            object_name: Target object
            face: Semantic face ("top", "bottom", etc.) or Face index
            size: Chamfer size
            edge_filter: Optional - "Line" for straight edges only
            name: Optional result name
        """
        obj, error = self._get_object(params)
        if error:
            return error

        face = params.get("face")
        if not face:
            return {"status": "error", "error": "Missing required parameter: 'face'"}

        size = params.get("size", 1)
        edge_filter = params.get("edge_filter")
        name = params.get("name")

        edge_indices = self._get_edges_of_face(obj, face)
        if not edge_indices:
            return {"status": "error", "error": f"No edges found for face '{face}'"}

        if edge_filter:
            type_edges = self._get_edges_by_type(obj, edge_filter)
            edge_indices = [i for i in edge_indices if i in type_edges]
            if not edge_indices:
                return {"status": "error", "error": f"No '{edge_filter}' edges on face '{face}'"}

        chamfer_params = {
            "object_name": obj.Name,
            "edges": edge_indices,
            "size": size,
        }
        if name:
            chamfer_params["name"] = name

        result = self._cmd_create_chamfer(chamfer_params)
        if result.get("status") == "success":
            result["face"] = face
            result["edge_count"] = len(edge_indices)
        return result

    def _cmd_create_hole(self, params):
        """
        Create a circular hole on a specific face.

        This is context-aware hole creation - specify "top" instead of calculating
        exact positions.

        Parameters:
            object_name: Target object to drill into
            face: Face to drill on ("top", "front", or Face index)
            diameter: Hole diameter
            depth: Hole depth
            offset: Optional [x, y] offset from face center
            name: Optional result name

        Example:
            {"command": "create_hole", "parameters": {
                "object_name": "Block",
                "face": "top",
                "diameter": 10,
                "depth": 15
            }}
        """
        obj, error = self._get_object(params)
        if error:
            return error

        face = params.get("face")
        if not face:
            return {"status": "error", "error": "Missing required parameter: 'face'"}

        diameter = params.get("diameter", 5)
        depth = params.get("depth", 10)
        offset = params.get("offset", [0, 0])
        result_name = params.get("name", f"{obj.Name}_WithHole")

        # Get face info
        face_idx = self._get_face_by_direction(obj, face) if isinstance(face, str) else face
        if not face_idx:
            return {"status": "error", "error": f"Could not find face '{face}'"}

        shape = obj.Shape
        if face_idx < 1 or face_idx > len(shape.Faces):
            return {"status": "error", "error": f"Face index {face_idx} out of range"}

        face_obj = shape.Faces[face_idx - 1]
        center = face_obj.CenterOfMass
        normal = self._get_face_normal(face_obj)

        try:
            doc = self._get_doc()

            # Create cylinder for the hole
            cyl_name = f"{obj.Name}_HoleTool"
            cyl = doc.addObject("Part::Cylinder", cyl_name)
            cyl.Radius = diameter / 2
            cyl.Height = depth + 0.1  # Small extra to ensure clean cut

            # Position: center of face, with optional offset
            # Note: offset is applied in local face coordinates (simplified for now)
            hole_pos = App.Vector(center.x + offset[0], center.y + offset[1], center.z)

            # Align cylinder: default Z-up, we need to point INTO the face
            normal_vec = App.Vector(normal[0], normal[1], normal[2])
            drill_dir = normal_vec.negative()

            z_axis = App.Vector(0, 0, 1)
            rotation = App.Rotation(z_axis, drill_dir)

            # Position slightly above face to ensure clean cut
            start_pos = hole_pos + normal_vec * 0.05
            cyl.Placement = App.Placement(start_pos, rotation)

            self._safe_recompute(doc)

            # Boolean cut
            result = obj.Shape.cut(cyl.Shape)
            cut_obj = doc.addObject("Part::Feature", result_name)
            cut_obj.Shape = result

            # Clean up
            doc.removeObject(cyl_name)
            self._safe_recompute(doc)

            return {
                "status": "success",
                "object_name": result_name,
                "face": face,
                "diameter": diameter,
                "depth": depth,
            }

        except Exception as e:
            return {"status": "error", "error": f"Failed to create hole: {e}"}

    def _cmd_create_pocket(self, params):
        """
        Create a rectangular pocket on a specific face.

        Parameters:
            object_name: Target object
            face: Face to pocket ("top", "front", or Face index)
            width: Pocket width
            length: Pocket length
            depth: Pocket depth
            offset: Optional [x, y] offset from face center
            name: Optional result name

        Example:
            {"command": "create_pocket", "parameters": {
                "object_name": "Block",
                "face": "top",
                "width": 20,
                "length": 30,
                "depth": 5
            }}
        """
        obj, error = self._get_object(params)
        if error:
            return error

        face = params.get("face")
        if not face:
            return {"status": "error", "error": "Missing required parameter: 'face'"}

        width = params.get("width", 10)
        length = params.get("length", 10)
        depth = params.get("depth", 5)
        offset = params.get("offset", [0, 0])
        result_name = params.get("name", f"{obj.Name}_WithPocket")

        face_idx = self._get_face_by_direction(obj, face) if isinstance(face, str) else face
        if not face_idx:
            return {"status": "error", "error": f"Could not find face '{face}'"}

        shape = obj.Shape
        if face_idx < 1 or face_idx > len(shape.Faces):
            return {"status": "error", "error": f"Face index {face_idx} out of range"}

        face_obj = shape.Faces[face_idx - 1]
        center = face_obj.CenterOfMass
        normal = self._get_face_normal(face_obj)

        try:
            doc = self._get_doc()

            # Create box for the pocket
            box_name = f"{obj.Name}_PocketTool"
            box = doc.addObject("Part::Box", box_name)
            box.Length = length
            box.Width = width
            box.Height = depth + 0.1

            # Align box to face normal
            normal_vec = App.Vector(normal[0], normal[1], normal[2])
            drill_dir = normal_vec.negative()
            z_axis = App.Vector(0, 0, 1)
            rotation = App.Rotation(z_axis, drill_dir)

            # Position box centered on face
            # Offset by half dimensions to center
            local_offset = App.Vector(-length / 2 + offset[0], -width / 2 + offset[1], 0)
            global_offset = rotation.multVec(local_offset)

            start_pos = App.Vector(center.x, center.y, center.z) + normal_vec * 0.05 + global_offset
            box.Placement = App.Placement(start_pos, rotation)

            self._safe_recompute(doc)

            # Boolean cut
            result = obj.Shape.cut(box.Shape)
            cut_obj = doc.addObject("Part::Feature", result_name)
            cut_obj.Shape = result

            doc.removeObject(box_name)
            self._safe_recompute(doc)

            return {
                "status": "success",
                "object_name": result_name,
                "face": face,
                "width": width,
                "length": length,
                "depth": depth,
            }

        except Exception as e:
            return {"status": "error", "error": f"Failed to create pocket: {e}"}

    # ==================== Smart Placement Commands ====================

    def _cmd_align_to_face(self, params):
        """
        Align one object to a face of another object.

        Useful for placing features relative to existing geometry.

        Parameters:
            object_name: Object to move
            target: Target object
            target_face: Face of target to align to ("top", "front", etc.)
            align_face: Which face of moving object to align (default: "bottom")
            gap: Gap between surfaces (default: 0)

        Example:
            {"command": "align_to_face", "parameters": {
                "object_name": "Widget",
                "target": "Base",
                "target_face": "top",
                "align_face": "bottom",
                "gap": 0.5
            }}
        """
        obj, error = self._get_object(params)
        if error:
            return error

        doc = self._get_doc()
        target_name = params.get("target")
        if not target_name:
            return {"status": "error", "error": "Missing required parameter: 'target'"}

        target = doc.getObject(target_name)
        if not target:
            return {"status": "error", "error": f"Target object '{target_name}' not found"}

        target_face = params.get("target_face", "top")
        align_face = params.get("align_face", "bottom")
        gap = params.get("gap", 0)

        # Get target face info
        target_face_idx = self._get_face_by_direction(target, target_face)
        if not target_face_idx:
            return {"status": "error", "error": f"Could not find target face '{target_face}'"}

        target_face_obj = target.Shape.Faces[target_face_idx - 1]
        target_center = target_face_obj.CenterOfMass
        target_normal = self._get_face_normal(target_face_obj)

        # Get source face info
        source_face_idx = self._get_face_by_direction(obj, align_face)
        if not source_face_idx:
            return {"status": "error", "error": f"Could not find align face '{align_face}'"}

        source_face_obj = obj.Shape.Faces[source_face_idx - 1]
        source_center = source_face_obj.CenterOfMass

        # Calculate movement: align centers, then offset by gap along normal
        normal_vec = App.Vector(target_normal[0], target_normal[1], target_normal[2])
        move_vec = (
            App.Vector(target_center.x, target_center.y, target_center.z)
            - App.Vector(source_center.x, source_center.y, source_center.z)
            + normal_vec * gap
        )

        obj.Placement.Base = obj.Placement.Base + move_vec
        self._safe_recompute(doc)

        return {
            "status": "success",
            "object_name": obj.Name,
            "aligned_to": target_name,
            "target_face": target_face,
            "moved_by": [move_vec.x, move_vec.y, move_vec.z],
        }

    def _cmd_place_on_face(self, params):
        """
        Place object centered on a face of another object.

        Parameters:
            object_name: Object to place
            target: Target object to place on
            face: Face of target ("top", "front", etc.)
            offset: Optional [x, y] offset on the face plane
            gap: Gap above the surface (default: 0)
        """
        obj, error = self._get_object(params)
        if error:
            return error

        doc = self._get_doc()
        target_name = params.get("target")
        if not target_name:
            return {"status": "error", "error": "Missing required parameter: 'target'"}

        target = doc.getObject(target_name)
        if not target:
            return {"status": "error", "error": f"Target object '{target_name}' not found"}

        face = params.get("face", "top")
        offset = params.get("offset", [0, 0])
        gap = params.get("gap", 0)

        # Get target face
        face_idx = self._get_face_by_direction(target, face)
        if not face_idx:
            return {"status": "error", "error": f"Could not find face '{face}'"}

        face_obj = target.Shape.Faces[face_idx - 1]
        center = face_obj.CenterOfMass
        normal = self._get_face_normal(face_obj)
        normal_vec = App.Vector(normal[0], normal[1], normal[2])

        # Calculate object's bounding box to determine offset for "sitting" on face
        obj_bbox = obj.Shape.BoundBox
        obj_center = obj_bbox.Center

        # Move object so its center is at face center, offset by half height + gap
        half_height = (
            obj_bbox.ZLength / 2
            if normal[2] > 0.5
            else (obj_bbox.YLength / 2 if abs(normal[1]) > 0.5 else obj_bbox.XLength / 2)
        )

        new_pos = App.Vector(center.x + offset[0], center.y + offset[1], center.z) + normal_vec * (half_height + gap)

        # Adjust for current object center offset
        current_base = obj.Placement.Base
        center_offset = App.Vector(obj_center.x, obj_center.y, obj_center.z) - current_base
        obj.Placement.Base = new_pos - center_offset

        self._safe_recompute(doc)

        return {
            "status": "success",
            "object_name": obj.Name,
            "placed_on": target_name,
            "face": face,
            "position": [obj.Placement.Base.x, obj.Placement.Base.y, obj.Placement.Base.z],
        }

    def _cmd_center_on(self, params):
        """
        Center one object on another in specified axes.

        Parameters:
            object_name: Object to center
            target: Reference object
            axes: List of axes to center on (default: ["x", "y"])
        """
        obj, error = self._get_object(params)
        if error:
            return error

        doc = self._get_doc()
        target_name = params.get("target")
        if not target_name:
            return {"status": "error", "error": "Missing required parameter: 'target'"}

        target = doc.getObject(target_name)
        if not target:
            return {"status": "error", "error": f"Target object '{target_name}' not found"}

        axes = params.get("axes", ["x", "y"])

        obj_center = obj.Shape.BoundBox.Center
        target_center = target.Shape.BoundBox.Center

        move = App.Vector(0, 0, 0)
        if "x" in axes:
            move.x = target_center.x - obj_center.x
        if "y" in axes:
            move.y = target_center.y - obj_center.y
        if "z" in axes:
            move.z = target_center.z - obj_center.z

        obj.Placement.Base = obj.Placement.Base + move
        self._safe_recompute(doc)

        return {
            "status": "success",
            "object_name": obj.Name,
            "centered_on": target_name,
            "axes": axes,
            "moved_by": [move.x, move.y, move.z],
        }

    # ==================== Shape Repair Commands ====================

    def _cmd_repair_shape(self, params):
        """Repair shape geometry (sew open shells, fix tolerances).

        Parameters:
            object_name: Object to repair
            tolerance: Sew tolerance in mm (default 0.01)
            name: Optional name for repaired copy (default modifies in-place)
        """
        obj, error = self._get_object(params)
        if error:
            return error

        if not hasattr(obj, "Shape"):
            return {"status": "error", "error": f"Object '{obj.Name}' has no shape"}

        shape = obj.Shape
        tolerance = params.get("tolerance", 0.01)
        new_name = params.get("name")

        before_valid = shape.isValid()
        before_shells = len(shape.Shells)
        before_faces = len(shape.Faces)

        try:
            import Part

            # Copy shape to avoid mutating original until we know repair works
            repaired = shape.copy()

            # Sew open shells together
            repaired.sewShape(tolerance)

            # Fix tolerance issues
            repaired.fix(tolerance, tolerance, tolerance)

            after_valid = repaired.isValid()
            after_shells = len(repaired.Shells)
            after_faces = len(repaired.Faces)

            if new_name:
                # Create new object with repaired shape
                doc = self._get_doc()
                new_obj = doc.addObject("Part::Feature", new_name)
                new_obj.Shape = repaired
                self._safe_recompute(doc)
                target_name = new_name
            else:
                # Modify in-place
                obj.Shape = repaired
                self._get_doc().recompute()
                target_name = obj.Name

            return {
                "status": "success",
                "object": target_name,
                "before": {
                    "valid": before_valid,
                    "shells": before_shells,
                    "faces": before_faces,
                },
                "after": {
                    "valid": after_valid,
                    "shells": after_shells,
                    "faces": after_faces,
                },
                "tolerance": tolerance,
            }

        except Exception as e:
            return {"status": "error", "error": f"Shape repair failed: {str(e)}"}

    def _cmd_remove_splitter(self, params):
        """Remove redundant edges/faces left by boolean operations.

        Applies Part.removeSplitter() and persists the result by creating a
        new Part::Feature object, avoiding the revert-on-save problem that
        occurs when directly assigning to a parametric object's Shape.

        Parameters:
            object_name: Object to clean
            name: Optional name for the result (default: replaces original)
        """
        obj, error = self._get_object(params)
        if error:
            return error

        if not hasattr(obj, "Shape"):
            return {"status": "error", "error": f"Object '{obj.Name}' has no shape"}

        shape = obj.Shape
        if shape.isNull():
            return {"status": "error", "error": f"Object '{obj.Name}' has a null shape"}

        before_faces = len(shape.Faces)
        before_edges = len(shape.Edges)

        try:
            cleaned = shape.removeSplitter()

            if cleaned.isNull():
                return {
                    "status": "error",
                    "error": "removeSplitter() produced a null shape — original preserved",
                }

            after_faces = len(cleaned.Faces)
            after_edges = len(cleaned.Edges)

            doc = self._get_doc()
            result_name = params.get("name") or obj.Name

            if result_name == obj.Name:
                # Replace in-place: create new Part::Feature, remove original
                original_label = obj.Label
                original_placement = obj.Placement
                original_visibility = obj.ViewObject.Visibility if hasattr(obj, "ViewObject") and obj.ViewObject else True

                # Remove the original parametric object
                doc.removeObject(obj.Name)

                # Create a non-parametric Part::Feature with the cleaned shape
                new_obj = doc.addObject("Part::Feature", result_name)
                new_obj.Label = original_label
                new_obj.Shape = cleaned
                new_obj.Placement = original_placement
                if hasattr(new_obj, "ViewObject") and new_obj.ViewObject:
                    new_obj.ViewObject.Visibility = original_visibility
            else:
                # Create a separate new object
                new_obj = doc.addObject("Part::Feature", result_name)
                new_obj.Label = result_name
                new_obj.Shape = cleaned
                new_obj.Placement = obj.Placement

            self._safe_recompute(doc)

            return {
                "status": "success",
                "object": new_obj.Name,
                "before": {"faces": before_faces, "edges": before_edges},
                "after": {"faces": after_faces, "edges": after_edges},
                "faces_removed": before_faces - after_faces,
                "edges_removed": before_edges - after_edges,
            }

        except Exception as e:
            return {"status": "error", "error": f"removeSplitter failed: {str(e)}"}

    # ==================== Analysis & Validation Commands ====================

    def _cmd_check_interference(self, params):
        """
        Check if two objects interfere (overlap/intersect).

        Parameters:
            object_a: First object
            object_b: Second object
            quick: If True (default), use fast bounding-box + distance check only
            calculate_volume: If True, calculate exact interference volume (slow!)

        Returns:
            interferes: Boolean
            distance: Minimum distance (0 if touching, negative if overlapping)
            interference_volume: Volume of intersection (if calculate_volume=True)
        """
        doc = self._get_doc()
        obj_a_name = params.get("object_a") or params.get("object_name")
        obj_b_name = params.get("object_b")
        quick_mode = params.get("quick", True)  # Default to fast mode

        if not obj_a_name or not obj_b_name:
            return {"status": "error", "error": "Both 'object_a' and 'object_b' required"}

        obj_a = doc.getObject(obj_a_name)
        obj_b = doc.getObject(obj_b_name)

        if not obj_a or not obj_b:
            return {"status": "error", "error": "One or both objects not found"}

        if not hasattr(obj_a, "Shape") or not hasattr(obj_b, "Shape"):
            return {"status": "error", "error": "Both objects must have shapes"}

        try:
            # FAST: Bounding box check
            bb_a = obj_a.Shape.BoundBox
            bb_b = obj_b.Shape.BoundBox
            bb_overlap = bb_a.intersect(bb_b)

            # If bounding boxes don't intersect, no interference possible
            if not bb_overlap:
                return {
                    "status": "success",
                    "object_a": obj_a_name,
                    "object_b": obj_b_name,
                    "interferes": False,
                    "distance": 1.0,  # Positive = no overlap
                    "method": "bounding_box",
                }

            # bbox_only mode: just report BB overlap, skip expensive shape checks
            # Useful for quick iterative checks on complex geometry
            if params.get("bbox_only", False):
                return {
                    "status": "success",
                    "object_a": obj_a_name,
                    "object_b": obj_b_name,
                    "interferes": True,  # BBs overlap, assume interference
                    "distance": 0.0,
                    "method": "bounding_box_only",
                }

            # Bounding boxes overlap - need more detailed check
            # Use distToShape which is faster than boolean common
            dist_info = obj_a.Shape.distToShape(obj_b.Shape)
            distance = dist_info[0]

            # In quick mode, use distance as proxy for interference
            # distance <= 0 means shapes are touching or overlapping
            if quick_mode:
                interferes = distance < 0.001  # Small tolerance for touching
                result = {
                    "status": "success",
                    "object_a": obj_a_name,
                    "object_b": obj_b_name,
                    "interferes": interferes,
                    "distance": round(distance, 4),
                    "method": "distance",
                }
                return result

            # SLOW: Full boolean intersection (only if explicitly requested)
            common = obj_a.Shape.common(obj_b.Shape)
            interferes = not common.isNull() and common.Volume > 0.001

            result = {
                "status": "success",
                "object_a": obj_a_name,
                "object_b": obj_b_name,
                "interferes": interferes,
                "distance": round(distance, 4),
                "method": "boolean",
            }

            if params.get("calculate_volume") and interferes:
                result["interference_volume"] = round(common.Volume, 4)

            return result

        except Exception as e:
            return {"status": "error", "error": f"Interference check failed: {e}"}

    def _cmd_analyze_fit(self, params):
        """
        Analyze if one object would fit inside/around another.

        Useful for checking if a phone fits in a holder, etc.

        Parameters:
            inner: Object that should fit inside
            outer: Container/holder object
            clearance: Required clearance on each side (default: 0)

        Returns:
            fits: Boolean
            clearances: {x, y, z} clearance on each axis
            recommendations: Suggestions if it doesn't fit
        """
        doc = self._get_doc()
        inner_name = params.get("inner")
        outer_name = params.get("outer")
        required_clearance = params.get("clearance", 0)

        if not inner_name or not outer_name:
            return {"status": "error", "error": "Both 'inner' and 'outer' required"}

        inner = doc.getObject(inner_name)
        outer = doc.getObject(outer_name)

        if not inner or not outer:
            return {"status": "error", "error": "One or both objects not found"}

        inner_bbox = inner.Shape.BoundBox
        outer_bbox = outer.Shape.BoundBox

        clearances = {
            "x": (outer_bbox.XLength - inner_bbox.XLength) / 2,
            "y": (outer_bbox.YLength - inner_bbox.YLength) / 2,
            "z": (outer_bbox.ZLength - inner_bbox.ZLength) / 2,
        }

        fits = all(c >= required_clearance for c in clearances.values())

        recommendations = []
        if not fits:
            for axis, clearance in clearances.items():
                if clearance < required_clearance:
                    needed = required_clearance - clearance
                    recommendations.append(f"Increase outer {axis.upper()} dimension by {round(needed * 2, 2)}mm")

        return {
            "status": "success",
            "inner": inner_name,
            "outer": outer_name,
            "fits": fits,
            "clearances": {k: round(v, 2) for k, v in clearances.items()},
            "required_clearance": required_clearance,
            "recommendations": recommendations if not fits else None,
        }

    def _cmd_find_face(self, params):
        """
        Find a face dynamically by criteria.

        Parameters:
            object_name: Target object
            direction: Semantic direction ("top", "front", etc.)
            area_gt: Minimum area
            area_lt: Maximum area
            type: Surface type ("Plane", "Cylinder", etc.)
            closest_to: [x, y, z] point to find nearest face

        Returns the best matching face with full details.
        """
        obj, error = self._get_object(params)
        if error:
            return error

        if not hasattr(obj, "Shape"):
            return {"status": "error", "error": f"Object '{obj.Name}' has no shape"}

        shape = obj.Shape
        direction = params.get("direction")
        area_gt = params.get("area_gt")
        area_lt = params.get("area_lt")
        face_type = params.get("type")
        closest_to = params.get("closest_to")

        candidates = []

        for i, f in enumerate(shape.Faces):
            normal = self._get_face_normal(f)

            # Direction filter
            if direction:
                target = self.SEMANTIC_DIRECTIONS.get(direction.lower())
                if target:
                    dot = sum(a * b for a, b in zip(normal, target))
                    if dot < 0.7:  # Allow some tolerance
                        continue

            # Area filter
            if area_gt is not None and f.Area <= area_gt:
                continue
            if area_lt is not None and f.Area >= area_lt:
                continue

            # Type filter
            if face_type and f.Surface.__class__.__name__ != face_type:
                continue

            # Calculate distance if closest_to specified
            dist = 0
            if closest_to:
                pt = App.Vector(closest_to[0], closest_to[1], closest_to[2])
                dist = f.CenterOfMass.distanceToPoint(pt)

            # Determine semantic name
            semantic = None
            for name, dir_vec in self.SEMANTIC_DIRECTIONS.items():
                dot = sum(a * b for a, b in zip(normal, dir_vec))
                if dot > 0.9:
                    semantic = name
                    break

            candidates.append(
                {
                    "index": i + 1,
                    "type": f.Surface.__class__.__name__,
                    "area": round(f.Area, 4),
                    "center": [round(f.CenterOfMass.x, 4), round(f.CenterOfMass.y, 4), round(f.CenterOfMass.z, 4)],
                    "normal": [round(n, 4) for n in normal],
                    "semantic": semantic,
                    "distance": round(dist, 4) if closest_to else None,
                }
            )

        # Sort by relevance
        if closest_to:
            candidates.sort(key=lambda x: x["distance"])
        elif direction:
            # Sort by alignment to direction
            target = self.SEMANTIC_DIRECTIONS.get(direction.lower(), [0, 0, 1])
            candidates.sort(key=lambda x: -sum(a * b for a, b in zip(x["normal"], target)))

        if not candidates:
            return {"status": "success", "object_name": obj.Name, "face": None, "message": "No matching face found"}

        return {
            "status": "success",
            "object_name": obj.Name,
            "face": candidates[0],
            "alternatives": candidates[1:5] if len(candidates) > 1 else [],
        }

    def _cmd_get_face_info(self, params):
        """
        Get detailed info about a specific face.

        Parameters:
            object_name: Target object
            face: Face index (1-based) or semantic name ("top", "front", etc.)

        Returns:
            Detailed face information including edges, area, normal, etc.
        """
        obj, error = self._get_object(params)
        if error:
            return error

        face = params.get("face")
        if not face:
            return {"status": "error", "error": "Missing required parameter: 'face'"}

        # Resolve face reference
        if isinstance(face, str):
            face_idx = self._get_face_by_direction(obj, face)
            if not face_idx:
                return {"status": "error", "error": f"Could not find face '{face}'"}
        else:
            face_idx = face

        if face_idx < 1 or face_idx > len(obj.Shape.Faces):
            return {"status": "error", "error": f"Face index {face_idx} out of range"}

        f = obj.Shape.Faces[face_idx - 1]
        normal = self._get_face_normal(f)
        edge_indices = self._get_face_edge_indices(obj.Shape, f)

        # Get edge details
        edges = []
        for idx in edge_indices:
            e = obj.Shape.Edges[idx - 1]
            edges.append(
                {
                    "index": idx,
                    "type": e.Curve.__class__.__name__,
                    "length": round(e.Length, 4),
                }
            )

        # Determine semantic name
        semantic = None
        for name, dir_vec in self.SEMANTIC_DIRECTIONS.items():
            dot = sum(a * b for a, b in zip(normal, dir_vec))
            if dot > 0.9:
                semantic = name
                break

        return {
            "status": "success",
            "object_name": obj.Name,
            "face": {
                "index": face_idx,
                "semantic": semantic,
                "type": f.Surface.__class__.__name__,
                "area": round(f.Area, 4),
                "center": [round(f.CenterOfMass.x, 4), round(f.CenterOfMass.y, 4), round(f.CenterOfMass.z, 4)],
                "normal": [round(n, 4) for n in normal],
                "edges": edges,
                "edge_count": len(edges),
            },
        }

    def _cmd_validate_geometry(self, params):
        """Validate geometry of one or all objects.

        Parameters:
            object_name: Object to validate (optional, validates all if omitted)

        Returns:
            Per-object validity, issues found, and summary.
        """
        doc = self._get_doc()
        target = params.get("object_name")

        objects_to_check = []
        if target:
            obj = doc.getObject(target)
            if not obj:
                return {"status": "error", "error": f"Object '{target}' not found"}
            objects_to_check = [obj]
        else:
            objects_to_check = [o for o in doc.Objects if hasattr(o, "Shape")]

        results = []
        for obj in objects_to_check:
            issues = []
            shape = obj.Shape
            is_null = shape.isNull()
            is_valid = False if is_null else shape.isValid()

            if is_null:
                issues.append("Shape is null")
            elif not is_valid:
                issues.append("Shape fails BRep validity check")

            if not is_null:
                bb = shape.BoundBox
                if bb.XLength < 1e-6 or bb.YLength < 1e-6 or bb.ZLength < 1e-6:
                    issues.append(f"Degenerate bounding box: {bb.XLength:.4f} x {bb.YLength:.4f} x {bb.ZLength:.4f}")
                if hasattr(shape, "Volume") and shape.Volume < 1e-10:
                    issues.append("Near-zero volume")
                if hasattr(shape, "Shells"):
                    for i, shell in enumerate(shape.Shells):
                        if not shell.isClosed():
                            issues.append(f"Shell {i + 1} is not closed (not watertight)")

            results.append(
                {
                    "object": obj.Name,
                    "type": obj.TypeId,
                    "valid": is_valid and not is_null,
                    "issues": issues,
                }
            )

        valid_count = sum(1 for r in results if r["valid"])
        return {
            "status": "success",
            "objects": results,
            "summary": {
                "total": len(results),
                "valid": valid_count,
                "invalid": len(results) - valid_count,
            },
        }

    # ==================== Check All & Suggest Fixes ====================

    def _cmd_check_all(self, params):
        """Run validate_geometry on all objects and check for interference pairs.

        Returns a rollup summary with per-object validity and interference info.
        """
        doc = self._get_doc()
        shape_objects = [o for o in doc.Objects if hasattr(o, "Shape")]

        # Validate each object
        object_results = []
        for obj in shape_objects:
            issues = []
            shape = obj.Shape
            is_null = shape.isNull()
            is_valid = False if is_null else shape.isValid()

            if is_null:
                issues.append("Shape is null")
            elif not is_valid:
                issues.append("Shape fails BRep validity check")

            if not is_null:
                bb = shape.BoundBox
                if bb.XLength < 1e-6 or bb.YLength < 1e-6 or bb.ZLength < 1e-6:
                    issues.append(f"Degenerate bounding box: {bb.XLength:.4f} x {bb.YLength:.4f} x {bb.ZLength:.4f}")
                if hasattr(shape, "Volume") and shape.Volume < 1e-10:
                    issues.append("Near-zero volume")
                if hasattr(shape, "Shells"):
                    for i, shell in enumerate(shape.Shells):
                        if not shell.isClosed():
                            issues.append(f"Shell {i + 1} is not closed (not watertight)")

            object_results.append(
                {
                    "object": obj.Name,
                    "type": obj.TypeId,
                    "valid": is_valid and not is_null,
                    "issues": issues,
                }
            )

        # Check interference between all pairs
        interferences = []
        for i in range(len(shape_objects)):
            for j in range(i + 1, len(shape_objects)):
                a, b = shape_objects[i], shape_objects[j]
                try:
                    bb_a = a.Shape.BoundBox
                    bb_b = b.Shape.BoundBox
                    if bb_a.intersect(bb_b):
                        dist = a.Shape.distToShape(b.Shape)[0]
                        if dist < 0.001:
                            interferences.append(
                                {
                                    "object_a": a.Name,
                                    "object_b": b.Name,
                                    "distance": round(dist, 4),
                                }
                            )
                except Exception:
                    pass

        valid_count = sum(1 for r in object_results if r["valid"])
        return {
            "status": "success",
            "objects": object_results,
            "interferences": interferences,
            "summary": {
                "total_objects": len(object_results),
                "valid": valid_count,
                "invalid": len(object_results) - valid_count,
                "interference_pairs": len(interferences),
            },
        }

    def _cmd_suggest_fixes(self, params):
        """Analyze validation results and suggest concrete actions.

        Runs check_all internally and returns actionable suggestions.
        """
        check_result = self._cmd_check_all(params)
        if check_result.get("status") != "success":
            return check_result

        suggestions = []
        for obj_result in check_result.get("objects", []):
            if obj_result["valid"]:
                continue
            name = obj_result["object"]
            for issue in obj_result.get("issues", []):
                if "not closed" in issue or "not watertight" in issue:
                    suggestions.append(
                        {
                            "object": name,
                            "issue": issue,
                            "fix": f"Use shell_object on '{name}' to close the shell, or recreate the geometry.",
                        }
                    )
                elif "null" in issue.lower():
                    suggestions.append(
                        {
                            "object": name,
                            "issue": issue,
                            "fix": f"Delete '{name}' and recreate it — the shape is empty.",
                        }
                    )
                elif "BRep validity" in issue:
                    suggestions.append(
                        {
                            "object": name,
                            "issue": issue,
                            "fix": f"Try delete_object('{name}') and recreate. BRep failures often come from bad boolean ops.",
                        }
                    )
                elif "Degenerate" in issue:
                    suggestions.append(
                        {
                            "object": name,
                            "issue": issue,
                            "fix": f"Object '{name}' is flat or zero-size. Check dimensions and recreate.",
                        }
                    )
                elif "Near-zero volume" in issue:
                    suggestions.append(
                        {
                            "object": name,
                            "issue": issue,
                            "fix": f"Object '{name}' has no volume. It may be a 2D shape — extrude it or delete.",
                        }
                    )
                else:
                    suggestions.append(
                        {
                            "object": name,
                            "issue": issue,
                            "fix": f"Inspect '{name}' manually and consider recreating.",
                        }
                    )

        for interference in check_result.get("interferences", []):
            suggestions.append(
                {
                    "object": f"{interference['object_a']} / {interference['object_b']}",
                    "issue": f"Objects overlap (distance: {interference['distance']})",
                    "fix": "Move one of the objects apart, or use boolean_cut to subtract one from the other.",
                }
            )

        return {
            "status": "success",
            "suggestions": suggestions,
            "summary": check_result["summary"],
        }

    # ==================== Measurement (Extended) ====================

    def _cmd_check_face_alignment(self, params):
        """Compare normals of two faces and report alignment.

        Parameters:
            object_a: First object name
            face_a: Face selector for first object (e.g. "top", "bottom", index)
            object_b: Second object name
            face_b: Face selector for second object
        """
        doc = self._get_doc()
        obj_a = doc.getObject(params.get("object_a"))
        obj_b = doc.getObject(params.get("object_b"))

        if not obj_a or not obj_b:
            return {"status": "error", "error": "One or both objects not found"}

        face_a_sel = params.get("face_a", "top")
        face_b_sel = params.get("face_b", "top")

        # Resolve face indices
        face_a_idx = self._resolve_face_index(obj_a, face_a_sel)
        face_b_idx = self._resolve_face_index(obj_b, face_b_sel)

        if face_a_idx is None:
            return {"status": "error", "error": f"Could not resolve face '{face_a_sel}' on {obj_a.Name}"}
        if face_b_idx is None:
            return {"status": "error", "error": f"Could not resolve face '{face_b_sel}' on {obj_b.Name}"}

        face_a_obj = obj_a.Shape.Faces[face_a_idx - 1]
        face_b_obj = obj_b.Shape.Faces[face_b_idx - 1]

        normal_a = self._get_face_normal(face_a_obj)
        normal_b = self._get_face_normal(face_b_obj)

        import math

        dot = sum(a * b for a, b in zip(normal_a, normal_b))
        dot = max(-1.0, min(1.0, dot))
        angle_rad = math.acos(dot)
        angle_deg = math.degrees(angle_rad)

        alignment = "other"
        if angle_deg < 5.0:
            alignment = "parallel_same_direction"
        elif abs(angle_deg - 180.0) < 5.0:
            alignment = "parallel_opposite"
        elif abs(angle_deg - 90.0) < 5.0:
            alignment = "perpendicular"

        # Check if faces are flush (co-planar and facing each other)
        flush = False
        if alignment == "parallel_opposite":
            center_a = face_a_obj.CenterOfMass
            center_b = face_b_obj.CenterOfMass
            diff = App.Vector(center_b.x - center_a.x, center_b.y - center_a.y, center_b.z - center_a.z)
            n = App.Vector(normal_a[0], normal_a[1], normal_a[2])
            proj = abs(diff.dot(n))
            lateral = (diff - n * proj).Length
            if lateral < 0.01 or proj < 0.01:
                flush = True

        return {
            "status": "success",
            "object_a": obj_a.Name,
            "face_a": face_a_sel,
            "normal_a": [round(n, 4) for n in normal_a],
            "object_b": obj_b.Name,
            "face_b": face_b_sel,
            "normal_b": [round(n, 4) for n in normal_b],
            "angle_degrees": round(angle_deg, 2),
            "alignment": alignment,
            "flush": flush,
        }

    def _cmd_measure_face_distance(self, params):
        """Get minimum distance between specific faces of two objects.

        Parameters:
            object_a: First object name
            face_a: Face selector for first object
            object_b: Second object name
            face_b: Face selector for second object
        """
        doc = self._get_doc()
        obj_a = doc.getObject(params.get("object_a"))
        obj_b = doc.getObject(params.get("object_b"))

        if not obj_a or not obj_b:
            return {"status": "error", "error": "One or both objects not found"}

        face_a_sel = params.get("face_a", "top")
        face_b_sel = params.get("face_b", "bottom")

        face_a_idx = self._resolve_face_index(obj_a, face_a_sel)
        face_b_idx = self._resolve_face_index(obj_b, face_b_sel)

        if face_a_idx is None:
            return {"status": "error", "error": f"Could not resolve face '{face_a_sel}' on {obj_a.Name}"}
        if face_b_idx is None:
            return {"status": "error", "error": f"Could not resolve face '{face_b_sel}' on {obj_b.Name}"}

        face_a_obj = obj_a.Shape.Faces[face_a_idx - 1]
        face_b_obj = obj_b.Shape.Faces[face_b_idx - 1]

        try:
            dist_info = face_a_obj.distToShape(face_b_obj)
            distance = dist_info[0]
            return {
                "status": "success",
                "object_a": obj_a.Name,
                "face_a": face_a_sel,
                "object_b": obj_b.Name,
                "face_b": face_b_sel,
                "distance": round(distance, 4),
            }
        except Exception as e:
            return {"status": "error", "error": f"Face distance measurement failed: {e}"}

    def _cmd_measure_gap(self, params):
        """Measure minimum gap (shape-to-shape distance) between two objects.

        Parameters:
            object_a: First object name
            object_b: Second object name
        """
        doc = self._get_doc()
        obj_a = doc.getObject(params.get("object_a"))
        obj_b = doc.getObject(params.get("object_b"))

        if not obj_a or not obj_b:
            return {"status": "error", "error": "One or both objects not found"}

        if not hasattr(obj_a, "Shape") or not hasattr(obj_b, "Shape"):
            return {"status": "error", "error": "Both objects must have shapes"}

        try:
            dist_info = obj_a.Shape.distToShape(obj_b.Shape)
            distance = dist_info[0]
            touching = distance < 0.001
            # Extract closest points if available
            closest_points = []
            if len(dist_info) > 1:
                for pair in dist_info[1]:
                    closest_points.append(
                        {
                            "point_a": [round(pair[0].x, 4), round(pair[0].y, 4), round(pair[0].z, 4)],
                            "point_b": [round(pair[1].x, 4), round(pair[1].y, 4), round(pair[1].z, 4)],
                        }
                    )
            return {
                "status": "success",
                "object_a": obj_a.Name,
                "object_b": obj_b.Name,
                "gap": round(distance, 4),
                "touching": touching,
                "closest_points": closest_points[:3],
            }
        except Exception as e:
            return {"status": "error", "error": f"Gap measurement failed: {e}"}

    # ==================== Contact & Search ====================

    def _cmd_find_contacts(self, params):
        """Find faces of two objects that are within tolerance distance (touching).

        Parameters:
            object_a: First object name
            object_b: Second object name
            tolerance: Distance threshold (default 0.1)
        """
        doc = self._get_doc()
        obj_a = doc.getObject(params.get("object_a"))
        obj_b = doc.getObject(params.get("object_b"))
        tolerance = params.get("tolerance", 0.1)

        if not obj_a or not obj_b:
            return {"status": "error", "error": "One or both objects not found"}

        if not hasattr(obj_a, "Shape") or not hasattr(obj_b, "Shape"):
            return {"status": "error", "error": "Both objects must have shapes"}

        contacts = []
        for i, face_a in enumerate(obj_a.Shape.Faces):
            for j, face_b in enumerate(obj_b.Shape.Faces):
                try:
                    dist = face_a.distToShape(face_b)[0]
                    if dist <= tolerance:
                        contacts.append(
                            {
                                "face_a_index": i + 1,
                                "face_b_index": j + 1,
                                "distance": round(dist, 4),
                            }
                        )
                except Exception:
                    pass

        return {
            "status": "success",
            "object_a": obj_a.Name,
            "object_b": obj_b.Name,
            "tolerance": tolerance,
            "contacts": contacts,
            "contact_count": len(contacts),
        }

    def _cmd_search_properties(self, params):
        """Search object properties by pattern.

        Parameters:
            property_name: Property to search (e.g. "Height", "Volume")
            operator: Comparison operator ("gt", "lt", "eq", "gte", "lte")
            value: Value to compare against
            pattern: Glob pattern for object names (default "*")
        """
        import fnmatch

        doc = self._get_doc()
        prop_name = params.get("property_name")
        operator = params.get("operator", "gt")
        value = params.get("value")
        name_pattern = params.get("pattern", "*")

        if not prop_name:
            return {"status": "error", "error": "Missing required parameter: 'property_name'"}

        ops = {
            "gt": lambda a, b: a > b,
            "lt": lambda a, b: a < b,
            "eq": lambda a, b: abs(a - b) < 1e-6,
            "gte": lambda a, b: a >= b,
            "lte": lambda a, b: a <= b,
        }
        op_fn = ops.get(operator)
        if not op_fn:
            return {"status": "error", "error": f"Unknown operator '{operator}'. Use: gt, lt, eq, gte, lte"}

        matches = []
        for obj in doc.Objects:
            if not fnmatch.fnmatch(obj.Name, name_pattern):
                continue

            # Check direct property
            prop_val = None
            if hasattr(obj, prop_name):
                prop_val = getattr(obj, prop_name, None)
            # Check shape properties
            elif hasattr(obj, "Shape"):
                shape = obj.Shape
                if hasattr(shape, prop_name):
                    prop_val = getattr(shape, prop_name, None)
                elif prop_name == "Height":
                    prop_val = shape.BoundBox.ZLength
                elif prop_name == "Width":
                    prop_val = shape.BoundBox.XLength
                elif prop_name == "Depth":
                    prop_val = shape.BoundBox.YLength

            if prop_val is not None and isinstance(prop_val, (int, float)):
                if value is None or op_fn(prop_val, value):
                    matches.append(
                        {
                            "object": obj.Name,
                            "property": prop_name,
                            "value": round(prop_val, 4) if isinstance(prop_val, float) else prop_val,
                        }
                    )

        return {
            "status": "success",
            "property": prop_name,
            "operator": operator,
            "value": value,
            "matches": matches,
            "count": len(matches),
        }

    # ==================== Assembly Relationships ====================

    def _cmd_add_relationship(self, params):
        """Add a relationship between two objects.

        Parameters:
            rel_type: Relationship type (aligned, flush, concentric, touching, offset)
            object_a: First object name
            object_b: Second object name
            metadata: Optional extra data (e.g. offset distance)
        """
        rel_type = params.get("rel_type")
        object_a = params.get("object_a")
        object_b = params.get("object_b")
        metadata = params.get("metadata", {})

        if not rel_type or not object_a or not object_b:
            return {"status": "error", "error": "Missing required: rel_type, object_a, object_b"}

        rel = {
            "id": len(self._relationships) + 1,
            "rel_type": rel_type,
            "object_a": object_a,
            "object_b": object_b,
            "metadata": metadata,
        }
        self._relationships.append(rel)

        return {"status": "success", "relationship": rel}

    def _cmd_get_relationships(self, params):
        """Return stored relationships.

        Parameters:
            object_name: Optional filter by object name
        """
        obj_filter = params.get("object_name")
        if obj_filter:
            filtered = [r for r in self._relationships if r["object_a"] == obj_filter or r["object_b"] == obj_filter]
        else:
            filtered = self._relationships

        return {
            "status": "success",
            "relationships": filtered,
            "count": len(filtered),
        }

    def _cmd_check_relationships(self, params):
        """Validate that stored relationships still hold geometrically."""
        doc = self._get_doc()
        results = []

        for rel in self._relationships:
            obj_a = doc.getObject(rel["object_a"])
            obj_b = doc.getObject(rel["object_b"])
            valid = False
            reason = ""

            if not obj_a or not obj_b:
                reason = "Object(s) no longer exist"
            elif not hasattr(obj_a, "Shape") or not hasattr(obj_b, "Shape"):
                reason = "Object(s) have no shape"
            else:
                try:
                    if rel["rel_type"] == "touching":
                        dist = obj_a.Shape.distToShape(obj_b.Shape)[0]
                        valid = dist < 0.1
                        if not valid:
                            reason = f"Distance is {dist:.4f}, expected < 0.1"
                    elif rel["rel_type"] == "concentric":
                        ca = obj_a.Shape.BoundBox.Center
                        cb = obj_b.Shape.BoundBox.Center
                        lateral = ((ca.x - cb.x) ** 2 + (ca.y - cb.y) ** 2) ** 0.5
                        valid = lateral < 0.1
                        if not valid:
                            reason = f"Centers offset by {lateral:.4f} laterally"
                    elif rel["rel_type"] == "aligned":
                        # Check if objects share a common axis line
                        ca = obj_a.Shape.BoundBox.Center
                        cb = obj_b.Shape.BoundBox.Center
                        valid = True
                        reason = "Alignment check passed (center-based)"
                    elif rel["rel_type"] == "flush":
                        # Check if any face pair is co-planar and close
                        dist = obj_a.Shape.distToShape(obj_b.Shape)[0]
                        valid = dist < 0.01
                        if not valid:
                            reason = f"Shapes are {dist:.4f} apart, expected flush"
                    else:
                        valid = True
                        reason = "Unknown rel_type, assumed OK"
                except Exception as e:
                    reason = str(e)

            results.append(
                {
                    "relationship": rel,
                    "valid": valid,
                    "reason": reason,
                }
            )

        valid_count = sum(1 for r in results if r["valid"])
        return {
            "status": "success",
            "results": results,
            "summary": {
                "total": len(results),
                "valid": valid_count,
                "invalid": len(results) - valid_count,
            },
        }

    def _cmd_detect_relationships(self, params):
        """Auto-detect relationships between objects.

        Parameters:
            object_a: Optional first object (detects for all pairs if omitted)
            object_b: Optional second object
            tolerance: Distance tolerance (default 0.5)
        """
        doc = self._get_doc()
        tolerance = params.get("tolerance", 0.5)
        obj_a_name = params.get("object_a")
        obj_b_name = params.get("object_b")

        shape_objects = [o for o in doc.Objects if hasattr(o, "Shape") and not o.Shape.isNull()]

        # Build pairs to check
        pairs = []
        if obj_a_name and obj_b_name:
            a = doc.getObject(obj_a_name)
            b = doc.getObject(obj_b_name)
            if a and b:
                pairs.append((a, b))
        else:
            for i in range(len(shape_objects)):
                for j in range(i + 1, len(shape_objects)):
                    pairs.append((shape_objects[i], shape_objects[j]))

        detected = []
        for a, b in pairs:
            try:
                dist = a.Shape.distToShape(b.Shape)[0]
                ca = a.Shape.BoundBox.Center
                cb = b.Shape.BoundBox.Center

                # Touching
                if dist < tolerance:
                    detected.append(
                        {
                            "rel_type": "touching",
                            "object_a": a.Name,
                            "object_b": b.Name,
                            "confidence": "high" if dist < 0.01 else "medium",
                            "distance": round(dist, 4),
                        }
                    )

                # Concentric (XY centers close)
                lateral = ((ca.x - cb.x) ** 2 + (ca.y - cb.y) ** 2) ** 0.5
                if lateral < tolerance:
                    detected.append(
                        {
                            "rel_type": "concentric",
                            "object_a": a.Name,
                            "object_b": b.Name,
                            "confidence": "high" if lateral < 0.01 else "medium",
                            "lateral_offset": round(lateral, 4),
                        }
                    )

                # Aligned (share an axis — one dimension very close)
                dx = abs(ca.x - cb.x)
                dy = abs(ca.y - cb.y)
                dz = abs(ca.z - cb.z)
                if (
                    (dx < tolerance and dy < tolerance)
                    or (dx < tolerance and dz < tolerance)
                    or (dy < tolerance and dz < tolerance)
                ):
                    detected.append(
                        {
                            "rel_type": "aligned",
                            "object_a": a.Name,
                            "object_b": b.Name,
                            "confidence": "medium",
                        }
                    )
            except Exception:
                pass

        return {
            "status": "success",
            "detected": detected,
            "count": len(detected),
            "pairs_checked": len(pairs),
        }

    def _cmd_suggest_relationships(self, params):
        """Analyze object pairs and suggest likely relationships.

        Thin wrapper over detect_relationships with explanations.
        """
        detect_result = self._cmd_detect_relationships(params)
        if detect_result.get("status") != "success":
            return detect_result

        suggestions = []
        for det in detect_result.get("detected", []):
            suggestion = {
                "rel_type": det["rel_type"],
                "object_a": det["object_a"],
                "object_b": det["object_b"],
                "confidence": det.get("confidence", "medium"),
            }
            if det["rel_type"] == "touching":
                suggestion["reason"] = f"Objects are {det.get('distance', 0)} apart"
            elif det["rel_type"] == "concentric":
                suggestion["reason"] = f"Centers are {det.get('lateral_offset', 0)} apart laterally"
            elif det["rel_type"] == "aligned":
                suggestion["reason"] = "Object centers share at least two close axes"
            suggestions.append(suggestion)

        return {
            "status": "success",
            "suggestions": suggestions,
            "count": len(suggestions),
        }

    def _cmd_auto_add_detected(self, params):
        """Detect relationships and add them in one step.

        Parameters:
            tolerance: Distance tolerance (default 0.5)
            min_confidence: Minimum confidence to add ("high" or "medium", default "high")
        """
        detect_result = self._cmd_detect_relationships(params)
        if detect_result.get("status") != "success":
            return detect_result

        min_confidence = params.get("min_confidence", "high")
        confidence_levels = {"high": 2, "medium": 1, "low": 0}
        min_level = confidence_levels.get(min_confidence, 2)

        added = []
        for det in detect_result.get("detected", []):
            det_level = confidence_levels.get(det.get("confidence", "low"), 0)
            if det_level >= min_level:
                rel = {
                    "id": len(self._relationships) + 1,
                    "rel_type": det["rel_type"],
                    "object_a": det["object_a"],
                    "object_b": det["object_b"],
                    "metadata": {"auto_detected": True, "confidence": det.get("confidence")},
                }
                self._relationships.append(rel)
                added.append(rel)

        return {
            "status": "success",
            "added": added,
            "added_count": len(added),
            "total_detected": len(detect_result.get("detected", [])),
        }

    # ==================== Design State & Sampling ====================

    def _cmd_capture_design_state(self, params):
        """Serialize current object names, positions, bounding boxes to a snapshot dict."""
        doc = self._get_doc()
        snapshot = {
            "document": doc.Name,
            "objects": [],
        }

        for obj in doc.Objects:
            obj_data = {
                "name": obj.Name,
                "label": obj.Label,
                "type": obj.TypeId,
            }
            if hasattr(obj, "Placement"):
                pos = obj.Placement.Base
                rot = obj.Placement.Rotation
                obj_data["position"] = [round(pos.x, 4), round(pos.y, 4), round(pos.z, 4)]
                obj_data["rotation"] = [
                    round(rot.Axis.x, 4),
                    round(rot.Axis.y, 4),
                    round(rot.Axis.z, 4),
                    round(rot.Angle, 4),
                ]
            if hasattr(obj, "Shape") and not obj.Shape.isNull():
                bb = obj.Shape.BoundBox
                obj_data["bounding_box"] = {
                    "min": [round(bb.XMin, 4), round(bb.YMin, 4), round(bb.ZMin, 4)],
                    "max": [round(bb.XMax, 4), round(bb.YMax, 4), round(bb.ZMax, 4)],
                    "size": [round(bb.XLength, 4), round(bb.YLength, 4), round(bb.ZLength, 4)],
                }
                if hasattr(obj.Shape, "Volume"):
                    obj_data["volume"] = round(obj.Shape.Volume, 4)
            snapshot["objects"].append(obj_data)

        snapshot["object_count"] = len(snapshot["objects"])
        return {"status": "success", "snapshot": snapshot}

    def _cmd_sample_geometry(self, params):
        """Return sample points on object surface using discretized edges.

        Parameters:
            object_name: Target object
            num_points: Approximate number of points (default 50)
        """
        obj, error = self._get_object(params)
        if error:
            return error

        if not hasattr(obj, "Shape") or obj.Shape.isNull():
            return {"status": "error", "error": f"Object '{obj.Name}' has no valid shape"}

        num_points = params.get("num_points", 50)
        shape = obj.Shape

        points = []
        # Sample from edges
        total_length = sum(e.Length for e in shape.Edges)
        for edge in shape.Edges:
            if total_length < 1e-6:
                break
            edge_points = max(1, int(num_points * (edge.Length / total_length)))
            try:
                pts = edge.discretize(edge_points + 1)
                for pt in pts:
                    points.append([round(pt.x, 4), round(pt.y, 4), round(pt.z, 4)])
            except Exception:
                pass

        # Also sample face centers
        for face in shape.Faces:
            c = face.CenterOfMass
            points.append([round(c.x, 4), round(c.y, 4), round(c.z, 4)])

        return {
            "status": "success",
            "object_name": obj.Name,
            "points": points[: num_points * 2],
            "point_count": min(len(points), num_points * 2),
        }

    def _cmd_sample_mesh_surface(self, params):
        """Tessellate object and return mesh vertices/normals.

        Parameters:
            object_name: Target object
            tolerance: Tessellation tolerance (default 0.1)
            max_vertices: Max vertices to return (default 500)
        """
        obj, error = self._get_object(params)
        if error:
            return error

        if not hasattr(obj, "Shape") or obj.Shape.isNull():
            return {"status": "error", "error": f"Object '{obj.Name}' has no valid shape"}

        tolerance = params.get("tolerance", 0.1)
        max_vertices = params.get("max_vertices", 500)

        try:
            mesh_data = obj.Shape.tessellate(tolerance)
            vertices = mesh_data[0]
            triangles = mesh_data[1]

            vert_list = []
            for v in vertices[:max_vertices]:
                vert_list.append([round(v.x, 4), round(v.y, 4), round(v.z, 4)])

            tri_list = []
            for t in triangles[:max_vertices]:
                tri_list.append(list(t))

            return {
                "status": "success",
                "object_name": obj.Name,
                "vertices": vert_list,
                "triangles": tri_list,
                "vertex_count": len(vert_list),
                "triangle_count": len(tri_list),
                "total_vertices": len(vertices),
                "total_triangles": len(triangles),
                "tolerance": tolerance,
            }
        except Exception as e:
            return {"status": "error", "error": f"Tessellation failed: {e}"}

    # ==================== View Info ====================

    def _cmd_get_view_info(self, params):
        """Return current camera position, direction, zoom, and view type."""
        if not HAS_GUI:
            return {"status": "error", "error": "No GUI available"}

        try:
            view = Gui.ActiveDocument.ActiveView
            cam = view.getCameraNode()

            # Get camera position and orientation
            pos = cam.position.getValue()
            orient = cam.orientation.getValue()
            # Get near/far and focal distance
            near = cam.nearDistance.getValue()
            far_dist = cam.farDistance.getValue()

            # Determine camera type
            cam_type = cam.getTypeId().getName()
            is_perspective = "Perspective" in cam_type

            result = {
                "status": "success",
                "camera_position": [round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)],
                "camera_orientation": [
                    round(orient[0], 4),
                    round(orient[1], 4),
                    round(orient[2], 4),
                    round(orient[3], 4),
                ],
                "near_distance": round(near, 2),
                "far_distance": round(far_dist, 2),
                "perspective": is_perspective,
                "camera_type": cam_type,
            }

            if is_perspective:
                result["height_angle"] = round(cam.heightAngle.getValue(), 4)
            else:
                result["height"] = round(cam.height.getValue(), 2)

            return result
        except Exception as e:
            return {"status": "error", "error": f"Failed to get view info: {e}"}

    # ==================== Import STL ====================

    def _cmd_import_stl(self, params):
        """Import an STL file into the document.

        Parameters:
            filepath: Path to the STL file
            name: Optional name for the imported object
            tolerance: Mesh-to-solid sewing tolerance (default 0.05)
        """
        import os

        import Mesh

        filepath = params.get("filepath") or params.get("input_path")
        name = params.get("name", "ImportedSTL")
        tolerance = params.get("tolerance", 0.05)

        if not filepath:
            return {"status": "error", "error": "Missing required parameter: 'filepath'"}

        if not os.path.exists(filepath):
            return {"status": "error", "error": f"File not found: {filepath}"}

        doc = self._get_doc()

        try:
            Mesh.insert(filepath, doc.Name)
            self._safe_recompute(doc)

            # Find the newly inserted mesh object (last added)
            mesh_obj = None
            for obj in reversed(doc.Objects):
                if hasattr(obj, "Mesh"):
                    mesh_obj = obj
                    break

            if mesh_obj and name != "ImportedSTL":
                mesh_obj.Label = name

            # Try to convert to Part::Feature for boolean compatibility
            converted = False
            conversion_error = None
            part_name = name
            try:
                import Part

                if mesh_obj:
                    shape = Part.Shape()
                    shape.makeShapeFromMesh(mesh_obj.Mesh.Topology, tolerance)
                    solid = Part.makeSolid(shape)
                    part_obj = doc.addObject("Part::Feature", name)
                    part_obj.Shape = solid
                    doc.removeObject(mesh_obj.Name)
                    self._safe_recompute(doc)
                    part_name = part_obj.Name
                    converted = True
            except Exception as conv_err:
                conversion_error = str(conv_err)
                part_name = mesh_obj.Name if mesh_obj else name

            result = {
                "status": "success",
                "name": part_name,
                "filepath": filepath,
                "converted_to_solid": converted,
            }
            if conversion_error:
                result["conversion_warning"] = (
                    f"Mesh kept as-is (solid conversion failed: {conversion_error}). "
                    f"Try adjusting 'tolerance' (current: {tolerance})."
                )
            return result
        except Exception as e:
            return {"status": "error", "error": f"STL import failed: {e}"}

    # ==================== Face Index Resolution Helper ====================

    def _resolve_face_index(self, obj, face_selector):
        """Resolve a face selector (string name or integer index) to a 1-based face index."""
        if isinstance(face_selector, int):
            if 1 <= face_selector <= len(obj.Shape.Faces):
                return face_selector
            return None
        if isinstance(face_selector, str):
            # Try as integer string
            try:
                idx = int(face_selector)
                if 1 <= idx <= len(obj.Shape.Faces):
                    return idx
                return None
            except ValueError:
                pass
            # Try as semantic direction
            return self._get_face_by_direction(obj, face_selector)
        return None

    # ==================== Export Commands ====================

    def _cmd_export_stl(self, params):
        """Export objects to STL files.

        Supports:
        - Single object via object_name or multiple via object_names
        - ["all"] to export all visible objects
        - Empty object_names with output_path defaults to all visible objects
        - combine=True to merge into single STL
        - combine=False to export individual files to directory
        """
        import os

        import Mesh

        doc = self._get_doc()
        # Accept singular, plural, and legacy parameter names
        object_names = params.get("object_names") or params.get("objects", [])
        single_name = params.get("object_name")
        if single_name and not object_names:
            object_names = [single_name]
        output_path = params.get("output_path") or params.get("filepath")
        # Accept both tolerance and precision
        tolerance = params.get("tolerance") or params.get("precision", 0.05)
        combine = params.get("combine", True)

        # Handle ["all"] or default to all visible when no objects specified
        if object_names == ["all"] or object_names == "all" or (not object_names and output_path):
            object_names = []
            for obj in doc.Objects:
                if hasattr(obj, "Shape") and hasattr(obj, "ViewObject"):
                    if obj.ViewObject and obj.ViewObject.Visibility:
                        object_names.append(obj.Name)
            if not object_names:
                all_objs = [o.Name for o in doc.Objects if hasattr(o, "Shape")]
                return {
                    "status": "error",
                    "error": "No visible objects with shapes found",
                    "all_shape_objects": all_objs,
                    "hint": "All shape objects are hidden. Use set_visibility or pass explicit object_names.",
                }

        if not object_names:
            return {
                "status": "error",
                "error": "No object_name or object_names specified and no output_path for auto-detect",
            }

        # Collect shapes - try to export even if isValid() returns False
        # Complex boolean results sometimes report invalid but still tessellate fine
        shapes = []
        shape_names = []
        skipped = []
        for name in object_names:
            obj = doc.getObject(name)
            if not obj:
                skipped.append(
                    {
                        "name": name,
                        "reason": f"object '{name}' not found in document (available: {[o.Name for o in doc.Objects][:10]})",
                    }
                )
                continue
            if not hasattr(obj, "Shape"):
                skipped.append({"name": name, "reason": f"no Shape attribute (type: {obj.TypeId})"})
                continue
            # Try to use shape even if not "valid" - tessellation may still work
            try:
                shape = obj.Shape
                if shape.isNull():
                    skipped.append({"name": name, "reason": "null shape (shape exists but contains no geometry)"})
                    continue
                shapes.append(shape)
                shape_names.append(name)
            except Exception as e:
                skipped.append({"name": name, "reason": f"shape access error: {str(e)}"})

        if not shapes:
            return {
                "status": "error",
                "error": f"No valid objects to export from {len(object_names)} requested: {object_names}",
                "skipped": skipped,
            }

        exported_files = []
        total_facets = 0
        export_errors = []

        if combine:
            # Combine all shapes into single STL by merging meshes (not boolean fuse - much faster)
            try:
                mesh = Mesh.Mesh()
                for shape in shapes:
                    mesh.addFacets(shape.tessellate(tolerance))
                mesh.write(output_path)
                facet_count = mesh.CountFacets
                total_facets = facet_count
                exported_files.append({"path": output_path, "objects": shape_names, "facets": facet_count})
            except Exception as e:
                return {"status": "error", "error": f"Failed to export combined STL: {str(e)}", "skipped": skipped}
        else:
            # Export individual files to directory
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            for shape, name in zip(shapes, shape_names):
                file_path = os.path.join(output_path, f"{name}.stl")
                try:
                    mesh = Mesh.Mesh()
                    mesh.addFacets(shape.tessellate(tolerance))
                    mesh.write(file_path)
                    facet_count = mesh.CountFacets
                    total_facets += facet_count
                    exported_files.append({"path": file_path, "object": name, "facets": facet_count})
                except Exception as e:
                    export_errors.append({"object": name, "error": str(e)})

        result = {
            "status": "success",
            "files": exported_files,
            "total_files": len(exported_files),
            "total_facets": total_facets,
            "tolerance": tolerance,
        }
        if skipped:
            result["skipped"] = skipped
        if export_errors:
            result["export_errors"] = export_errors
        return result

    def _cmd_export_step(self, params):
        """Export to STEP."""
        doc = self._get_doc()
        objects = params.get("objects", [])
        filepath = params.get("filepath")

        shapes = []
        for name in objects:
            obj = doc.getObject(name)
            if obj and hasattr(obj, "Shape"):
                shapes.append(obj.Shape)

        if shapes:
            combined = shapes[0]
            for shape in shapes[1:]:
                combined = combined.fuse(shape)
            combined.exportStep(filepath)
            return {"status": "success", "filepath": filepath}
        return {"status": "error", "error": "No valid objects to export"}

    def _cmd_export_formatted(self, params):
        """Export objects to multiple formats with batch capability.

        Parameters:
            object_names: List of object names (or ["all"] for all visible)
            output_dir: Output directory path
            formats: List of formats - "STL", "STEP", "IGES", "3MF" (default ["STL"])
            tolerance: Mesh tolerance for STL in mm (default 0.05)
            combine: If True, combine all objects into single file per format
        """
        import os

        doc = self._get_doc()
        object_names = params.get("object_names", [])
        output_dir = params.get("output_dir", "/tmp")
        formats = [f.upper() for f in params.get("formats", ["STL"])]
        tolerance = params.get("tolerance", 0.05)
        combine = params.get("combine", False)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Resolve ["all"]
        if object_names == ["all"] or object_names == "all":
            object_names = [
                obj.Name
                for obj in doc.Objects
                if hasattr(obj, "Shape") and hasattr(obj, "ViewObject") and obj.ViewObject and obj.ViewObject.Visibility
            ]

        # Collect shapes
        shapes = []
        shape_names = []
        for name in object_names:
            obj = doc.getObject(name)
            if obj and hasattr(obj, "Shape") and not obj.Shape.isNull():
                shapes.append(obj.Shape)
                shape_names.append(name)

        if not shapes:
            return {"status": "error", "error": "No valid objects to export"}

        exported = []
        errors = []

        for fmt in formats:
            try:
                if fmt == "STL":
                    import Mesh

                    if combine:
                        path = os.path.join(output_dir, "combined.stl")
                        mesh = Mesh.Mesh()
                        for s in shapes:
                            mesh.addFacets(s.tessellate(tolerance))
                        mesh.write(path)
                        exported.append(
                            {"format": "STL", "path": path, "objects": shape_names, "facets": mesh.CountFacets}
                        )
                    else:
                        for s, n in zip(shapes, shape_names):
                            path = os.path.join(output_dir, f"{n}.stl")
                            mesh = Mesh.Mesh()
                            mesh.addFacets(s.tessellate(tolerance))
                            mesh.write(path)
                            exported.append({"format": "STL", "path": path, "object": n, "facets": mesh.CountFacets})

                elif fmt == "STEP":
                    if combine:
                        path = os.path.join(output_dir, "combined.step")
                        merged = shapes[0]
                        for s in shapes[1:]:
                            merged = merged.fuse(s)
                        merged.exportStep(path)
                        exported.append({"format": "STEP", "path": path, "objects": shape_names})
                    else:
                        for s, n in zip(shapes, shape_names):
                            path = os.path.join(output_dir, f"{n}.step")
                            s.exportStep(path)
                            exported.append({"format": "STEP", "path": path, "object": n})

                elif fmt == "IGES":
                    if combine:
                        path = os.path.join(output_dir, "combined.iges")
                        merged = shapes[0]
                        for s in shapes[1:]:
                            merged = merged.fuse(s)
                        merged.exportIges(path)
                        exported.append({"format": "IGES", "path": path, "objects": shape_names})
                    else:
                        for s, n in zip(shapes, shape_names):
                            path = os.path.join(output_dir, f"{n}.iges")
                            s.exportIges(path)
                            exported.append({"format": "IGES", "path": path, "object": n})

                else:
                    errors.append({"format": fmt, "error": f"Unsupported format: {fmt}"})

            except Exception as e:
                errors.append({"format": fmt, "error": str(e)})

        result = {
            "status": "success",
            "files": exported,
            "total_files": len(exported),
        }
        if errors:
            result["errors"] = errors
        return result

    # ==================== View Commands ====================

    def _cmd_set_visibility(self, params):
        """Set visibility of one or more objects.

        Parameters:
            object_name: Single object name (str)
            object_names: List of object names (list)
            visible: True to show, False to hide (default True)
            solo: If True, hide all other objects and show only named ones (default False)
        """
        if not HAS_GUI:
            return {"status": "error", "error": "No GUI available for visibility control"}

        doc = self._get_doc()
        visible = params.get("visible", True)
        solo = params.get("solo", False)

        # Accept both singular and plural
        names = params.get("object_names") or []
        single = params.get("object_name")
        if single and single not in names:
            names.append(single)

        if not names:
            return {"status": "error", "error": "No object_name or object_names specified"}

        # Validate all names exist
        not_found = [n for n in names if not doc.getObject(n)]
        if not_found:
            return {"status": "error", "error": f"Objects not found: {not_found}"}

        affected = []

        if solo:
            # Hide everything, then show only named objects
            for obj in doc.Objects:
                if hasattr(obj, "ViewObject") and obj.ViewObject:
                    was_visible = obj.ViewObject.Visibility
                    if obj.Name in names:
                        obj.ViewObject.Visibility = True
                        if not was_visible:
                            affected.append({"name": obj.Name, "visible": True})
                    else:
                        obj.ViewObject.Visibility = False
                        if was_visible:
                            affected.append({"name": obj.Name, "visible": False})
        else:
            for name in names:
                obj = doc.getObject(name)
                if hasattr(obj, "ViewObject") and obj.ViewObject:
                    obj.ViewObject.Visibility = visible
                    affected.append({"name": name, "visible": visible})

        return {
            "status": "success",
            "affected": affected,
            "total_affected": len(affected),
        }

    def _cmd_set_view(self, params):
        """Set view direction."""
        if not HAS_GUI:
            return {"status": "error", "error": "No GUI available"}

        direction = params.get("direction", "isometric")
        view = Gui.ActiveDocument.ActiveView

        view_map = {
            "top": "viewTop",
            "bottom": "viewBottom",
            "front": "viewFront",
            "back": "viewRear",
            "left": "viewLeft",
            "right": "viewRight",
            "isometric": "viewIsometric",
        }

        method = getattr(view, view_map.get(direction.lower(), "viewIsometric"), None)
        if method:
            method()
        view.fitAll()
        return {"status": "success", "view": direction}

    def _cmd_capture_view(self, params):
        """Capture screenshot of current view.

        Parameters:
            output_path/filepath: Path to save the image (PNG format)
            width: Image width in pixels (default 800)
            height: Image height in pixels (default 600)
            view: View angle - "iso", "front", "back", "top", "bottom", "left", "right", "current"
            background: Background color - "white", "black", "gradient", "transparent"
        """
        if not HAS_GUI:
            return {"status": "error", "error": "No GUI available"}

        # Accept both 'output_path' (MCP) and 'filepath' (legacy) parameter names
        filepath = params.get("output_path") or params.get("filepath")
        if not filepath:
            return {"status": "error", "error": "No output path specified (use 'output_path' or 'filepath')"}

        width = params.get("width", 800)
        height = params.get("height", 600)
        view_direction = params.get("view", "current")
        background = params.get("background", "white")

        try:
            active_view = Gui.ActiveDocument.ActiveView

            # Set view direction using direct view methods
            view_methods = {
                "iso": "viewIsometric",
                "isometric": "viewIsometric",
                "front": "viewFront",
                "back": "viewRear",
                "rear": "viewRear",
                "top": "viewTop",
                "bottom": "viewBottom",
                "left": "viewLeft",
                "right": "viewRight",
            }

            view_key = view_direction.lower()
            if view_key in view_methods:
                method_name = view_methods[view_key]
                try:
                    method = getattr(active_view, method_name, None)
                    if method:
                        method()
                        Gui.updateGui()
                except Exception:
                    pass  # View change is optional, continue with capture

            # Fit all objects in view
            try:
                active_view.fitAll()
                Gui.updateGui()
            except Exception:
                pass

            # Map background parameter to FreeCAD saveImage format
            bg_map = {
                "white": "white",
                "black": "black",
                "transparent": "Transparent",
                "gradient": "Current",  # Uses FreeCAD's current gradient
            }
            bg_color = bg_map.get(background.lower(), "white")

            # Capture the image with background color
            active_view.saveImage(str(filepath), int(width), int(height), bg_color)

            return {
                "status": "success",
                "filepath": filepath,
                "width": width,
                "height": height,
                "view": view_direction,
                "background": background,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_capture_object(self, params):
        """Capture a single object as an image.

        Isolates the object by hiding others, captures the view, then restores visibility.

        Parameters:
            object_name: Name of object to capture
            output_path: Path to save the image
            width: Image width in pixels (default 600)
            height: Image height in pixels (default 600)
            padding: Zoom padding factor (default 1.2)
            view: View angle - "iso", "front", "top", "right"
            background: Background color - "white", "black", "gradient", "transparent"
            highlight: Apply highlight color to object (default False)
        """
        if not HAS_GUI:
            return {"status": "error", "error": "No GUI available"}

        object_name = params.get("object_name")
        filepath = params.get("output_path") or params.get("filepath")
        if not object_name:
            return {"status": "error", "error": "No object_name specified"}
        if not filepath:
            return {"status": "error", "error": "No output path specified"}

        width = params.get("width", 600)
        height = params.get("height", 600)
        view_direction = params.get("view", "iso")
        background = params.get("background", "white")
        highlight = params.get("highlight", False)

        doc = self._get_doc()
        target_obj = doc.getObject(object_name)
        if not target_obj:
            return {"status": "error", "error": f"Object '{object_name}' not found"}

        original_color = None
        try:
            # Store visibility states and hide all except target
            visibility_states = {}
            for obj in doc.Objects:
                if obj.ViewObject:
                    visibility_states[obj.Name] = obj.ViewObject.Visibility
                    obj.ViewObject.Visibility = obj.Name == object_name

            # Apply highlight color if requested
            if highlight and target_obj.ViewObject:
                original_color = target_obj.ViewObject.ShapeColor
                target_obj.ViewObject.ShapeColor = (0.2, 0.6, 0.9)  # Highlight blue
                Gui.Selection.addSelection(target_obj)

            Gui.updateGui()

            # Get active view and set direction using direct methods
            active_view = Gui.ActiveDocument.ActiveView
            view_methods = {
                "iso": "viewIsometric",
                "isometric": "viewIsometric",
                "front": "viewFront",
                "top": "viewTop",
                "right": "viewRight",
            }

            view_key = view_direction.lower()
            if view_key in view_methods:
                method_name = view_methods[view_key]
                try:
                    method = getattr(active_view, method_name, None)
                    if method:
                        method()
                        Gui.updateGui()
                except Exception:
                    pass

            # Fit to the visible object
            try:
                active_view.fitAll()
                Gui.updateGui()
            except Exception:
                pass

            # Map background parameter to FreeCAD saveImage format
            bg_map = {
                "white": "white",
                "black": "black",
                "transparent": "Transparent",
                "gradient": "Current",
            }
            bg_color = bg_map.get(background.lower(), "white")

            # Capture with background color
            active_view.saveImage(str(filepath), int(width), int(height), bg_color)

            # Restore highlight color if applied
            if highlight and original_color is not None and target_obj.ViewObject:
                target_obj.ViewObject.ShapeColor = original_color
                Gui.Selection.clearSelection()

            # Restore visibility
            for obj_name, was_visible in visibility_states.items():
                obj = doc.getObject(obj_name)
                if obj and obj.ViewObject:
                    obj.ViewObject.Visibility = was_visible

            Gui.updateGui()

            return {
                "status": "success",
                "filepath": filepath,
                "object": object_name,
                "width": width,
                "height": height,
                "background": background,
                "highlighted": highlight,
            }

        except Exception as e:
            # Attempt to restore visibility and color on error
            try:
                if highlight and original_color is not None and target_obj.ViewObject:
                    target_obj.ViewObject.ShapeColor = original_color
                    Gui.Selection.clearSelection()
                for obj_name, was_visible in visibility_states.items():
                    obj = doc.getObject(obj_name)
                    if obj and obj.ViewObject:
                        obj.ViewObject.Visibility = was_visible
            except Exception:
                pass
            return {"status": "error", "error": str(e)}

    # ==================== History Commands ====================

    def _cmd_undo(self, params):
        """Undo the last operation."""
        doc = self._get_doc()
        if doc.UndoCount == 0:
            return {"status": "error", "error": "Nothing to undo"}
        name = doc.UndoNames[0] if doc.UndoNames else "unknown"
        doc.undo()
        self._safe_recompute(doc)
        return {"status": "success", "undone": name, "remaining_undos": doc.UndoCount}

    def _cmd_redo(self, params):
        """Redo the last undone operation."""
        doc = self._get_doc()
        if doc.RedoCount == 0:
            return {"status": "error", "error": "Nothing to redo"}
        name = doc.RedoNames[0] if doc.RedoNames else "unknown"
        doc.redo()
        self._safe_recompute(doc)
        return {"status": "success", "redone": name, "remaining_redos": doc.RedoCount}

    def _cmd_get_history(self, params):
        """Get undo/redo history."""
        doc = self._get_doc()
        return {
            "status": "success",
            "undo_count": doc.UndoCount,
            "redo_count": doc.RedoCount,
            "undo_names": list(doc.UndoNames) if doc.UndoNames else [],
            "redo_names": list(doc.RedoNames) if doc.RedoNames else [],
        }

    # ==================== Feature Tree Commands ====================

    def _cmd_get_feature_tree(self, params):
        """Get the full feature tree with parameters and dependencies."""
        doc = self._get_doc()
        skip_props = {"Shape", "Placement", "Label", "ExpressionEngine", "Visibility", "Label2"}
        result = []
        for obj in doc.Objects:
            entry = {
                "name": obj.Name,
                "label": obj.Label,
                "type": obj.TypeId,
                "in_list": [o.Name for o in obj.InList] if hasattr(obj, "InList") else [],
                "out_list": [o.Name for o in obj.OutList] if hasattr(obj, "OutList") else [],
            }
            # Collect editable parameters
            parameters = {}
            for prop in obj.PropertiesList:
                if prop in skip_props:
                    continue
                try:
                    val = getattr(obj, prop)
                    if isinstance(val, (int, float, str, bool)):
                        parameters[prop] = val
                    elif hasattr(val, "Value"):
                        parameters[prop] = val.Value
                except Exception:
                    continue
            entry["parameters"] = parameters
            # Volume and area
            if hasattr(obj, "Shape") and hasattr(obj.Shape, "Volume"):
                try:
                    entry["volume"] = round(obj.Shape.Volume, 4)
                    entry["area"] = round(obj.Shape.Area, 4)
                except Exception:
                    pass
            result.append(entry)
        return {"status": "success", "objects": result, "count": len(result)}

    def _cmd_modify_feature(self, params):
        """Modify an editable property on a FreeCAD object."""
        obj = self._get_object(params)
        self._require_params(params, "property_name", "value")
        prop_name = params["property_name"]
        new_value = params["value"]
        if prop_name not in obj.PropertiesList:
            return {"status": "error", "error": f"Property '{prop_name}' not found on {obj.Name}"}
        old_val = getattr(obj, prop_name)
        if hasattr(old_val, "Value"):
            old_value = old_val.Value
            old_val.Value = new_value
        else:
            old_value = old_val
            setattr(obj, prop_name, new_value)
        doc = self._get_doc()
        self._safe_recompute(doc)
        return {
            "status": "success",
            "object": obj.Name,
            "property": prop_name,
            "old_value": old_value,
            "new_value": new_value,
        }

    # ==================== Snapshot Commands ====================

    def _cmd_save_snapshot(self, params):
        """Save a named snapshot of the current document state."""
        import json as _json

        doc = self._get_doc()
        name = params.get("name", f"snapshot_{int(time.time())}")
        # Load existing snapshots
        snapshots = []
        if hasattr(doc, "Conjure_Snapshots"):
            try:
                snapshots = _json.loads(doc.Conjure_Snapshots)
            except Exception:
                snapshots = []
        else:
            doc.addProperty("App::PropertyString", "Conjure_Snapshots", "Conjure", "Snapshot data")
        # Check duplicate
        if any(s.get("name") == name for s in snapshots):
            return {"status": "error", "error": f"Snapshot '{name}' already exists"}
        # Capture state
        objects = []
        for obj in doc.Objects:
            obj_data = {"name": obj.Name, "label": obj.Label, "type": obj.TypeId}
            if hasattr(obj, "Shape") and hasattr(obj.Shape, "Volume"):
                try:
                    obj_data["volume"] = round(obj.Shape.Volume, 4)
                except Exception:
                    pass
            if hasattr(obj, "Placement"):
                try:
                    pos = obj.Placement.Base
                    obj_data["position"] = [pos.x, pos.y, pos.z]
                except Exception:
                    pass
            objects.append(obj_data)
        snapshot = {
            "name": name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "undo_count_at_save": doc.UndoCount,
            "objects": objects,
        }
        snapshots.append(snapshot)
        doc.Conjure_Snapshots = _json.dumps(snapshots)
        return {"status": "success", "snapshot": snapshot}

    def _cmd_list_snapshots(self, params):
        """List all saved snapshots."""
        import json as _json

        doc = self._get_doc()
        if not hasattr(doc, "Conjure_Snapshots"):
            return {"status": "success", "snapshots": []}
        try:
            snapshots = _json.loads(doc.Conjure_Snapshots)
        except Exception:
            snapshots = []
        return {"status": "success", "snapshots": snapshots}

    def _cmd_restore_snapshot(self, params):
        """Restore a previously saved snapshot using undo operations."""
        import json as _json

        doc = self._get_doc()
        self._require_params(params, "name")
        name = params["name"]
        if not hasattr(doc, "Conjure_Snapshots"):
            return {"status": "error", "error": "No snapshots found"}
        try:
            snapshots = _json.loads(doc.Conjure_Snapshots)
        except Exception:
            return {"status": "error", "error": "Failed to read snapshots"}
        target = None
        for s in snapshots:
            if s.get("name") == name:
                target = s
                break
        if not target:
            return {"status": "error", "error": f"Snapshot '{name}' not found"}
        saved_undo = target.get("undo_count_at_save")
        if saved_undo is None:
            return {
                "status": "warning",
                "message": "Snapshot has no undo reference, cannot restore",
                "snapshot": target,
            }
        current_undo = doc.UndoCount
        steps = current_undo - saved_undo
        if steps <= 0:
            return {
                "status": "warning",
                "message": "No undo steps to apply (already at or before snapshot)",
                "snapshot": target,
            }
        if steps > current_undo:
            return {
                "status": "warning",
                "message": f"Need {steps} undos but only {current_undo} available",
                "snapshot": target,
            }
        for _ in range(steps):
            doc.undo()
        self._safe_recompute(doc)
        return {
            "status": "success",
            "message": f"Restored snapshot '{name}' ({steps} undos applied)",
            "snapshot": target,
        }

    # ==================== Script Command ====================

    @staticmethod
    def _serialize_value(value):
        """Serialize a value for JSON transport, handling FreeCAD types.

        Converts FreeCAD-specific types (Vector, Placement, BoundBox, Shape)
        to JSON-serializable representations. Falls back to str() for
        unrecognized types.

        Args:
            value: Any Python value to serialize

        Returns:
            A JSON-serializable representation of the value
        """
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [ConjureServer._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): ConjureServer._serialize_value(v) for k, v in value.items()}
        # FreeCAD Vector
        if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z") and type(value).__name__ == "Vector":
            return {"x": value.x, "y": value.y, "z": value.z}
        # FreeCAD Placement
        if hasattr(value, "Base") and hasattr(value, "Rotation") and type(value).__name__ == "Placement":
            return {
                "Base": ConjureServer._serialize_value(value.Base),
                "Rotation": {"Angle": value.Rotation.Angle, "Axis": ConjureServer._serialize_value(value.Rotation.Axis)},
            }
        # FreeCAD BoundBox
        if hasattr(value, "XMin") and hasattr(value, "XMax") and type(value).__name__ == "BoundBox":
            return {
                "XMin": value.XMin, "YMin": value.YMin, "ZMin": value.ZMin,
                "XMax": value.XMax, "YMax": value.YMax, "ZMax": value.ZMax,
            }
        # FreeCAD Shape - return summary instead of full geometry
        if hasattr(value, "ShapeType") and hasattr(value, "Volume"):
            return {"ShapeType": value.ShapeType, "Volume": value.Volume}
        # Fallback: convert to string
        return str(value)

    def _cmd_run_script(self, params):
        """Run arbitrary Python script (escape hatch) with timeout protection.

        After execution, auto-captures:
        - The 'result' dict (explicit results set by the script)
        - The '__return__' variable (if set by the script)
        - All new user-defined variables created during execution

        FreeCAD types (Vector, Placement, etc.) are auto-serialized to JSON.

        Set read_only=true to skip FreeCAD transaction wrapping (default: false).
        """
        import signal
        import sys

        SCRIPT_TIMEOUT = 10  # seconds

        script = params.get("script", "")

        # Basic input validation
        if not script or not isinstance(script, str):
            return {"status": "error", "error": "Script must be a non-empty string"}

        if len(script) > 50000:
            return {"status": "error", "error": "Script too long (max 50000 chars)"}

        # Block dangerous imports at string level (defense in depth)
        dangerous_patterns = ["import subprocess", "import os", "__import__", "eval(", "compile("]
        for pattern in dangerous_patterns:
            if pattern in script:
                return {"status": "error", "error": f"Blocked pattern detected: {pattern}"}

        # Timeout handler (Unix only - Windows will skip timeout)
        class ScriptTimeoutError(Exception):
            pass

        def timeout_handler(signum, frame):
            raise ScriptTimeoutError("Script execution exceeded 10s limit")

        # Try to set up signal-based timeout (Unix only)
        use_signal_timeout = hasattr(signal, "SIGALRM") and sys.platform != "win32"

        if use_signal_timeout:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(SCRIPT_TIMEOUT)

        try:
            result = {}
            # Import math for script use
            import math as _math

            # Pre-import commonly needed modules
            Part = __import__("Part") if "Part" in sys.modules else None
            Draft = __import__("Draft") if "Draft" in sys.modules else None
            Sketcher = __import__("Sketcher") if "Sketcher" in sys.modules else None

            # For local development, use fuller builtins to allow FreeCAD operations
            # The server-side validation prevents dangerous patterns like subprocess/os
            safe_builtins = {
                "True": True,
                "False": False,
                "None": None,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "reversed": reversed,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "pow": pow,
                "print": print,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "type": type,
                "object": object,
                "property": property,
                "staticmethod": staticmethod,
                "classmethod": classmethod,
                "super": super,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "RuntimeError": RuntimeError,
                # Allow __import__ for FreeCAD internal use (Part::Feature creation)
                "__import__": __import__,
            }
            exec_globals = {
                "__builtins__": safe_builtins,
                "App": App,
                "FreeCAD": App,  # Alias for convenience
                "Gui": Gui if HAS_GUI else None,
                "FreeCADGui": Gui if HAS_GUI else None,
                "result": result,
                "Part": Part,
                "Draft": Draft,
                "Sketcher": Sketcher,
                "math": _math,
            }
            # Track pre-existing keys to identify user-defined variables
            pre_keys = set(exec_globals.keys())

            exec(script, exec_globals)

            # Build captured variables from user-defined names
            captured = {}
            for key, value in exec_globals.items():
                if key in pre_keys or key.startswith("_"):
                    continue
                try:
                    captured[key] = self._serialize_value(value)
                except Exception:
                    captured[key] = str(value)

            # Build response: always include result dict, add __return__ and captured vars
            response = {"status": "success", "result": result}
            if "__return__" in exec_globals:
                try:
                    response["__return__"] = self._serialize_value(exec_globals["__return__"])
                except Exception:
                    response["__return__"] = str(exec_globals["__return__"])
            if captured:
                response["variables"] = captured
            return response
        except ScriptTimeoutError as e:
            return {"status": "error", "error": str(e)}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            if use_signal_timeout:
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)

    def _cmd_eval_expression(self, params):
        """Evaluate a Python expression and return its value.

        Unlike run_script which runs statements, this uses eval() and directly
        returns the expression result. Better for reading values.

        For multi-statement code, use run_script instead.
        """
        import signal
        import sys

        EVAL_TIMEOUT = 10  # seconds

        expression = params.get("expression", "")

        # Basic input validation
        if not expression or not isinstance(expression, str):
            return {"status": "error", "error": "Expression must be a non-empty string"}

        if len(expression) > 10000:
            return {"status": "error", "error": "Expression too long (max 10000 chars)"}

        # Block dangerous imports at string level (defense in depth)
        dangerous_patterns = ["import subprocess", "import os", "__import__", "eval(", "compile("]
        for pattern in dangerous_patterns:
            if pattern in expression:
                return {"status": "error", "error": f"Blocked pattern detected: {pattern}"}

        # Timeout handler (Unix only - Windows will skip timeout)
        class EvalTimeoutError(Exception):
            pass

        def timeout_handler(signum, frame):
            raise EvalTimeoutError("Expression evaluation exceeded 10s limit")

        # Try to set up signal-based timeout (Unix only)
        use_signal_timeout = hasattr(signal, "SIGALRM") and sys.platform != "win32"

        if use_signal_timeout:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(EVAL_TIMEOUT)

        try:
            import math as _math

            Part = __import__("Part") if "Part" in sys.modules else None
            Draft = __import__("Draft") if "Draft" in sys.modules else None
            Sketcher = __import__("Sketcher") if "Sketcher" in sys.modules else None

            safe_builtins = {
                "True": True,
                "False": False,
                "None": None,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "len": len,
                "range": range,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "pow": pow,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "type": type,
            }
            eval_globals = {
                "__builtins__": safe_builtins,
                "App": App,
                "FreeCAD": App,
                "Gui": Gui if HAS_GUI else None,
                "FreeCADGui": Gui if HAS_GUI else None,
                "Part": Part,
                "Draft": Draft,
                "Sketcher": Sketcher,
                "math": _math,
            }
            result = eval(expression, eval_globals)
            # Convert non-serializable results to str(result) for safe JSON transport
            try:
                serialized = self._serialize_value(result)
            except Exception:
                serialized = str(result)
            return {"status": "success", "result": serialized}
        except SyntaxError as e:
            return {
                "status": "error",
                "error": f"SyntaxError: {e}. Hint: eval() only supports expressions. "
                "For statements (assignments, loops, etc.), use run_script instead.",
            }
        except EvalTimeoutError as e:
            return {"status": "error", "error": str(e)}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            if use_signal_timeout:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    # ==================== Engineering Material Commands ====================

    def _cmd_list_engineering_materials(self, params):
        """List available engineering materials from the server."""
        if not self.materials_client:
            return {"status": "error", "error": "Materials client not available"}

        category = params.get("category")
        try:
            materials = self.materials_client.list_materials(category=category)
            return {
                "status": "success",
                "materials": [m.to_dict() for m in materials],
                "count": len(materials),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_get_engineering_material(self, params):
        """Get a specific engineering material by ID."""
        if not self.materials_client:
            return {"status": "error", "error": "Materials client not available"}

        material_id = params.get("material_id")
        if not material_id:
            return {"status": "error", "error": "material_id required"}

        try:
            material = self.materials_client.get_material(material_id)
            if material:
                return {"status": "success", "material": material.to_dict()}
            else:
                return {"status": "error", "error": f"Material not found: {material_id}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_assign_engineering_material(self, params):
        """Assign an engineering material to a FreeCAD object."""
        if not self.materials_client:
            return {"status": "error", "error": "Materials client not available"}

        object_name = params.get("object_name")
        material_id = params.get("material_id")

        if not object_name or not material_id:
            return {"status": "error", "error": "object_name and material_id required"}

        doc = self._get_doc()
        obj = doc.getObject(object_name)
        if not obj:
            return {"status": "error", "error": f"Object not found: {object_name}"}

        # Get the material
        material = self.materials_client.get_material(material_id)
        if not material:
            return {"status": "error", "error": f"Material not found: {material_id}"}

        # Store assignment in materials client
        self.materials_client.assign_material(object_name, material_id)

        # Store material info as custom properties on the FreeCAD object
        # FreeCAD objects support dynamic properties
        try:
            # Add custom properties if they don't exist
            if not hasattr(obj, "ConjureMaterialId"):
                obj.addProperty("App::PropertyString", "ConjureMaterialId", "Conjure", "Engineering material ID")
            if not hasattr(obj, "ConjureMaterialName"):
                obj.addProperty("App::PropertyString", "ConjureMaterialName", "Conjure", "Engineering material name")
            if not hasattr(obj, "ConjureDensity"):
                obj.addProperty("App::PropertyFloat", "ConjureDensity", "Conjure", "Density (kg/m³)")
            if not hasattr(obj, "ConjureYoungsModulus"):
                obj.addProperty("App::PropertyFloat", "ConjureYoungsModulus", "Conjure", "Young's modulus (Pa)")

            # Set property values
            obj.ConjureMaterialId = material.id
            obj.ConjureMaterialName = material.name
            if material.density_kg_m3:
                obj.ConjureDensity = material.density_kg_m3
            if material.youngs_modulus_pa:
                obj.ConjureYoungsModulus = material.youngs_modulus_pa

            self._safe_recompute(doc)

            # Apply visual material if GUI available
            if HAS_GUI and material.base_color:
                try:
                    view_obj = obj.ViewObject
                    if view_obj:
                        r, g, b = material.base_color
                        view_obj.ShapeColor = (r, g, b)
                except Exception:
                    pass  # Visual update is optional

        except Exception as e:
            return {"status": "error", "error": f"Failed to set properties: {e}"}

        return {
            "status": "success",
            "object": object_name,
            "material_id": material_id,
            "material_name": material.name,
            "density_kg_m3": material.density_kg_m3,
        }

    def _cmd_get_object_engineering_material(self, params):
        """Get the engineering material assigned to an object."""
        object_name = params.get("object_name")
        if not object_name:
            return {"status": "error", "error": "object_name required"}

        doc = self._get_doc()
        obj = doc.getObject(object_name)
        if not obj:
            return {"status": "error", "error": f"Object not found: {object_name}"}

        # Check for material properties on the object
        if hasattr(obj, "ConjureMaterialId") and obj.ConjureMaterialId:
            material_id = obj.ConjureMaterialId
            material_name = getattr(obj, "ConjureMaterialName", "")
            density = getattr(obj, "ConjureDensity", None)
            youngs_modulus = getattr(obj, "ConjureYoungsModulus", None)

            # Try to get full material from cache
            if self.materials_client:
                material = self.materials_client.get_material(material_id)
                if material:
                    return {
                        "status": "success",
                        "object": object_name,
                        "material": material.to_dict(),
                    }

            # Fallback to stored properties
            return {
                "status": "success",
                "object": object_name,
                "material": {
                    "id": material_id,
                    "name": material_name,
                    "density_kg_m3": density,
                    "youngs_modulus_pa": youngs_modulus,
                },
            }
        else:
            return {"status": "success", "object": object_name, "material": None}

    def _cmd_clear_engineering_material(self, params):
        """Clear the engineering material assignment from an object."""
        object_name = params.get("object_name")
        if not object_name:
            return {"status": "error", "error": "object_name required"}

        doc = self._get_doc()
        obj = doc.getObject(object_name)
        if not obj:
            return {"status": "error", "error": f"Object not found: {object_name}"}

        # Clear from materials client
        if self.materials_client:
            self.materials_client.clear_object_material(object_name)

        # Clear custom properties
        try:
            if hasattr(obj, "ConjureMaterialId"):
                obj.ConjureMaterialId = ""
            if hasattr(obj, "ConjureMaterialName"):
                obj.ConjureMaterialName = ""
            if hasattr(obj, "ConjureDensity"):
                obj.ConjureDensity = 0.0
            if hasattr(obj, "ConjureYoungsModulus"):
                obj.ConjureYoungsModulus = 0.0

            # Reset color to default
            if HAS_GUI:
                try:
                    view_obj = obj.ViewObject
                    if view_obj:
                        view_obj.ShapeColor = (0.8, 0.8, 0.8)  # Default gray
                except Exception:
                    pass

            self._safe_recompute(doc)
        except Exception as e:
            return {"status": "error", "error": f"Failed to clear properties: {e}"}

        return {"status": "success", "object": object_name, "cleared": True}

    def _cmd_refresh_materials_cache(self, params):
        """Force refresh of the materials cache."""
        if not self.materials_client:
            return {"status": "error", "error": "Materials client not available"}

        try:
            success = self.materials_client.refresh_cache()
            if success:
                materials = self.materials_client.list_materials()
                return {
                    "status": "success",
                    "refreshed": True,
                    "materials_count": len(materials),
                }
            else:
                return {"status": "error", "error": "Failed to refresh cache"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_get_materials_categories(self, params):
        """Get list of material categories."""
        if not self.materials_client:
            return {"status": "error", "error": "Materials client not available"}

        try:
            categories = self.materials_client.get_categories()
            return {"status": "success", "categories": categories}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ==================== Object Metadata ====================

    _META_PREFIX = "ConjureMeta_"

    def _cmd_set_metadata(self, params):
        """Set arbitrary key-value metadata on an object.

        Stores each key as a FreeCAD App::PropertyString with a
        'ConjureMeta_' prefix so it persists with the document.

        Parameters:
            object_name: Target object name
            key: Metadata key (alphanumeric + underscores)
            value: Metadata value (stored as string)
        """
        object_name = params.get("object_name")
        key = params.get("key")
        value = params.get("value")

        if not object_name:
            return {"status": "error", "error": "object_name required"}
        if not key:
            return {"status": "error", "error": "key required"}
        if value is None:
            return {"status": "error", "error": "value required"}

        doc = self._get_doc()
        obj = doc.getObject(object_name)
        if not obj:
            return {"status": "error", "error": f"Object not found: {object_name}"}

        prop_name = f"{self._META_PREFIX}{key}"
        str_value = str(value)

        try:
            if not hasattr(obj, prop_name):
                obj.addProperty("App::PropertyString", prop_name, "Conjure Metadata", key)
            setattr(obj, prop_name, str_value)
            return {"status": "success", "object": object_name, "key": key, "value": str_value}
        except Exception as e:
            return {"status": "error", "error": f"Failed to set metadata: {e}"}

    def _cmd_get_metadata(self, params):
        """Get metadata from an object.

        Parameters:
            object_name: Target object name
            key: Optional specific key to retrieve. If omitted, returns all metadata.
        """
        object_name = params.get("object_name")
        if not object_name:
            return {"status": "error", "error": "object_name required"}

        doc = self._get_doc()
        obj = doc.getObject(object_name)
        if not obj:
            return {"status": "error", "error": f"Object not found: {object_name}"}

        key = params.get("key")
        prefix = self._META_PREFIX

        if key:
            prop_name = f"{prefix}{key}"
            if hasattr(obj, prop_name):
                return {"status": "success", "object": object_name, "key": key, "value": getattr(obj, prop_name)}
            return {"status": "error", "error": f"No metadata key '{key}' on {object_name}"}

        # Return all metadata
        metadata = {}
        for prop in obj.PropertiesList:
            if prop.startswith(prefix):
                metadata[prop[len(prefix) :]] = getattr(obj, prop, "")
        return {"status": "success", "object": object_name, "metadata": metadata}

    def _cmd_delete_metadata(self, params):
        """Delete a metadata key from an object.

        Parameters:
            object_name: Target object name
            key: Metadata key to remove
        """
        object_name = params.get("object_name")
        key = params.get("key")

        if not object_name:
            return {"status": "error", "error": "object_name required"}
        if not key:
            return {"status": "error", "error": "key required"}

        doc = self._get_doc()
        obj = doc.getObject(object_name)
        if not obj:
            return {"status": "error", "error": f"Object not found: {object_name}"}

        prop_name = f"{self._META_PREFIX}{key}"
        if not hasattr(obj, prop_name):
            return {"status": "error", "error": f"No metadata key '{key}' on {object_name}"}

        try:
            obj.removeProperty(prop_name)
            return {"status": "success", "object": object_name, "key": key, "deleted": True}
        except Exception as e:
            return {"status": "error", "error": f"Failed to delete metadata: {e}"}

    # ==================== Standards Library ====================

    def _cmd_list_standards(self, params):
        """
        List available standards in the library.

        Args:
            category: Optional filter - "socket", "fastener", "material",
                     "thread", "profile", or "gear"

        Returns:
            Dictionary of category -> list of standard IDs
        """
        if not HAS_STANDARDS:
            return {"status": "error", "error": "Standards library not available"}

        lib = get_standards_library()
        if not lib:
            return {"status": "error", "error": "Standards library not available"}

        try:
            category = params.get("category")
            result = lib.list_standards(category)
            return {"status": "success", "standards": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_get_standard(self, params):
        """
        Get a specific standard specification.

        Args:
            category: Standard category ("socket", "fastener", "material",
                     "thread", "profile", "gear")
            spec_id: Standard ID (e.g., "hex_8mm", "spur_m1_z20")

        Returns:
            Specification dictionary with all dimensions
        """
        if not HAS_STANDARDS:
            return {"status": "error", "error": "Standards library not available"}

        lib = get_standards_library()
        if not lib:
            return {"status": "error", "error": "Standards library not available"}

        category = params.get("category")
        spec_id = params.get("spec_id")

        if not category:
            return {"status": "error", "error": "category is required"}
        if not spec_id:
            return {"status": "error", "error": "spec_id is required"}

        try:
            spec = lib.get_standard(category, spec_id)
            if not spec:
                return {"status": "error", "error": f"Standard '{spec_id}' not found in {category}"}
            return {"status": "success", "spec": spec}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_search_standards(self, params):
        """
        Search for standards matching a query.

        Args:
            query: Search term (matches ID, description, or use cases)
            category: Optional category filter

        Returns:
            List of matching specifications
        """
        if not HAS_STANDARDS:
            return {"status": "error", "error": "Standards library not available"}

        lib = get_standards_library()
        if not lib:
            return {"status": "error", "error": "Standards library not available"}

        query = params.get("query")
        if not query:
            return {"status": "error", "error": "query is required"}

        category = params.get("category")

        try:
            results = lib.search_standards(query, category)
            return {"status": "success", "results": results, "count": len(results)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_get_gear(self, params):
        """
        Get gear specification with calculated involute geometry.

        Args:
            gear_id: Gear ID (e.g., "spur_m1_z20", "planetary_sun_m1_z12")

        Returns:
            Complete gear specification including:
            - module_mm, num_teeth, pressure_angle_deg
            - pitch_diameter_mm, base_diameter_mm
            - addendum_mm, dedendum_mm
            - tip_diameter_mm, root_diameter_mm
            - tooth_thickness_mm, root_fillet_radius_mm
        """
        if not HAS_STANDARDS:
            return {"status": "error", "error": "Standards library not available"}

        lib = get_standards_library()
        if not lib:
            return {"status": "error", "error": "Standards library not available"}

        gear_id = params.get("gear_id")
        if not gear_id:
            return {"status": "error", "error": "gear_id is required"}

        try:
            gear = lib.get_gear(gear_id)
            if not gear:
                available = lib.list_gears()[:10]
                return {
                    "status": "error",
                    "error": f"Gear '{gear_id}' not found",
                    "available_samples": available,
                }
            return {"status": "success", "gear": gear}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_list_gears(self, params):
        """
        List all available gear specifications.

        Returns:
            List of gear IDs
        """
        if not HAS_STANDARDS:
            return {"status": "error", "error": "Standards library not available"}

        lib = get_standards_library()
        if not lib:
            return {"status": "error", "error": "Standards library not available"}

        try:
            gears = lib.list_gears()
            return {"status": "success", "gears": gears, "count": len(gears)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cmd_get_gear_formulas(self, params):
        """
        Get gear geometry formulas for reference.

        Returns:
            Dictionary of involute gear formulas and design guidelines
        """
        if not HAS_STANDARDS:
            return {"status": "error", "error": "Standards library not available"}

        lib = get_standards_library()
        if not lib:
            return {"status": "error", "error": "Standards library not available"}

        try:
            formulas = lib.get_formulas("gears")
            return {"status": "success", "formulas": formulas}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ==================== Simulation ====================

    def _cmd_calculate_dynamic_properties(self, params):
        """
        Calculate dynamic properties (mass, volume, CoM, inertia) for an object.

        Uses FreeCAD's Part module for precise B-rep calculations.
        """
        obj_name = params.get("object")
        material_id = params.get("material_id")
        density = params.get("density")

        if not obj_name:
            return {"status": "error", "error": "object is required"}

        doc = self._get_doc()
        obj = doc.getObject(obj_name)
        if not obj:
            return {"status": "error", "error": f"Object '{obj_name}' not found"}

        if not hasattr(obj, "Shape"):
            return {"status": "error", "error": f"Object '{obj_name}' has no Shape"}

        shape = obj.Shape

        # Get density from material library or parameter
        if density is None:
            if material_id and self.materials_client:
                material = self.materials_client.get_material(material_id)
                if material and material.density_kg_m3:
                    density = material.density_kg_m3
                else:
                    density = 7850  # Default steel
            else:
                density = 7850  # Default steel

        try:
            # FreeCAD volumes are in mm³, convert to m³
            volume_mm3 = shape.Volume
            volume_m3 = volume_mm3 * 1e-9  # mm³ to m³

            # Mass in kg
            mass_kg = volume_m3 * density

            # Surface area (mm² to m²)
            surface_area_m2 = shape.Area * 1e-6

            # Center of mass (FreeCAD uses mm, convert to m)
            com = shape.CenterOfMass
            center_of_mass_m = [com.x * 0.001, com.y * 0.001, com.z * 0.001]

            # Moments of inertia - FreeCAD provides MatrixOfInertia
            # This is the inertia tensor at the center of mass
            # Units: g·mm² - need to convert to kg·m²
            inertia_matrix = shape.MatrixOfInertia

            # Extract diagonal and off-diagonal elements
            # MatrixOfInertia is a 4x4 matrix, upper 3x3 is the inertia tensor
            # Convert from g·mm² to kg·m² (multiply by 1e-9)
            conversion = density / 1000 * 1e-9  # g/mm³ to kg/m³ * mm² to m²

            moments_of_inertia = {
                "Ixx": inertia_matrix.A11 * conversion,
                "Iyy": inertia_matrix.A22 * conversion,
                "Izz": inertia_matrix.A33 * conversion,
                "Ixy": inertia_matrix.A12 * conversion,
                "Ixz": inertia_matrix.A13 * conversion,
                "Iyz": inertia_matrix.A23 * conversion,
            }

            # Bounding box (mm to m)
            bbox = shape.BoundBox
            bounding_box = {
                "min": [bbox.XMin * 0.001, bbox.YMin * 0.001, bbox.ZMin * 0.001],
                "max": [bbox.XMax * 0.001, bbox.YMax * 0.001, bbox.ZMax * 0.001],
                "size": [
                    (bbox.XMax - bbox.XMin) * 0.001,
                    (bbox.YMax - bbox.YMin) * 0.001,
                    (bbox.ZMax - bbox.ZMin) * 0.001,
                ],
                "center": [
                    (bbox.XMin + bbox.XMax) * 0.0005,
                    (bbox.YMin + bbox.YMax) * 0.0005,
                    (bbox.ZMin + bbox.ZMax) * 0.0005,
                ],
            }

            return {
                "status": "success",
                "object": obj_name,
                "dynamic_properties": {
                    "mass_kg": mass_kg,
                    "volume_m3": volume_m3,
                    "surface_area_m2": surface_area_m2,
                    "center_of_mass_m": center_of_mass_m,
                    "moments_of_inertia_kg_m2": moments_of_inertia,
                    "bounding_box_m": bounding_box,
                    "density_kg_m3": density,
                },
            }

        except Exception as e:
            return {"status": "error", "error": f"Calculation failed: {str(e)}"}

    def _cmd_get_simulation_capabilities(self, params):
        """Get the simulation capabilities of this FreeCAD client."""
        return {
            "status": "success",
            "capabilities": {
                "dynamic_properties": {
                    "supported": True,
                    "mass_calculation": True,
                    "volume_calculation": True,
                    "center_of_mass": True,
                    "moments_of_inertia": True,
                    "surface_area": True,
                    "precision": "high",  # B-rep based
                },
                "stress_analysis": {
                    "supported": False,
                    "note": "Use FEM workbench or server-side BeamEstimator",
                },
                "thermal_analysis": {
                    "supported": False,
                    "note": "Use FEM workbench or server-side ThermalEstimator",
                },
                "physics_simulation": {
                    "supported": False,
                    "note": "FreeCAD does not have real-time physics",
                },
            },
            "geometry_kernel": "OpenCASCADE",
            "precision": "double",
        }

    def _cmd_export_geometry_ugf(self, params):
        """
        Export object geometry in Universal Geometry Format (UGF).

        UGF is a mesh-based format for geometry interchange between Conjure clients.
        """
        obj_name = params.get("object")
        include_materials = params.get("include_materials", False)
        mesh_quality = params.get("mesh_quality", 0.1)  # Linear deflection

        if not obj_name:
            return {"status": "error", "error": "object is required"}

        doc = self._get_doc()
        obj = doc.getObject(obj_name)
        if not obj:
            return {"status": "error", "error": f"Object '{obj_name}' not found"}

        if not hasattr(obj, "Shape"):
            return {"status": "error", "error": f"Object '{obj_name}' has no Shape"}

        try:
            import Part

            shape = obj.Shape

            # Tessellate the shape
            mesh_data = shape.tessellate(mesh_quality)
            vertices = mesh_data[0]  # List of Vector
            faces = mesh_data[1]  # List of tuples (v1, v2, v3)

            # Convert to lists (mm to m)
            vertices_list = [[v.x * 0.001, v.y * 0.001, v.z * 0.001] for v in vertices]
            faces_list = [list(f) for f in faces]

            # Compute per-vertex normals by averaging adjacent face normals
            normals_list = [[0.0, 0.0, 0.0] for _ in vertices]
            for face in faces:
                # Compute face normal from triangle vertices
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                edge1 = FreeCAD.Vector(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z)
                edge2 = FreeCAD.Vector(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z)
                fn = edge1.cross(edge2)
                for vi in face:
                    normals_list[vi][0] += fn.x
                    normals_list[vi][1] += fn.y
                    normals_list[vi][2] += fn.z
            # Normalize
            for n in normals_list:
                length = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
                if length > 1e-12:
                    n[0] /= length
                    n[1] /= length
                    n[2] /= length
                else:
                    n[2] = 1.0  # Fallback to Z-up for degenerate vertices

            # Get transform
            placement = obj.Placement
            location = [
                placement.Base.x * 0.001,
                placement.Base.y * 0.001,
                placement.Base.z * 0.001,
            ]
            # Rotation as Euler angles (degrees)
            yaw, pitch, roll = placement.Rotation.toEuler()

            ugf = {
                "format": "UGF",
                "version": "1.0",
                "source": "FreeCAD",
                "object_name": obj_name,
                "vertices": vertices_list,
                "faces": faces_list,
                "normals": normals_list,
                "transform": {
                    "location": location,
                    "rotation": [roll, pitch, yaw],  # XYZ Euler
                    "scale": [1.0, 1.0, 1.0],
                },
            }

            if include_materials:
                # Check for engineering material assignment
                mat_id = obj.ViewObject.ShapeColor if HAS_GUI else None
                if hasattr(obj, "conjure_material_id"):
                    mat_id = obj.conjure_material_id
                    ugf["material_id"] = mat_id

            return {
                "status": "success",
                "object": obj_name,
                "ugf": ugf,
                "vertex_count": len(vertices_list),
                "face_count": len(faces_list),
            }

        except Exception as e:
            return {"status": "error", "error": f"Export failed: {str(e)}"}

    def _cmd_import_geometry_ugf(self, params):
        """
        Import geometry from Universal Geometry Format (UGF).

        Creates a mesh object from the UGF data.
        """
        ugf_data = params.get("ugf")
        object_name = params.get("name")

        if not ugf_data:
            return {"status": "error", "error": "ugf data is required"}

        if ugf_data.get("format") != "UGF":
            return {"status": "error", "error": "Invalid UGF format"}

        try:
            import Mesh
            import Part

            vertices = ugf_data.get("vertices", [])
            faces = ugf_data.get("faces", [])
            name = object_name or ugf_data.get("object_name", "ImportedMesh")

            if not vertices or not faces:
                return {"status": "error", "error": "UGF has no geometry data"}

            doc = self._get_doc()

            # Convert vertices from m to mm
            vertices_mm = [(v[0] * 1000, v[1] * 1000, v[2] * 1000) for v in vertices]

            # Create mesh
            mesh_obj = doc.addObject("Mesh::Feature", name)
            mesh = Mesh.Mesh()

            for face in faces:
                if len(face) >= 3:
                    v1 = vertices_mm[face[0]]
                    v2 = vertices_mm[face[1]]
                    v3 = vertices_mm[face[2]]
                    mesh.addFacet(v1, v2, v3)

            mesh_obj.Mesh = mesh

            # Apply transform if provided
            if "transform" in ugf_data:
                t = ugf_data["transform"]
                if "location" in t:
                    loc = t["location"]
                    mesh_obj.Placement.Base = App.Vector(loc[0] * 1000, loc[1] * 1000, loc[2] * 1000)
                if "rotation" in t:
                    rot = t["rotation"]
                    mesh_obj.Placement.Rotation = App.Rotation(rot[2], rot[1], rot[0])

            self._safe_recompute(doc)

            return {
                "status": "success",
                "object": mesh_obj.Name,
                "vertex_count": len(vertices),
                "face_count": len(faces),
            }

        except Exception as e:
            return {"status": "error", "error": f"Import failed: {str(e)}"}

    def _cmd_get_adapter_capabilities(self, params):
        """Get full adapter capabilities for server registration."""
        import platform

        return {
            "status": "success",
            "adapter_type": "freecad",
            "adapter_id": f"freecad_{id(self)}",
            "version": "1.0.0",
            "freecad_version": App.Version()[0] + "." + App.Version()[1],
            "capabilities": {
                "cad_operations": [
                    "primitives",
                    "booleans",
                    "transforms",
                    "modifiers",
                    "queries",
                    "export",
                    "topology",
                    "faces",
                    "measurements",
                ],
                "simulation": {
                    "dynamic_properties": {
                        "supported": True,
                        "mass_calculation": True,
                        "volume_calculation": True,
                        "center_of_mass": True,
                        "moments_of_inertia": True,
                        "surface_area": True,
                    },
                    "stress_analysis": {
                        "supported": False,
                        "note": "Use server-side BeamEstimator",
                    },
                    "thermal_analysis": {
                        "supported": False,
                        "note": "Use server-side ThermalEstimator",
                    },
                    "physics": {
                        "supported": False,
                        "note": "Use Blender client for physics simulations",
                    },
                },
                "geometry_kernel": "OpenCASCADE",
                "export_formats": ["STEP", "STL", "IGES", "BREP"],
                "import_formats": ["STEP", "STL", "IGES", "BREP", "OBJ"],
                "streaming_results": True,
            },
            "resource_limits": {
                "max_vertices": 50_000_000,
                "max_objects": 10_000,
            },
            "system_info": {
                "os": platform.system(),
                "python_version": platform.python_version(),
            },
        }

    # ==================== Helpers ====================

    def _get_doc(self):
        """Get or create active document."""
        doc = App.ActiveDocument
        if not doc:
            doc = App.newDocument("Untitled")
        return doc


# Global server instance
_server = None


def start_server():
    """Start the Conjure server."""
    global _server
    if _server is None:
        # Get server URL from environment, converting WebSocket URL to HTTP if needed
        import os

        server_url = os.environ.get("CONJURE_SERVER_URL", "https://conjure.lautrek.com")
        # Convert ws:// or wss:// to http:// or https://
        if server_url.startswith("ws://"):
            server_url = server_url.replace("ws://", "http://")
        elif server_url.startswith("wss://"):
            server_url = server_url.replace("wss://", "https://")
        # Remove WebSocket path if present
        if "/api/v1/adapter/ws" in server_url:
            server_url = server_url.split("/api/v1/adapter/ws")[0]
        _server = ConjureServer(server_url=server_url)
    _server.start()
    return _server


def stop_server():
    """Stop the Conjure server."""
    global _server
    if _server:
        _server.stop()
        _server = None


def get_server():
    """Get the server instance."""
    return _server


# ============================================================================
# CLOUD BRIDGE - WebSocket connection to hosted Conjure server
# ============================================================================

import base64
import hashlib
import ssl
import struct
from urllib.parse import urlparse


class SimpleWebSocket:
    """
    Minimal WebSocket client using only Python stdlib.
    Supports wss:// (TLS) connections for secure communication.
    """

    GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(self, url, headers=None):
        self.url = url
        self.headers = headers or {}
        self.sock = None
        self.connected = False

    def connect(self):
        """Establish WebSocket connection."""
        parsed = urlparse(self.url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        # Create socket
        raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_sock.settimeout(10)

        if parsed.scheme == "wss":
            context = ssl.create_default_context()
            self.sock = context.wrap_socket(raw_sock, server_hostname=host)
        else:
            self.sock = raw_sock

        self.sock.connect((host, port))

        # WebSocket handshake
        key = base64.b64encode(os.urandom(16)).decode()
        handshake = f"GET {path} HTTP/1.1\r\n"
        handshake += f"Host: {host}\r\n"
        handshake += "Upgrade: websocket\r\n"
        handshake += "Connection: Upgrade\r\n"
        handshake += f"Sec-WebSocket-Key: {key}\r\n"
        handshake += "Sec-WebSocket-Version: 13\r\n"
        for k, v in self.headers.items():
            handshake += f"{k}: {v}\r\n"
        handshake += "\r\n"

        self.sock.sendall(handshake.encode())

        # Read response
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self.sock.recv(1024)
            if not chunk:
                raise ConnectionError("Connection closed during handshake")
            response += chunk

        # Verify upgrade
        if b"101" not in response.split(b"\r\n")[0]:
            raise ConnectionError(f"WebSocket upgrade failed: {response[:100]}")

        # Verify accept key
        expected = base64.b64encode(hashlib.sha1((key + self.GUID).encode()).digest()).decode()
        if expected.encode() not in response:
            raise ConnectionError("Invalid Sec-WebSocket-Accept")

        self.connected = True

    def send(self, data):
        """Send text frame."""
        if isinstance(data, str):
            data = data.encode()

        # Build frame: FIN + text opcode
        frame = bytearray([0x81])  # FIN=1, opcode=1 (text)

        # Length + mask bit
        length = len(data)
        if length < 126:
            frame.append(0x80 | length)  # Mask bit set
        elif length < 65536:
            frame.append(0x80 | 126)
            frame.extend(struct.pack(">H", length))
        else:
            frame.append(0x80 | 127)
            frame.extend(struct.pack(">Q", length))

        # Mask key and masked data
        mask = os.urandom(4)
        frame.extend(mask)
        for i, byte in enumerate(data):
            frame.append(byte ^ mask[i % 4])

        self.sock.sendall(frame)

    def recv(self):
        """Receive text frame. Returns string or None on close."""
        # Read frame header
        header = self._recv_exact(2)
        if not header:
            return None

        # fin = header[0] & 0x80  # FIN bit (unused, we don't handle fragmentation)
        opcode = header[0] & 0x0F
        masked = header[1] & 0x80
        length = header[1] & 0x7F

        # Extended length
        if length == 126:
            length = struct.unpack(">H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self._recv_exact(8))[0]

        # Mask key (if server sends masked, which it shouldn't)
        mask = self._recv_exact(4) if masked else None

        # Payload
        data = self._recv_exact(length)
        if mask:
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))

        # Handle opcodes
        if opcode == 0x08:  # Close
            return None
        elif opcode == 0x09:  # Ping
            self._send_pong(data)
            return self.recv()
        elif opcode == 0x0A:  # Pong
            return self.recv()
        elif opcode in (0x01, 0x02):  # Text or binary
            return data.decode() if opcode == 0x01 else data

        return None

    def _recv_exact(self, n):
        """Receive exactly n bytes."""
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        return data

    def _send_pong(self, data):
        """Send pong frame."""
        frame = bytearray([0x8A, 0x80 | len(data)])
        mask = os.urandom(4)
        frame.extend(mask)
        for i, byte in enumerate(data):
            frame.append(byte ^ mask[i % 4])
        self.sock.sendall(frame)

    def close(self):
        """Close connection."""
        if self.sock:
            try:
                # Send close frame
                self.sock.sendall(bytes([0x88, 0x80, 0, 0, 0, 0]))
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None
        self.connected = False

    def settimeout(self, timeout):
        """Set socket timeout."""
        if self.sock:
            self.sock.settimeout(timeout)


class CloudBridge:
    """
    WebSocket bridge to Conjure Cloud server.

    Connects to the cloud server, receives commands, and executes them locally.
    This enables AI assistants (Claude, Cursor, etc.) to control FreeCAD.
    Uses only Python stdlib - no external dependencies required.
    """

    def __init__(self, server: ConjureServer, api_key: str, server_url: str = None):
        self.server = server
        self.api_key = api_key
        self.server_url = server_url or "wss://conjure.lautrek.com/api/v1/adapter/ws"
        self.ws = None
        self.running = False
        self.thread = None
        self.adapter_id = None
        self._heartbeat_thread = None
        self._ws_lock = threading.Lock()  # Protect WebSocket send

    def start(self):
        """Start cloud bridge in background thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        App.Console.PrintMessage(f"Cloud bridge connecting to {self.server_url}\n")

    def stop(self):
        """Stop cloud bridge."""
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.ws = None
        self.adapter_id = None
        App.Console.PrintMessage("Cloud bridge stopped\n")

    def _run_loop(self):
        """Main connection loop with reconnection."""
        import time

        while self.running:
            try:
                self._connect_and_run()
            except Exception as e:
                App.Console.PrintWarning(f"Cloud bridge error: {e}\n")
                self.adapter_id = None

            if self.running:
                # Wait before reconnecting
                time.sleep(5)

    def _connect_and_run(self):
        """Connect and process messages."""
        # Create WebSocket connection with auth header
        self.ws = SimpleWebSocket(
            self.server_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        self.ws.connect()

        App.Console.PrintMessage("Cloud bridge: WebSocket connected\n")

        # Send registration with full capabilities
        registration = {
            "type": "adapter_registration",
            "adapter_type": "freecad",
            "adapter_id": f"freecad_{id(self.server)}",
            "version": "1.0.0",
            "freecad_version": App.Version()[0] + "." + App.Version()[1],
            "capabilities": {
                "cad_operations": [
                    "primitives",
                    "booleans",
                    "transforms",
                    "modifiers",
                    "queries",
                    "export",
                    "topology",
                    "faces",
                    "measurements",
                ],
                "simulation": {
                    "dynamic_properties": {
                        "supported": True,
                        "mass_calculation": True,
                        "volume_calculation": True,
                        "center_of_mass": True,
                        "moments_of_inertia": True,
                        "surface_area": True,
                    },
                    "stress_analysis": {"supported": False},
                    "thermal_analysis": {"supported": False},
                    "physics": {"supported": False},
                },
                "geometry_kernel": "OpenCASCADE",
                "export_formats": ["STEP", "STL", "IGES", "BREP"],
                "import_formats": ["STEP", "STL", "IGES", "BREP", "OBJ"],
            },
        }
        self.ws.send(json.dumps(registration))

        # Wait for registration confirmation
        resp = json.loads(self.ws.recv())
        if resp.get("type") == "registration_confirmed":
            self.adapter_id = resp.get("adapter_id")
            App.Console.PrintMessage(f"Cloud bridge: Registered as {self.adapter_id}\n")
        else:
            App.Console.PrintWarning(f"Cloud bridge: Unexpected response: {resp}\n")
            return

        # Start proactive heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        # Message loop
        while self.running:
            try:
                self.ws.settimeout(60)  # Longer timeout, heartbeats are proactive now
                raw = self.ws.recv()
                if raw is None:
                    # Connection closed
                    App.Console.PrintWarning("Cloud bridge: Connection closed\n")
                    break
                message = json.loads(raw)
                self._handle_message(message)
            except socket.timeout:
                # Timeout just means no messages, heartbeat thread handles keepalive
                pass
            except (ConnectionError, OSError) as e:
                App.Console.PrintWarning(f"Cloud bridge: Connection error: {e}\n")
                break
            except Exception as e:
                App.Console.PrintWarning(f"Cloud bridge message error: {e}\n")
                break

    def _handle_message(self, message):
        """Handle incoming message from cloud."""
        msg_type = message.get("type")

        if msg_type == "execute_command":
            self._handle_execute_command(message)
        elif msg_type == "heartbeat_ack":
            pass  # Heartbeat acknowledged
        else:
            App.Console.PrintWarning(f"Cloud bridge: Unknown message type: {msg_type}\n")

    def _handle_execute_command(self, message):
        """Execute command and send result back."""
        request_id = message.get("request_id")
        command_type = message.get("command_type")
        params = message.get("params", {})

        App.Console.PrintMessage(f"Cloud bridge: Executing {command_type}\n")

        # Execute via thread-safe method (queues to main thread)
        try:
            result_data = self.server.execute_threadsafe(command_type, params, timeout=30.0)

            # Send result back
            response = {
                "type": "command_result",
                "request_id": request_id,
                "success": result_data.get("status") == "success",
                "data": result_data,
                "error": result_data.get("error"),
            }
            with self._ws_lock:
                self.ws.send(json.dumps(response))

        except Exception as e:
            # Send error back
            response = {
                "type": "command_result",
                "request_id": request_id,
                "success": False,
                "data": {},
                "error": str(e),
            }
            with self._ws_lock:
                self.ws.send(json.dumps(response))

    def _heartbeat_loop(self):
        """Proactive heartbeat sender - runs in separate thread."""
        import time

        while self.running and self.ws:
            time.sleep(25)  # Send heartbeat every 25 seconds (server timeout is 60s)
            if not self.running or not self.ws:
                break
            try:
                with self._ws_lock:
                    self.ws.send(json.dumps({"type": "heartbeat"}))
            except Exception:
                # Connection issue, main loop will handle reconnect
                break


# Global cloud bridge instance
_cloud_bridge = None


def start_cloud_bridge(api_key: str, server_url: str = None):
    """Start the cloud bridge."""
    global _cloud_bridge, _server

    if _server is None:
        start_server()

    # Derive HTTP API URL from WebSocket URL and update materials client
    if server_url and _server and MaterialsClient:
        try:
            # Convert wss://host/path to https://host
            api_url = server_url.replace("wss://", "https://").replace("ws://", "http://")
            # Remove the WebSocket path to get base URL
            if "/api/" in api_url:
                api_url = api_url.split("/api/")[0]
            elif "/ws" in api_url:
                api_url = api_url.split("/ws")[0]

            # Update or create materials client with correct URL
            _server.materials_client = MaterialsClient(api_url, api_key)
            App.Console.PrintMessage(f"Conjure: Materials API URL set to {api_url}\n")
        except Exception as e:
            App.Console.PrintWarning(f"Conjure: Failed to configure materials client: {e}\n")

    if _cloud_bridge is None:
        _cloud_bridge = CloudBridge(_server, api_key, server_url)

    _cloud_bridge.start()

    # Hook up placement observer for constraint propagation
    observer = get_placement_observer()
    observer.set_cloud_bridge(_cloud_bridge)

    return _cloud_bridge


def stop_cloud_bridge():
    """Stop the cloud bridge."""
    global _cloud_bridge
    if _cloud_bridge:
        _cloud_bridge.stop()
        _cloud_bridge = None


def get_cloud_bridge():
    """Get the cloud bridge instance."""
    return _cloud_bridge
