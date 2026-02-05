"""
Conjure FreeCAD Client Logger

Provides persistent file logging for diagnosing crashes and issues.
Logs are written to ~/.conjure/logs/freecad_client.log

Features:
- Rotating log files (max 5MB, keeps 3 backups)
- Command execution logging with timing
- Exception logging with full stack traces
- Configurable log level
"""

import logging
import sys
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

# Log directory
LOG_DIR = Path.home() / ".conjure" / "logs"
LOG_FILE = LOG_DIR / "freecad_client.log"
CRASH_FILE = LOG_DIR / "crashes.log"

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create logger
_logger = logging.getLogger("conjure.freecad")
_logger.setLevel(logging.DEBUG)

# File handler with rotation
_file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5 * 1024 * 1024,  # 5MB
    backupCount=3,
    encoding="utf-8",
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
_logger.addHandler(_file_handler)

# Crash handler - separate file for crashes
_crash_handler = logging.FileHandler(CRASH_FILE, encoding="utf-8")
_crash_handler.setLevel(logging.ERROR)
_crash_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
_logger.addHandler(_crash_handler)


def get_logger() -> logging.Logger:
    """Get the Conjure logger instance."""
    return _logger


def log_startup(port: int, server_url: str):
    """Log server startup."""
    _logger.info("=" * 60)
    _logger.info("Conjure FreeCAD Client Starting")
    _logger.info(f"  Port: {port}")
    _logger.info(f"  Server URL: {server_url}")
    _logger.info(f"  Python: {sys.version}")
    _logger.info(f"  Log file: {LOG_FILE}")
    _logger.info("=" * 60)


def log_shutdown():
    """Log server shutdown."""
    _logger.info("Conjure FreeCAD Client Shutdown")
    _logger.info("-" * 60)


def log_command(
    command_type: str,
    params: Dict[str, Any],
    duration_ms: float,
    success: bool,
    error: Optional[str] = None,
):
    """Log command execution with timing."""
    status = "OK" if success else "FAILED"
    param_str = _summarize_params(params)

    if success:
        _logger.info(f"CMD {command_type} | {status} | {duration_ms:.1f}ms | {param_str}")
    else:
        _logger.warning(f"CMD {command_type} | {status} | {duration_ms:.1f}ms | {param_str} | Error: {error}")


def log_exception(context: str, exc: Exception):
    """Log an exception with full stack trace."""
    tb = traceback.format_exc()
    _logger.error(f"EXCEPTION in {context}")
    _logger.error(f"  Type: {type(exc).__name__}")
    _logger.error(f"  Message: {str(exc)}")
    _logger.error(f"  Traceback:\n{tb}")


def log_crash(context: str, exc: Exception, doc_state: Optional[Dict] = None):
    """Log a crash with document state for debugging."""
    _logger.critical("=" * 60)
    _logger.critical(f"CRASH DETECTED: {context}")
    _logger.critical(f"  Time: {datetime.now().isoformat()}")
    _logger.critical(f"  Exception: {type(exc).__name__}: {str(exc)}")

    if doc_state:
        _logger.critical(f"  Document: {doc_state.get('document', 'unknown')}")
        _logger.critical(f"  Objects: {doc_state.get('object_count', 'unknown')}")

    _logger.critical(f"  Traceback:\n{traceback.format_exc()}")
    _logger.critical("=" * 60)


def log_connection(event: str, details: str = ""):
    """Log connection events."""
    _logger.info(f"CONNECTION {event}: {details}")


def log_debug(message: str):
    """Log debug message."""
    _logger.debug(message)


def log_info(message: str):
    """Log info message."""
    _logger.info(message)


def log_warning(message: str):
    """Log warning message."""
    _logger.warning(message)


def log_error(message: str):
    """Log error message."""
    _logger.error(message)


def _summarize_params(params: Dict[str, Any], max_len: int = 100) -> str:
    """Summarize params dict for logging."""
    if not params:
        return "{}"

    # Filter out large values
    summary = {}
    for k, v in params.items():
        if isinstance(v, str) and len(v) > 50:
            summary[k] = f"{v[:47]}..."
        elif isinstance(v, (list, dict)) and len(str(v)) > 50:
            summary[k] = f"<{type(v).__name__}:{len(v)} items>"
        else:
            summary[k] = v

    result = str(summary)
    if len(result) > max_len:
        return result[: max_len - 3] + "..."
    return result


def get_recent_logs(lines: int = 100) -> str:
    """Get recent log lines for debugging."""
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception:
        return "Could not read log file"


def log_signal_crash(signal_number: int, signal_name: str):
    """Log a fatal signal crash and flush immediately.

    Called from signal handler — must be minimal and flush before process dies.
    """
    _logger.critical("=" * 60)
    _logger.critical(f"FATAL SIGNAL: {signal_name} ({signal_number})")
    _logger.critical(f"  Time: {datetime.now().isoformat()}")
    _logger.critical("  Process is dying — this is a C-level crash, not a Python exception")
    _logger.critical("=" * 60)
    # Force flush all handlers before process terminates
    for handler in _logger.handlers:
        handler.flush()


def get_log_file_path() -> str:
    """Get the path to the log file."""
    return str(LOG_FILE)


def get_crash_log_path() -> str:
    """Get the path to the crash log file."""
    return str(CRASH_FILE)
