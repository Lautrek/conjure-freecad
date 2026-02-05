"""
Conjure Module Initialization

This file initializes the Conjure - AI CAD Control workbench and
registers all available tools and commands.
"""

import logging
import os
import sys
import tempfile

import FreeCAD as App

# Configure logging with cross-platform support
# On Windows GUI apps, stdout/stderr can be None, so we use a file handler
logger = logging.getLogger("Conjure")
logger.setLevel(logging.INFO)

# Determine log file path (cross-platform)
_log_file = os.path.join(tempfile.gettempdir(), "conjure_freecad.log")

try:
    # Try to add a file handler
    _file_handler = logging.FileHandler(_log_file, mode="a", encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_file_handler)
except Exception:
    # If file handler fails, try stream handler only if stdout exists
    if sys.stdout is not None:
        try:
            _stream_handler = logging.StreamHandler(sys.stdout)
            _stream_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(_stream_handler)
        except Exception:
            pass  # Logging will be silent but won't crash


def initialize_workbench():
    """Initialize the Conjure workbench."""
    try:
        # Add src directory to path for imports
        # Use __file__ to get the module location reliably (works with symlinks and copies)
        conjure_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(conjure_dir, "src")

        if conjure_dir not in sys.path:
            sys.path.insert(0, conjure_dir)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Import main Conjure module
        import conjure

        logger.info("Conjure module initialized successfully")
        logger.info(f"  Module location: {conjure_dir}")
        logger.info(f"  FreeCAD version: {App.Version()}")
        logger.info(f"  Log file: {_log_file}")
        return True

    except ImportError as e:
        logger.warning(f"Could not import Conjure core module: {e}")
        logger.warning("  Extended features will not be available")
        logger.warning("  Check that all Python dependencies are installed")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize Conjure: {e}")
        return False


# Auto-initialize on module load
try:
    initialize_workbench()
except Exception as e:
    logger.warning(f"Conjure initialization incomplete: {e}")

# Force load InitGui when in GUI mode
# FreeCAD should auto-load this, but some versions don't
if getattr(App, "GuiUp", False):
    try:
        pass
    except Exception as e:
        logger.warning(f"Could not load InitGui: {e}")
