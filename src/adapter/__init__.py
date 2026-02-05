"""
Thin FreeCAD Adapter

This is a lightweight adapter that:
1. Connects to the hosted Conjure server via WebSocket
2. Receives CAD commands from the server
3. Executes them in FreeCAD via socket
4. Returns results back to the server

All business logic lives server-side. This adapter just executes commands.
"""

from .freecad_adapter import FreeCADAdapter

__all__ = [
    "FreeCADAdapter",
]
