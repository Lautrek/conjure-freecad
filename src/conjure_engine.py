"""
Conjure Engine - Handles low-level communication with FreeCAD server.

This module provides the core socket-based communication with FreeCAD,
abstracting the transport layer from the MCP protocol handling.

Features:
- Socket-based communication with FreeCAD instance
- Request/response serialization
- Error handling and timeouts
- Request ID tracking for correlation
"""

import json
import socket
import uuid
from typing import Any, Dict, Optional


class ConjureConnectionError(Exception):
    """Raised when FreeCAD communication fails."""

    pass


class ConjureEngine:
    """Manages communication with FreeCAD server instance."""

    def __init__(self, host: str = "localhost", port: int = 9876):
        """Initialize FreeCAD engine.

        Args:
            host: FreeCAD server host (default: localhost)
            port: FreeCAD server port (default: 9876)
        """
        self.host = host
        self.port = port
        self.timeout = 30  # 30 second timeout for operations
        self._request_id = None

    def get_request_id(self) -> str:
        """Get or generate request ID for correlation."""
        if not self._request_id:
            self._request_id = str(uuid.uuid4())[:8]
        return self._request_id

    def set_request_id(self, request_id: str) -> None:
        """Set request ID for correlation."""
        self._request_id = request_id

    def _connect(self) -> socket.socket:
        """Create and configure socket connection.

        Returns:
            Connected socket object

        Raises:
            ConjureConnectionError: If connection fails
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            return sock
        except ConnectionRefusedError as e:
            raise ConjureConnectionError(
                f"Cannot connect to FreeCAD server at {self.host}:{self.port} (connection refused). Is FreeCAD running?"
            ) from e
        except socket.timeout as e:
            raise ConjureConnectionError(f"Connection to FreeCAD server timed out after {self.timeout}s") from e
        except Exception as e:
            raise ConjureConnectionError(f"Failed to connect to FreeCAD server: {str(e)}") from e

    def _send_command(self, sock: socket.socket, command: Dict[str, Any]) -> None:
        """Send command to FreeCAD via socket.

        Args:
            sock: Connected socket
            command: Command dictionary to send

        Raises:
            ConjureConnectionError: If send fails
        """
        try:
            command_json = json.dumps(command) + "\n"  # Newline-terminated for socket server
            sock.sendall(command_json.encode("utf-8"))
        except Exception as e:
            sock.close()
            raise ConjureConnectionError(f"Failed to send command to FreeCAD: {str(e)}") from e

    def _receive_response(self, sock: socket.socket) -> Dict[str, Any]:
        """Receive response from FreeCAD via socket.

        Handles multi-chunk responses and validates JSON.

        Args:
            sock: Connected socket

        Returns:
            Parsed response dictionary

        Raises:
            ConjureConnectionError: If receive fails or response is invalid
        """
        try:
            chunks = []
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break

                chunks.append(chunk)

                # Try to parse - if valid JSON, we're done
                try:
                    data = b"".join(chunks)
                    json.loads(data.decode("utf-8"))
                    break
                except json.JSONDecodeError:
                    # Keep reading more chunks
                    continue

            if not chunks:
                sock.close()
                raise ConjureConnectionError("FreeCAD connection closed without response")

            response_data = b"".join(chunks).decode("utf-8")
            return json.loads(response_data)

        except json.JSONDecodeError as e:
            sock.close()
            raise ConjureConnectionError(f"Invalid JSON response from FreeCAD: {str(e)}") from e
        except socket.timeout as e:
            sock.close()
            raise ConjureConnectionError(f"Timeout waiting for FreeCAD response after {self.timeout}s") from e
        except Exception as e:
            sock.close()
            raise ConjureConnectionError(f"Failed to receive response from FreeCAD: {str(e)}") from e

    def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute command in FreeCAD and return response.

        Args:
            command: Command dictionary with 'type' and 'params'

        Returns:
            Response dictionary from FreeCAD

        Raises:
            ConjureConnectionError: If communication fails
        """
        # Add request ID for correlation if not present
        if "request_id" not in command:
            command["request_id"] = self.get_request_id()

        sock = self._connect()
        try:
            self._send_command(sock, command)
            response = self._receive_response(sock)
            return response
        finally:
            sock.close()

    def send_command(self, command_text: str, get_context: bool = True) -> Dict[str, Any]:
        """Send FreeCAD command and get document context.

        Args:
            command_text: FreeCAD command to execute
            get_context: Whether to include document context in response

        Returns:
            Response with command result and context
        """
        cmd = {"type": "send_command", "params": {"command": command_text, "get_context": get_context}}
        return self.execute(cmd)

    def run_script(self, script: str) -> Dict[str, Any]:
        """Run Python script in FreeCAD context.

        Args:
            script: Python script to execute

        Returns:
            Script execution result
        """
        cmd = {"type": "run_script", "params": {"script": script}}
        return self.execute(cmd)

    def get_state(self, verbose: bool = False) -> Dict[str, Any]:
        """Get current FreeCAD document state.

        Args:
            verbose: If True, include full geometry details

        Returns:
            Document state (lite or verbose mode)
        """
        cmd = {"type": "get_enhanced_state", "params": {"verbose": verbose}}
        return self.execute(cmd)

    def health_check(self) -> bool:
        """Check if FreeCAD server is running and responsive.

        Returns:
            True if server is responding, False otherwise
        """
        try:
            response = self.get_state(verbose=False)
            return response.get("status") != "error"
        except ConjureConnectionError:
            return False


# Global engine instance
_engine: Optional[ConjureEngine] = None


def get_engine() -> ConjureEngine:
    """Get or create global FreeCAD engine instance."""
    global _engine
    if _engine is None:
        _engine = ConjureEngine()
    return _engine


def init_engine(host: str = "localhost", port: int = 9876) -> ConjureEngine:
    """Initialize global FreeCAD engine with custom parameters.

    Args:
        host: FreeCAD server host
        port: FreeCAD server port

    Returns:
        Configured FreeCAD engine
    """
    global _engine
    _engine = ConjureEngine(host, port)
    return _engine
