"""
Server Client

WebSocket client that connects to the hosted Conjure server.
Receives commands, executes them via the FreeCAD adapter, returns results.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

try:
    import websockets
    from websockets.client import WebSocketClientProtocol

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketClientProtocol = None

from .freecad_adapter import AdapterResult, FreeCADAdapter

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    """Client connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


@dataclass
class ServerConfig:
    """Configuration for server connection."""

    server_url: str = "wss://conjure.lautrek.com/api/v1/adapter/ws"
    api_key: Optional[str] = None
    adapter_id: Optional[str] = None
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    heartbeat_interval: float = 30.0


class ServerClient:
    """
    WebSocket client for connecting to hosted Conjure server.

    Flow:
    1. Connect to server with API key
    2. Register as adapter (send capabilities)
    3. Listen for commands
    4. Execute commands via FreeCADAdapter
    5. Send results back to server
    """

    def __init__(
        self,
        config: Optional[ServerConfig] = None,
        adapter: Optional[FreeCADAdapter] = None,
    ):
        """Initialize server client.

        Args:
            config: Server connection configuration
            adapter: FreeCAD adapter instance
        """
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required. Install with: pip install websockets")

        self.config = config or ServerConfig()
        self.adapter = adapter or FreeCADAdapter()

        self._ws: Optional[WebSocketClientProtocol] = None
        self._state = ConnectionState.DISCONNECTED
        self._reconnect_attempts = 0
        self._running = False
        self._message_handlers: Dict[str, Callable] = {}

        # Register default handlers
        self._register_handlers()

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._state == ConnectionState.CONNECTED and self._ws is not None

    def _register_handlers(self):
        """Register message handlers."""
        self._message_handlers = {
            "execute_command": self._handle_execute_command,
            "health_check": self._handle_health_check,
            "disconnect": self._handle_disconnect,
        }

    async def connect(self) -> bool:
        """Connect to the hosted server.

        Returns:
            True if connected successfully
        """
        if self.is_connected:
            return True

        self._state = ConnectionState.CONNECTING

        try:
            # Build headers with authentication
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            self._ws = await websockets.connect(
                self.config.server_url,
                extra_headers=headers,
                ping_interval=self.config.heartbeat_interval,
            )

            # Send registration message
            await self._register()

            self._state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            logger.info(f"Connected to server: {self.config.server_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            self._state = ConnectionState.DISCONNECTED
            return False

    async def _register(self):
        """Register adapter with server."""
        registration = {
            "type": "adapter_registration",
            "adapter_id": self.config.adapter_id,
            "adapter_type": "freecad",
            "capabilities": [
                "primitives",
                "booleans",
                "transforms",
                "queries",
                "export",
            ],
            "version": "1.0.0",
        }
        await self._send(registration)

    async def disconnect(self):
        """Disconnect from server."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from server")

    async def run(self):
        """Main run loop - connect and process messages."""
        self._running = True

        while self._running:
            # Connect if not connected
            if not self.is_connected:
                connected = await self.connect()
                if not connected:
                    self._reconnect_attempts += 1
                    if self._reconnect_attempts > self.config.max_reconnect_attempts:
                        logger.error("Max reconnection attempts reached")
                        break
                    await asyncio.sleep(self.config.reconnect_delay)
                    continue

            # Process messages
            try:
                async for message in self._ws:
                    await self._handle_message(message)
            except websockets.ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
                self._state = ConnectionState.RECONNECTING
                self._ws = None
            except Exception as e:
                logger.error(f"Error in message loop: {e}")
                self._state = ConnectionState.RECONNECTING
                await asyncio.sleep(1)

    async def _send(self, message: Dict[str, Any]):
        """Send message to server."""
        if not self._ws:
            raise RuntimeError("Not connected to server")

        await self._ws.send(json.dumps(message))

    async def _handle_message(self, raw_message: str):
        """Handle incoming message from server."""
        try:
            message = json.loads(raw_message)
            message_type = message.get("type")

            handler = self._message_handlers.get(message_type)
            if handler:
                response = await handler(message)
                if response:
                    await self._send(response)
            else:
                logger.warning(f"Unknown message type: {message_type}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _handle_execute_command(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle command execution request from server."""
        request_id = message.get("request_id")
        command_type = message.get("command_type")
        params = message.get("params", {})

        logger.debug(f"Executing command: {command_type}")

        # Execute via adapter
        result = await self.adapter.execute(command_type, params)

        # Return result to server
        return {
            "type": "command_result",
            "request_id": request_id,
            "success": result.success,
            "data": result.data,
            "error": result.error,
        }

    async def _handle_health_check(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check request."""
        freecad_healthy = self.adapter.health_check()

        return {
            "type": "health_check_response",
            "request_id": message.get("request_id"),
            "adapter_healthy": True,
            "freecad_healthy": freecad_healthy,
        }

    async def _handle_disconnect(self, message: Dict[str, Any]) -> None:
        """Handle disconnect request from server."""
        logger.info("Server requested disconnect")
        await self.disconnect()
        return None


async def run_adapter(
    server_url: str = "wss://conjure.lautrek.com/api/v1/adapter/ws",
    api_key: Optional[str] = None,
    freecad_host: str = "localhost",
    freecad_port: int = 9876,
):
    """Run the adapter as a standalone service.

    Args:
        server_url: Hosted server WebSocket URL
        api_key: API key for authentication
        freecad_host: FreeCAD server host
        freecad_port: FreeCAD server port
    """
    # Create adapter and client
    adapter = FreeCADAdapter(host=freecad_host, port=freecad_port)
    config = ServerConfig(
        server_url=server_url,
        api_key=api_key,
    )
    client = ServerClient(config=config, adapter=adapter)

    # Run
    try:
        await client.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await client.disconnect()


def main():
    """CLI entry point for running adapter."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Conjure FreeCAD adapter")
    parser.add_argument(
        "--server-url",
        default="wss://conjure.lautrek.com/api/v1/adapter/ws",
        help="Hosted server WebSocket URL",
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication",
    )
    parser.add_argument(
        "--freecad-host",
        default="localhost",
        help="FreeCAD server host",
    )
    parser.add_argument(
        "--freecad-port",
        type=int,
        default=9876,
        help="FreeCAD server port",
    )

    args = parser.parse_args()

    asyncio.run(
        run_adapter(
            server_url=args.server_url,
            api_key=args.api_key,
            freecad_host=args.freecad_host,
            freecad_port=args.freecad_port,
        )
    )


if __name__ == "__main__":
    main()
