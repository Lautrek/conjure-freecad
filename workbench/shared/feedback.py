"""
Feedback client for Conjure orchestration layer.

Collects anonymized feedback from local CAD operations and reports to the
server for iterative improvement. All data is privacy-compliant:
- Session IDs are hashed locally before transmission
- No geometry data or file names are sent
- Only command types and outcomes

This enables the server to learn patterns like:
- Common failure modes for specific commands
- Effective adjustment strategies
- Success rates by operation type
"""

import hashlib
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class CommandFeedback:
    """Anonymized feedback for a single command execution."""

    command_type: str
    success: bool
    execution_time_ms: float
    retry_count: int = 0
    verification_passed: bool = True
    failure_category: Optional[str] = None  # e.g., "geometry_invalid", "constraint_conflict"
    adjustment_applied: Optional[str] = None  # e.g., "reduce_radius", "simplify_edges"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "command_type": self.command_type,
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "retry_count": self.retry_count,
            "verification_passed": self.verification_passed,
            "failure_category": self.failure_category,
            "adjustment_applied": self.adjustment_applied,
            "timestamp": self.timestamp,
        }


class FeedbackClient:
    """
    Client for reporting anonymized feedback to the Conjure server.

    Features:
    - Asynchronous batched reporting (doesn't block command execution)
    - Local buffering with automatic retry
    - Privacy-preserving: session IDs hashed, no PII
    - Graceful degradation if server unavailable

    Usage:
        client = FeedbackClient(server_url="http://localhost:8000")
        client.start()

        # After each command execution:
        client.record(
            command_type="create_fillet",
            success=True,
            execution_time_ms=150.5,
            failure_category=None
        )

        # On shutdown:
        client.stop()
    """

    # Buffer limits
    MAX_BUFFER_SIZE = 1000
    BATCH_SIZE = 50
    FLUSH_INTERVAL_SECONDS = 30

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        session_id: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize feedback client.

        Args:
            server_url: Base URL of the Conjure server
            api_key: API key for authentication (optional)
            session_id: Session identifier (will be hashed before sending)
            enabled: Whether feedback collection is enabled
        """
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.enabled = enabled

        # Generate or hash session ID
        if session_id:
            self._session_hash = hashlib.sha256(session_id.encode()).hexdigest()[:16]
        else:
            # Generate random session hash
            import uuid

            self._session_hash = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:16]

        # Buffer for pending feedback
        self._buffer: queue.Queue = queue.Queue(maxsize=self.MAX_BUFFER_SIZE)
        self._failed_buffer: list[CommandFeedback] = []  # Retry buffer

        # Background thread for sending
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        # Statistics
        self._stats = {
            "sent": 0,
            "failed": 0,
            "dropped": 0,
        }

        # HTTP session (lazy initialized)
        self._http_session = None

    def start(self) -> None:
        """Start the background feedback sender thread."""
        if not self.enabled:
            logger.info("Feedback client disabled")
            return

        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._thread.start()
        logger.info(f"Feedback client started (server: {self.server_url})")

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the feedback client and flush remaining data.

        Args:
            timeout: Maximum time to wait for flush
        """
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

        # Final flush attempt
        self._flush_sync()
        logger.info(f"Feedback client stopped (stats: {self._stats})")

    def record(
        self,
        command_type: str,
        success: bool,
        execution_time_ms: float,
        retry_count: int = 0,
        verification_passed: bool = True,
        failure_category: Optional[str] = None,
        adjustment_applied: Optional[str] = None,
    ) -> bool:
        """
        Record feedback for a command execution.

        Args:
            command_type: Type of command (e.g., "create_box", "fillet_face")
            success: Whether the command succeeded
            execution_time_ms: Execution time in milliseconds
            retry_count: Number of retries before success/failure
            verification_passed: Whether verification passed (if applicable)
            failure_category: Category of failure (if failed)
            adjustment_applied: Adjustment that was applied (if any)

        Returns:
            True if feedback was buffered, False if dropped
        """
        if not self.enabled:
            return False

        feedback = CommandFeedback(
            command_type=command_type,
            success=success,
            execution_time_ms=execution_time_ms,
            retry_count=retry_count,
            verification_passed=verification_passed,
            failure_category=failure_category,
            adjustment_applied=adjustment_applied,
        )

        try:
            self._buffer.put_nowait(feedback)
            return True
        except queue.Full:
            # Buffer full - drop oldest
            self._stats["dropped"] += 1
            logger.warning("Feedback buffer full, dropping entry")
            return False

    def categorize_failure(self, error_message: str) -> str:
        """
        Categorize a failure based on error message.

        Args:
            error_message: The error message from the command

        Returns:
            Failure category string
        """
        msg = error_message.lower()

        if "null" in msg or "invalid" in msg or "none" in msg:
            return "geometry_invalid"
        if "constraint" in msg or "conflict" in msg:
            return "constraint_conflict"
        if "interference" in msg or "collision" in msg or "overlap" in msg:
            return "interference"
        if "size" in msg or "dimension" in msg or "too large" in msg or "too small" in msg:
            return "dimension_error"
        if "face" in msg or "edge" in msg or "vertex" in msg:
            return "topology_error"
        if "timeout" in msg or "timed out" in msg:
            return "timeout"
        if "not found" in msg:
            return "not_found"
        if "permission" in msg or "access" in msg:
            return "permission_error"

        return "unknown"

    def get_stats(self) -> dict:
        """Get feedback client statistics."""
        return {
            **self._stats,
            "buffered": self._buffer.qsize(),
            "failed_pending": len(self._failed_buffer),
            "enabled": self.enabled,
            "running": self._running,
        }

    def _sender_loop(self) -> None:
        """Background loop for sending feedback batches."""
        last_flush = time.time()

        while self._running or not self._buffer.empty():
            # Wait for stop event or flush interval
            self._stop_event.wait(timeout=1.0)

            # Check if we should flush
            buffer_size = self._buffer.qsize()
            time_since_flush = time.time() - last_flush

            should_flush = (
                buffer_size >= self.BATCH_SIZE
                or (buffer_size > 0 and time_since_flush >= self.FLUSH_INTERVAL_SECONDS)
                or (not self._running and buffer_size > 0)
            )

            if should_flush:
                self._flush()
                last_flush = time.time()

    def _flush(self) -> None:
        """Flush buffered feedback to server."""
        batch = []

        # Collect batch from buffer
        while len(batch) < self.BATCH_SIZE:
            try:
                feedback = self._buffer.get_nowait()
                batch.append(feedback)
            except queue.Empty:
                break

        # Add failed items from retry buffer
        if self._failed_buffer and len(batch) < self.BATCH_SIZE:
            retry_count = min(len(self._failed_buffer), self.BATCH_SIZE - len(batch))
            batch.extend(self._failed_buffer[:retry_count])
            self._failed_buffer = self._failed_buffer[retry_count:]

        if not batch:
            return

        # Send batch
        success = self._send_batch(batch)

        if not success:
            # Add back to failed buffer for retry
            self._failed_buffer.extend(batch)
            # Limit failed buffer size
            if len(self._failed_buffer) > self.MAX_BUFFER_SIZE:
                dropped = len(self._failed_buffer) - self.MAX_BUFFER_SIZE
                self._failed_buffer = self._failed_buffer[-self.MAX_BUFFER_SIZE :]
                self._stats["dropped"] += dropped

    def _flush_sync(self) -> None:
        """Synchronous flush for shutdown."""
        batch = []
        while not self._buffer.empty() and len(batch) < self.MAX_BUFFER_SIZE:
            try:
                batch.append(self._buffer.get_nowait())
            except queue.Empty:
                break

        batch.extend(self._failed_buffer)
        self._failed_buffer = []

        if batch:
            self._send_batch(batch)

    def _send_batch(self, batch: list[CommandFeedback]) -> bool:
        """
        Send a batch of feedback to the server.

        Args:
            batch: List of feedback items

        Returns:
            True if successful, False otherwise
        """
        try:
            import urllib.error
            import urllib.request

            url = f"{self.server_url}/api/v1/feedback"

            payload = {
                "session_hash": self._session_hash,
                "feedback": [f.to_dict() for f in batch],
            }

            data = json.dumps(payload).encode("utf-8")

            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["X-API-Key"] = self.api_key

            req = urllib.request.Request(url, data=data, headers=headers, method="POST")

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    self._stats["sent"] += len(batch)
                    logger.debug(f"Sent {len(batch)} feedback items")
                    return True
                else:
                    self._stats["failed"] += len(batch)
                    logger.warning(f"Feedback send failed: {response.status}")
                    return False

        except urllib.error.URLError as e:
            self._stats["failed"] += len(batch)
            logger.debug(f"Feedback send failed (server unavailable): {e}")
            return False
        except Exception as e:
            self._stats["failed"] += len(batch)
            logger.warning(f"Feedback send error: {e}")
            return False


# Global feedback client instance
_feedback_client: Optional[FeedbackClient] = None


def get_feedback_client() -> FeedbackClient:
    """Get or create the global feedback client."""
    global _feedback_client
    if _feedback_client is None:
        _feedback_client = FeedbackClient()
    return _feedback_client


def init_feedback_client(
    server_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
    enabled: bool = True,
) -> FeedbackClient:
    """
    Initialize and start the global feedback client.

    Args:
        server_url: Base URL of the Conjure server
        api_key: API key for authentication
        session_id: Session identifier (will be hashed)
        enabled: Whether feedback collection is enabled

    Returns:
        The initialized feedback client
    """
    global _feedback_client

    if _feedback_client:
        _feedback_client.stop()

    _feedback_client = FeedbackClient(
        server_url=server_url,
        api_key=api_key,
        session_id=session_id,
        enabled=enabled,
    )
    _feedback_client.start()
    return _feedback_client


def stop_feedback_client() -> None:
    """Stop the global feedback client."""
    global _feedback_client
    if _feedback_client:
        _feedback_client.stop()
        _feedback_client = None
