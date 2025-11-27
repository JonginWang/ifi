#!/usr/bin/env python3
"""
VEST DB Polling Worker
======================

This module provides a polling worker that periodically queries the VEST
database for the latest (next) shot code and notifies a callback when the
value increases. It is designed to be used from Tkinter-based GUI code as
opposed to Qt/QThread, while following the intent of Task 33.

The worker runs in its own thread and must be started and stopped explicitly
from the GUI layer.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

try:
    from ...utils.common import LogManager, log_tag
    from ...db_controller.vest_db import VEST_DB
except ImportError as e:  # pragma: no cover - import fallback
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.utils.common import LogManager, log_tag
    from ifi.db_controller.vest_db import VEST_DB


LogManager()
logger = LogManager().get_logger(__name__)


class VestDbPollingWorker(threading.Thread):
    """
    Worker thread that periodically polls the VEST DB for shot codes.

    This is a Tkinter-friendly analogue to the QThread-based worker described
    in Task 33. Instead of emitting Qt signals, it invokes a callback
    `on_new_shot_code` in the GUI layer when a new shot code is detected.

    Args:
        vest_db: Connected VEST_DB instance to query.
        on_new_shot_code: Callback invoked with the new shot code (int) when
            it increases compared to the previously seen value.
        poll_interval: Polling interval in seconds. Defaults to 30.0.

    Attributes:
        _running: Internal flag controlling the main polling loop.
        _last_shot_code: The last seen shot code, used to detect increments.

    TODO:
        Add an optional, integration-style test that exercises this worker
        against a real VEST DB instance. Such a test should:
            - Use a test configuration file/DB with controlled data.
            - Start the worker, let it run for several intervals, and verify
              that the callback is invoked when new rows are inserted.
            - Be guarded by a pytest marker or environment flag so it is
              skipped by default in environments without DB access.
    """

    def __init__(
        self,
        vest_db: VEST_DB,
        on_new_shot_code: Callable[[int], None],
        poll_interval: float = 30.0,
    ) -> None:
        super().__init__(daemon=True)
        self._vest_db = vest_db
        self._on_new_shot_code = on_new_shot_code
        self._poll_interval = poll_interval
        self._running = threading.Event()
        self._last_shot_code: Optional[int] = None

    def start_polling(self) -> None:
        """
        Start the polling loop in the background thread.

        If the worker is already running, this call is a no-op.
        """
        if self._running.is_set():
            return
        self._running.set()
        if not self.is_alive():
            self.start()

    def stop_polling(self) -> None:
        """
        Request the polling loop to stop.

        The underlying thread will exit gracefully after the current sleep
        interval completes.
        """
        self._running.clear()

    def run(self) -> None:
        """Main polling loop executed in the background thread."""
        logger.info(f"{log_tag('VESTB', 'POLL ')} VEST DB polling worker started.")

        while self._running.is_set():
            try:
                shot_code = self._vest_db.get_next_shot_code()
                if isinstance(shot_code, int):
                    if self._last_shot_code is None:
                        self._last_shot_code = shot_code
                    elif shot_code > self._last_shot_code:
                        logger.info(
                            f"{log_tag('VESTB', 'POLL ')} New shot code detected: {shot_code}"
                        )
                        self._last_shot_code = shot_code
                        # Notify GUI layer about the new shot code
                        self._on_new_shot_code(shot_code)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(
                    f"{log_tag('VESTB', 'POLL ')} Error while polling VEST DB: {exc}"
                )

            time.sleep(self._poll_interval)

        logger.info(f"{log_tag('VESTB', 'POLL ')} VEST DB polling worker stopped.")


