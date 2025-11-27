#!/usr/bin/env python3
"""
Tests for VestDbPollingWorker.

These tests validate the basic polling logic of VestDbPollingWorker using
fake VEST_DB objects so that no real database is required.
"""

from __future__ import annotations

import threading
import time
from typing import List

import pytest

from ifi.gui.workers.vest_db_worker import VestDbPollingWorker


class FakeVestDb:
    """
    Simple fake VEST_DB implementation for testing.

    It returns a configurable sequence of shot codes each time
    get_next_shot_code() is called.
    """

    def __init__(self, shot_codes: List[int]) -> None:
        self._shot_codes = shot_codes
        self._index = 0

    def get_next_shot_code(self) -> int:
        if self._index >= len(self._shot_codes):
            return self._shot_codes[-1]
        value = self._shot_codes[self._index]
        self._index += 1
        return value


def test_vest_db_polling_worker_invokes_callback_on_increase() -> None:
    """
    The worker should invoke the callback when it observes a strictly
    increasing shot code.
    """
    fake_db = FakeVestDb([100, 100, 101, 101])

    received: list[int] = []
    callback_event = threading.Event()

    def on_new_shot_code(value: int) -> None:
        received.append(value)
        callback_event.set()

    worker = VestDbPollingWorker(
        vest_db=fake_db,
        on_new_shot_code=on_new_shot_code,
        poll_interval=0.01,
    )

    try:
        worker.start_polling()

        # Wait briefly for the callback to be invoked.
        assert callback_event.wait(timeout=0.5) is True

        # We should have observed at least the new value 101.
        assert 101 in received
    finally:
        worker.stop_polling()
        # Give the thread a moment to exit to avoid leaking threads in tests.
        worker.join(timeout=0.5)


def test_vest_db_polling_worker_does_not_call_callback_when_constant() -> None:
    """
    When the shot code does not increase, the callback should never be
    invoked.
    """
    fake_db = FakeVestDb([200, 200, 200])

    received: list[int] = []
    callback_event = threading.Event()

    def on_new_shot_code(value: int) -> None:
        received.append(value)
        callback_event.set()

    worker = VestDbPollingWorker(
        vest_db=fake_db,
        on_new_shot_code=on_new_shot_code,
        poll_interval=0.01,
    )

    try:
        worker.start_polling()

        # Wait a bit; callback should not be set.
        with pytest.raises(AssertionError):
            assert callback_event.wait(timeout=0.2) is True

        assert received == []
    finally:
        worker.stop_polling()
        worker.join(timeout=0.5)


