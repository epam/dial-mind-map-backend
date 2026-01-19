import threading
import time
from contextvars import ContextVar
from typing import ClassVar

cur_run_id = ContextVar[str]("run_id", default="system")


class HybridIDGenerator:
    """
    A thread-safe hybrid ID generator
    that combines timestamp and a counter.

    Generates unique IDs by using microsecond-precision timestamps
    and a counter that increments when multiple IDs are requested
    within the same microsecond.
    """

    _prev_time: ClassVar[int] = 0
    _counter: ClassVar[int] = 0
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_next_id(cls) -> str:
        """
        Generate a unique ID by combining current time and a counter.

        Returns:
            str: A unique ID in the format "timestamp-counter"
        """
        with cls._lock:
            # Use time_ns() for better precision, convert to microseconds
            current_time = time.time_ns() // 1000

            if current_time == cls._prev_time:
                cls._counter += 1
            else:
                cls._counter = 0
                cls._prev_time = current_time

            return f"{current_time}-{cls._counter}"
