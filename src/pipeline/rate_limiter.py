import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, max_calls: int, period_seconds: float):
        self._max_calls = max_calls
        self._period = period_seconds
        self._calls = deque()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            self._evict_expired()
            if len(self._calls) >= self._max_calls:
                sleep_time = self._period - (time.monotonic() - self._calls[0]) + 0.05
                time.sleep(max(sleep_time, 0))
                self._evict_expired()
            self._calls.append(time.monotonic())

    def _evict_expired(self):
        now = time.monotonic()
        while self._calls and now - self._calls[0] >= self._period:
            self._calls.popleft()