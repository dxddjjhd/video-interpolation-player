"""有界线程安全队列（流水线缓冲）。

采用"队满丢最旧"策略：实时播放场景下，解码/插帧跟不上时
丢弃最旧的缓冲帧，保证流水线延迟可控、不阻塞上游线程。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class BoundedQueue(Generic[T]):
    """有界队列：put 满时丢弃队首（最旧）元素，保证永不阻塞。"""

    def __init__(self, maxsize: int) -> None:
        assert maxsize > 0
        self._maxsize = maxsize
        self._q: deque[T] = deque()
        self._cv = threading.Condition()
        self._closed = False

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def put(self, item: T) -> bool:
        """入队；若已关闭返回 False，队满则丢弃最旧元素。"""
        with self._cv:
            if self._closed:
                return False
            if len(self._q) >= self._maxsize:
                self._q.popleft()
            self._q.append(item)
            self._cv.notify()
            return True

    def put_limited(self, item: T, limit: int | None = None) -> bool:
        """入队但队内元素 >= limit 时等待（不丢数据）。

        用于给上游解码器限速：队列有空位才继续解码，
        避免无音频背压时解码过快、丢帧。返回 False 表示队列已关闭。
        """
        limit = self._maxsize if limit is None else min(limit, self._maxsize)
        with self._cv:
            while len(self._q) >= limit:
                self._cv.wait(timeout=0.05)
                if self._closed:
                    return False
            self._q.append(item)
            self._cv.notify()
            return True

    def get(self, timeout: float | None = None) -> T | None:
        """出队；超时或队列已关闭且清空时返回 None。"""
        with self._cv:
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self._q:
                if self._closed:
                    return None
                if deadline is None:
                    self._cv.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cv.wait(remaining)
            return self._q.popleft()

    def qsize(self) -> int:
        with self._cv:
            return len(self._q)

    def empty(self) -> bool:
        with self._cv:
            return not self._q

    def is_closed(self) -> bool:
        with self._cv:
            return self._closed

    def clear(self) -> None:
        with self._cv:
            self._q.clear()
            self._cv.notify_all()

    def close(self) -> None:
        """关闭队列：清空并唤醒所有等待者。"""
        with self._cv:
            self._closed = True
            self._q.clear()
            self._cv.notify_all()
