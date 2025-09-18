"""Shared utility helpers for the synthesis pipeline."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator


@contextmanager
def timer() -> Iterator[Callable[[], float]]:
    """Yield a callable that returns the elapsed time when invoked."""

    start = time.perf_counter()
    elapsed = lambda: time.perf_counter() - start
    yield elapsed


def format_score(passed: bool, score: float) -> str:
    status = "PASS" if passed else "FAIL"
    return f"[{status}] score={score:.2f}"
