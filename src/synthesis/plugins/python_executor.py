"""Execution plugin for running synthesized Python programs."""

from __future__ import annotations

import builtins
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List

from ..tasks import CandidateProgram, TaskSpecification


_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in [
        "abs",
        "all",
        "any",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "pow",
        "print",
        "range",
        "sum",
        "zip",
    ]
}


@dataclass
class ExecutionReport:
    """Result of executing a candidate program across IO examples."""

    passed: bool
    outputs: List[Any]
    failures: List[str]


class PythonExecutor:
    """Runs synthesized Python code against IO examples."""

    def __init__(self, entry_filename: str = "candidate.py") -> None:
        self.entry_filename = entry_filename

    def run(
        self,
        candidate: CandidateProgram,
        task: TaskSpecification,
    ) -> ExecutionReport:
        try:
            namespace = self._build_namespace(candidate.code)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = self._format_compile_error(exc)
            failure_count = max(len(task.examples), 1)
            return ExecutionReport(
                passed=False, outputs=[], failures=[message] * failure_count
            )

        entry = namespace.get(task.entry_point)
        if not callable(entry):
            failure = f"Entry point '{task.entry_point}' not found in candidate program."
            return ExecutionReport(passed=False, outputs=[], failures=[failure])

        outputs: List[Any] = []
        failures: List[str] = []

        for example in task.examples:
            try:
                result = entry(*example.args, **example.kwargs)
            except Exception as exc:  # pragma: no cover - defensive guard
                failures.append(f"Runtime error: {exc!r}")
                continue

            outputs.append(result)
            if result != example.expected:
                failures.append(
                    "Mismatch for input "
                    f"args={example.args!r} kwargs={example.kwargs!r}:"
                    f" expected {example.expected!r} got {result!r}"
                )

        passed = not failures
        return ExecutionReport(passed=passed, outputs=outputs, failures=failures)

    # ------------------------------------------------------------------
    def _build_namespace(self, code: str) -> Dict[str, Any]:
        compiled_globals: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS.copy()}
        compiled_locals: Dict[str, Any] = {}
        exec(code, compiled_globals, compiled_locals)
        compiled_globals.update(compiled_locals)
        return compiled_globals

    def _format_compile_error(self, exc: Exception) -> str:
        summary = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        if not summary:
            summary = repr(exc)
        return f"Compilation error: {summary}"
