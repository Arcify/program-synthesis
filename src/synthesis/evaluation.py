"""Evaluation utilities for candidate programs."""

from __future__ import annotations

from typing import Optional

from .plugins.python_executor import PythonExecutor
from .tasks import CandidateProgram, EvaluationOutcome, TaskSpecification


def evaluate_candidate(
    candidate: CandidateProgram,
    task: TaskSpecification,
    executor: Optional[PythonExecutor] = None,
) -> EvaluationOutcome:
    """Run the candidate against the task examples and compute a score."""

    executor = executor or PythonExecutor()
    report = executor.run(candidate, task)
    total_cases = max(len(task.examples), 1)
    failure_ratio = min(len(report.failures), total_cases) / total_cases
    score = max(0.0, 1.0 - failure_ratio)

    if report.passed:
        feedback = "All tests passed."
    else:
        joined = "\n".join(report.failures)
        feedback = f"Failures detected:\n{joined}"

    return EvaluationOutcome(
        passed=report.passed,
        score=score,
        feedback=feedback,
        details={
            "outputs": report.outputs,
            "failures": report.failures,
        },
    )
