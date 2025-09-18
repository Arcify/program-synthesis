"""Task specifications and synthesis artifact representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class IOExample:
    """Represents a single input/output example for a task."""

    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    expected: Any = None
    explanation: Optional[str] = None

    def pretty(self) -> str:
        arg_repr = ", ".join(repr(arg) for arg in self.args)
        kw_repr = ", ".join(f"{key}={value!r}" for key, value in self.kwargs.items())
        call = ", ".join(filter(None, [arg_repr, kw_repr]))
        lines = [f"call: ({call})"]
        lines.append(f"expected: {self.expected!r}")
        if self.explanation:
            lines.append(f"note: {self.explanation}")
        return "\n".join(lines)


@dataclass
class TaskSpecification:
    """Defines a program synthesis challenge."""

    task_id: str
    description: str
    language: str
    entry_point: str
    examples: List[IOExample]
    constraints: Optional[str] = None
    reward_model: Optional[str] = None

    def format_examples(self) -> str:
        return "\n\n".join(example.pretty() for example in self.examples)


@dataclass
class CandidateProgram:
    """Stores a candidate program along with metadata."""

    code: str
    language: str
    attempt: int
    origin: str = "llm"


@dataclass
class EvaluationOutcome:
    """Holds evaluation results for a candidate program."""

    passed: bool
    score: float
    feedback: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisState:
    """Tracks the iterative synthesis process for a single task."""

    task: TaskSpecification
    attempt: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, candidate: CandidateProgram, outcome: EvaluationOutcome) -> None:
        self.attempt += 1
        self.history.append(
            {
                "attempt": candidate.attempt,
                "code": candidate.code,
                "passed": outcome.passed,
                "score": outcome.score,
                "feedback": outcome.feedback,
            }
        )

    def last_feedback(self) -> Optional[str]:
        if not self.history:
            return None
        return self.history[-1]["feedback"]

    def best_score(self) -> float:
        scores = [entry["score"] for entry in self.history]
        return max(scores) if scores else 0.0

    def successes(self) -> Iterable[Dict[str, Any]]:
        return [entry for entry in self.history if entry["passed"]]
