"""Demonstration of the program synthesis pipeline with a mock LLM."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthesis import (
    IOExample,
    MockLLM,
    SynthesisConfig,
    SynthesisPipeline,
    TaskSpecification,
)


def main() -> None:
    llm = MockLLM(
        canned_responses={
            "sum": "def compute_sum(values):\n    return sum(values)\n",
        }
    )

    task = TaskSpecification(
        task_id="sum-list",
        description="Write a function compute_sum that returns the sum of a list of integers.",
        language="python",
        entry_point="compute_sum",
        examples=[
            IOExample(args=([1, 2, 3],), expected=6),
            IOExample(args=([5, -2, 7],), expected=10),
        ],
        constraints="Avoid using external libraries.",
    )

    pipeline = SynthesisPipeline(llm, config=SynthesisConfig(max_iterations=3))
    outcome = pipeline.run(task)

    print("Best outcome:", outcome.passed, outcome.score)
    print("Feedback:\n", outcome.feedback)
    print("\nAttempt history:")
    print(pipeline.summarize())


if __name__ == "__main__":
    main()
