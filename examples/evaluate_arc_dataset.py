"""Batch evaluation of the synthesis pipeline on ARC-AGI training tasks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from pathlib import Path

from synthesis import (
    IOExample,
    MockLLM,
    OpenAIClient,
    OpenAIConfig,
    SynthesisConfig,
    SynthesisPipeline,
    TaskSpecification,
)


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    score: float
    attempts: int
    duration_sec: float
    feedback: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--challenges",
        type=Path,
        default=Path("tests/data/arc/arc-agi_training_challenges.json"),
        help="Path to the ARC training challenges JSON file.",
    )
    parser.add_argument(
        "--solutions",
        type=Path,
        default=Path("tests/data/arc/arc-agi_training_solutions.json"),
        help="Path to the ARC training solutions JSON file.",
    )
    parser.add_argument(
        "--llm",
        choices=("openai", "mock"),
        default="openai",
        help="LLM backend to use for synthesis runs.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="OpenAI model identifier (when using the openai backend).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of synthesis iterations per task.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of tasks processed (after filtering and sorting).",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Run only specific ARC task ids. May be provided multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write detailed JSON results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the tasks that would run without invoking the LLM.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def build_examples(
    task_id: str,
    payload: Dict[str, object],
    solution_lookup: Dict[str, List[object]],
) -> Iterable[IOExample]:
    train_pairs = payload.get("train", [])  # type: ignore[assignment]
    for index, pair in enumerate(train_pairs, start=1):
        yield IOExample(
            args=(pair["input"],),
            expected=pair["output"],
            explanation=f"{task_id} train #{index}",
        )

    expected_tests = solution_lookup.get(task_id, [])
    test_pairs = payload.get("test", [])  # type: ignore[assignment]
    for index, pair in enumerate(test_pairs, start=1):
        if index - 1 >= len(expected_tests):
            continue
        yield IOExample(
            args=(pair["input"],),
            expected=expected_tests[index - 1],
            explanation=f"{task_id} test #{index}",
        )


def make_task_spec(task_id: str, payload: Dict[str, object], solutions: Dict[str, List[object]]) -> Optional[TaskSpecification]:
    examples = list(build_examples(task_id, payload, solutions))
    if not examples:
        return None
    description = (
        "Implement solve_arc(grid) to transform ARC inputs into outputs for task "
        f"{task_id}."
    )
    constraints = "Use only pure Python; return a new grid without modifying inputs."
    return TaskSpecification(
        task_id=task_id,
        description=description,
        language="python",
        entry_point="solve_arc",
        examples=examples,
        constraints=constraints,
    )


def build_llm(args: argparse.Namespace):
    if args.llm == "mock":
        return MockLLM()
    config = OpenAIConfig(model=args.model)
    return OpenAIClient(config=config)


def run_task(pipeline: SynthesisPipeline, spec: TaskSpecification) -> TaskResult:
    start = time.perf_counter()
    outcome = pipeline.run(spec)
    duration = time.perf_counter() - start
    state = pipeline.last_state
    attempts = len(state.history) if state else 0
    return TaskResult(
        task_id=spec.task_id,
        passed=outcome.passed,
        score=outcome.score,
        attempts=attempts,
        duration_sec=duration,
        feedback=outcome.feedback,
    )


def summarize(results: List[TaskResult]) -> Dict[str, object]:
    if not results:
        return {"total": 0, "passed": 0, "pass_rate": 0.0, "mean_score": 0.0}
    passed = sum(1 for item in results if item.passed)
    scores = [item.score for item in results]
    durations = [item.duration_sec for item in results]
    return {
        "total": len(results),
        "passed": passed,
        "pass_rate": passed / len(results),
        "mean_score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "total_runtime_sec": sum(durations),
    }


def main() -> None:
    args = parse_args()
    challenges = load_json(args.challenges)
    solutions = load_json(args.solutions)

    llm = build_llm(args)
    pipeline = SynthesisPipeline(llm, config=SynthesisConfig(max_iterations=args.max_iterations))

    task_ids = sorted(challenges.keys())
    if args.task_ids:
        wanted = set(args.task_ids)
        task_ids = [task_id for task_id in task_ids if task_id in wanted]
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    if args.dry_run:
        for task_id in task_ids:
            print(task_id)
        return

    results: List[TaskResult] = []
    for index, task_id in enumerate(task_ids, start=1):
        payload = challenges[task_id]  # type: ignore[index]
        spec = make_task_spec(task_id, payload, solutions)
        if spec is None:
            print(f"[{index}/{len(task_ids)}] Skipping {task_id}: no examples available.")
            continue

        print(f"[{index}/{len(task_ids)}] Running task {task_id}...", flush=True)
        try:
            result = run_task(pipeline, spec)
        except Exception as exc:  # pragma: no cover - diagnostic output
            print(f"    Failed with exception: {exc}")
            continue

        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"    {status} score={result.score:.2f} attempts={result.attempts} "
            f"time={result.duration_sec:.1f}s",
            flush=True,
        )

    summary = summarize(results)
    print("\nSummary:")
    print(
        f"  tasks={summary['total']} passed={summary['passed']} pass_rate={summary['pass_rate']:.2%}"
    )
    print(
        f"  mean_score={summary['mean_score']:.3f} median_score={summary['median_score']:.3f}"
    )
    print(f"  total_runtime={summary['total_runtime_sec']:.1f}s")

    if args.output:
        payload = {
            "summary": summary,
            "results": [asdict(item) for item in results],
            "config": {
                "llm": args.llm,
                "model": args.model,
                "max_iterations": args.max_iterations,
                "dataset": {
                    "challenges": str(args.challenges),
                    "solutions": str(args.solutions),
                    "count": len(task_ids),
                },
            },
        }
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
