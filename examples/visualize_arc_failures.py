"""Visualize failing ARC tasks from evaluation results."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import textwrap

Grid = Sequence[Sequence[Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("arc_results.json"),
        help="Path to the JSON results file produced by evaluate_arc_dataset.py.",
    )
    parser.add_argument(
        "--challenges",
        type=Path,
        default=Path("tests/data/arc/arc-agi_training_challenges.json"),
        help="ARC training challenges JSON (inputs).",
    )
    parser.add_argument(
        "--solutions",
        type=Path,
        default=Path("tests/data/arc/arc-agi_training_solutions.json"),
        help="ARC training solutions JSON (expected test outputs).",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Focus on specific task ids. May be supplied multiple times.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"Missing file: {path}")


def grid_to_key(grid: Grid) -> Tuple[Tuple[Any, ...], ...]:
    return tuple(tuple(row) for row in grid)


def format_grid(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def format_diff(expected: Grid, actual: Grid) -> Optional[str]:
    if not isinstance(expected, Sequence) or not isinstance(actual, Sequence):
        return None
    if len(expected) != len(actual):
        return None
    diff_rows: List[str] = []
    for exp_row, act_row in zip(expected, actual):
        if not isinstance(exp_row, Sequence) or not isinstance(act_row, Sequence):
            return None
        if len(exp_row) != len(act_row):
            return None
        markers = ["." if e == a else "X" for e, a in zip(exp_row, act_row)]
        diff_rows.append(" ".join(markers))
    return "\n".join(diff_rows)


def build_example_index(
    task_id: str,
    challenges: Dict[str, Any],
    solutions: Dict[str, Any],
) -> Dict[Tuple[Tuple[Any, ...], ...], List[Dict[str, Any]]]:
    payload = challenges.get(task_id)
    if payload is None:
        return {}

    index: Dict[Tuple[Tuple[Any, ...], ...], List[Dict[str, Any]]] = {}

    for idx, pair in enumerate(payload.get("train", []), start=1):
        entry = {
            "split": "train",
            "index": idx,
            "input": pair["input"],
            "expected": pair["output"],
        }
        index.setdefault(grid_to_key(pair["input"]), []).append(entry)

    expected_tests: List[Any] = solutions.get(task_id, [])
    for idx, pair in enumerate(payload.get("test", []), start=1):
        expected = expected_tests[idx - 1] if idx - 1 < len(expected_tests) else None
        entry = {
            "split": "test",
            "index": idx,
            "input": pair["input"],
            "expected": expected,
        }
        index.setdefault(grid_to_key(pair["input"]), []).append(entry)

    return index


def parse_feedback(feedback: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for raw_line in feedback.splitlines():
        line = raw_line.strip()
        if not line or line == "Failures detected:":
            continue
        if not line.startswith("Mismatch for input args="):
            issues.append({"message": line})
            continue

        try:
            prefix, expected_part = line.split(" expected ", 1)
            expected_str, got_str = expected_part.split(" got ", 1)
            args_part = prefix[len("Mismatch for input args="):]
            args_str, kwargs_part = args_part.split(" kwargs=", 1)
            kwargs_str = kwargs_part.rstrip(":")
            parsed = {
                "args": ast.literal_eval(args_str),
                "kwargs": ast.literal_eval(kwargs_str),
                "expected": ast.literal_eval(expected_str),
                "got": ast.literal_eval(got_str),
            }
            issues.append(parsed)
        except (ValueError, SyntaxError):
            issues.append({"message": line})
    return issues


def pick_example(
    index: Dict[Tuple[Tuple[Any, ...], ...], List[Dict[str, Any]]],
    input_grid: Grid,
    expected_grid: Grid,
) -> Optional[Dict[str, Any]]:
    matches = index.get(grid_to_key(input_grid), [])
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    for candidate in matches:
        if candidate.get("expected") == expected_grid:
            return candidate
    return matches[0]


def visualize_task(
    result: Dict[str, Any],
    challenges: Dict[str, Any],
    solutions: Dict[str, Any],
) -> None:
    task_id = result["task_id"]
    index = build_example_index(task_id, challenges, solutions)
    issues = parse_feedback(result.get("feedback", ""))

    print(f"\nTask {task_id} | score={result['score']:.2f} attempts={result['attempts']} duration={result['duration_sec']:.1f}s")
    if not issues:
        print("  No detailed failure information available.")
        return

    for position, issue in enumerate(issues, start=1):
        if "message" in issue:
            print(f"  Issue {position}: {issue['message']}")
            continue

        args = issue.get("args", ())
        input_grid = args[0] if args else None
        expected_grid = issue.get("expected")
        actual_grid = issue.get("got")

        if input_grid is None:
            print(f"  Issue {position}: unable to parse input grid.")
            continue

        example = pick_example(index, input_grid, expected_grid)
        if example:
            label = f"{example['split']} #{example['index']}"
        else:
            label = "unknown example"

        print(f"  Issue {position}: {label}")
        print("    Input:")
        print(textwrap.indent(format_grid(input_grid), "      "))
        if expected_grid is not None:
            print("    Expected:")
            print(textwrap.indent(format_grid(expected_grid), "      "))
        if actual_grid is not None:
            print("    Got:")
            print(textwrap.indent(format_grid(actual_grid), "      "))
            if expected_grid is not None:
                diff = format_diff(expected_grid, actual_grid)
                if diff:
                    print("    Diff (X = mismatch):")
                    print(textwrap.indent(diff, "      "))


def main() -> None:
    args = parse_args()
    results_payload = load_json(args.results)
    challenges = load_json(args.challenges)
    solutions = load_json(args.solutions)
    task_filter = set(args.tasks) if args.tasks else None

    results: Iterable[Dict[str, Any]] = results_payload.get("results", [])
    displayed = False
    for item in results:
        if item.get("passed"):
            continue
        if task_filter and item.get("task_id") not in task_filter:
            continue
        visualize_task(item, challenges, solutions)
        displayed = True

    if not displayed:
        print("No failing tasks matched the provided filters.")


if __name__ == "__main__":
    main()
