"""Inspect the pipeline solving the ARC-AGI 0d3d703e benchmark puzzle."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

from synthesis import (
    IOExample,
    MockLLM,
    OpenAIClient,
    OpenAIConfig,
    SynthesisConfig,
    SynthesisPipeline,
    TaskSpecification,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARC_SAMPLE_PATH = ROOT / "tests" / "data" / "arc" / "0d3d703e.json"


def build_color_map(arc_payload: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[int, int]:
    """Derive the palette permutation from ARC training pairs."""

    mapping: Dict[int, int] = {}
    for pair in arc_payload["train"]:
        for input_row, output_row in zip(pair["input"], pair["output"]):
            for input_value, output_value in zip(input_row, output_row):
                mapping[input_value] = output_value
    return mapping


def synthesize_program(mapping: Dict[int, int]) -> str:
    """Materialize the Python solution that remaps colors using the learned palette."""

    lines = ["def solve_arc(grid):"]
    lines.append("    color_map = {")
    for key in sorted(mapping):
        lines.append(f"        {key}: {mapping[key]},")
    lines.append("    }")
    lines.append("    return [[color_map[value] for value in row] for row in grid]")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm",
        choices=("mock", "openai"),
        default="mock",
        help="Which LLM backend to use. `mock` provides a deterministic canned solution.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Override the OpenAI model identifier when using the openai backend.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum synthesis iterations to allow when calling a real LLM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arc_data = json.loads(ARC_SAMPLE_PATH.read_text())

    examples = []
    for split in ("train", "test"):
        for pair in arc_data.get(split, []):
            examples.append(
                IOExample(
                    args=(pair["input"],),
                    expected=pair["output"],
                    explanation=f"ARC {split} pair",
                )
            )

    if args.llm == "mock":
        color_map = build_color_map(arc_data)
        solution_code = synthesize_program(color_map)
        llm = MockLLM(canned_responses={"0d3d703e": solution_code})
        max_iterations = 1
    else:
        llm = OpenAIClient(config=OpenAIConfig(model=args.model))
        solution_code = None
        max_iterations = args.max_iterations

    task = TaskSpecification(
        task_id="arc-0d3d703e",
        description="Solve the ARC-AGI color permutation puzzle 0d3d703e by remapping palette values.",
        language="python",
        entry_point="solve_arc",
        examples=examples,
        constraints="Return a grid with remapped colors for every cell.",
    )

    pipeline = SynthesisPipeline(llm, config=SynthesisConfig(max_iterations=max_iterations))
    outcome = pipeline.run(task)

    print("Outcome: passed=", outcome.passed, " score=", outcome.score, sep="")

    state = pipeline.last_state
    if state and state.history:
        final_code = state.history[-1]["code"].strip()
        print("\nSynthesized candidate code:\n")
        print(final_code)
    elif solution_code:
        print("\nSynthesized candidate code (mock):\n")
        print(solution_code)

    print("\nOutputs per example:")
    for index, grid in enumerate(outcome.details["outputs"], start=1):
        print(f"Example {index}: {grid}")

    print("\nAttempt history:")
    print(pipeline.summarize())


if __name__ == "__main__":
    main()
