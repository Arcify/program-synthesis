import json
import pathlib
import unittest

from synthesis import (
    IOExample,
    MockLLM,
    SynthesisConfig,
    SynthesisPipeline,
    TaskSpecification,
)

ARC_SAMPLE_PATH = pathlib.Path(__file__).parent / "data" / "arc" / "0d3d703e.json"


class ARCBenchmarkTest(unittest.TestCase):
    """Integration test that exercises an ARC-AGI benchmark puzzle (Apache License 2.0)."""

    def test_color_permutation_task(self) -> None:
        arc_data = json.loads(ARC_SAMPLE_PATH.read_text())

        color_map = {}
        for pair in arc_data["train"]:
            for input_row, output_row in zip(pair["input"], pair["output"]):
                for input_value, output_value in zip(input_row, output_row):
                    color_map[input_value] = output_value

        solution_lines = ["def solve_arc(grid):"]
        solution_lines.append("    color_map = {")
        for key in sorted(color_map):
            solution_lines.append(f"        {key}: {color_map[key]},")
        solution_lines.append("    }")
        solution_lines.append("    return [[color_map[value] for value in row] for row in grid]")
        solution_code = "\n".join(solution_lines)

        llm = MockLLM(canned_responses={"0d3d703e": solution_code})

        examples = []
        expected_outputs = []
        for split in ("train", "test"):
            for pair in arc_data.get(split, []):
                examples.append(
                    IOExample(
                        args=(pair["input"],),
                        expected=pair["output"],
                        explanation=f"ARC {split} pair",
                    )
                )
                expected_outputs.append(pair["output"])

        task = TaskSpecification(
            task_id="arc-0d3d703e",
            description=("Solve the ARC-AGI color permutation puzzle 0d3d703e by remapping palette values."),
            language="python",
            entry_point="solve_arc",
            examples=examples,
            constraints="Return a grid with remapped colors for every cell.",
        )

        pipeline = SynthesisPipeline(llm, config=SynthesisConfig(max_iterations=1))
        outcome = pipeline.run(task)

        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.score, 1.0)
        self.assertEqual(expected_outputs, outcome.details["outputs"])

        state = pipeline.last_state
        self.assertIsNotNone(state)
        assert state is not None  # mypy guard
        self.assertEqual(len(state.history), 1)
        self.assertEqual(state.history[0]["code"].strip(), solution_code)


if __name__ == "__main__":
    unittest.main()
