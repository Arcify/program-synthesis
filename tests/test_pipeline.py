import unittest

from synthesis import (
    IOExample,
    MockLLM,
    SynthesisConfig,
    SynthesisPipeline,
    TaskSpecification,
)


class PipelineTest(unittest.TestCase):
    def test_pipeline_completes_successfully(self) -> None:
        expected_code = "def reverse_string(value):\n    return value[::-1]\n"
        llm = MockLLM(
            canned_responses={
                "reverse": expected_code,
            }
        )

        task = TaskSpecification(
            task_id="reverse-string",
            description="Implement `reverse_string` that reverses a string.",
            language="python",
            entry_point="reverse_string",
            examples=[
                IOExample(args=("hello",), expected="olleh"),
                IOExample(args=("Codex",), expected="xedoC"),
            ],
        )

        pipeline = SynthesisPipeline(llm, config=SynthesisConfig(max_iterations=2))
        outcome = pipeline.run(task)

        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.score, 1.0)

    def test_pipeline_exposes_best_candidate_metadata(self) -> None:
        expected_code = "def reverse_string(value):\n    return value[::-1]\n"
        llm = MockLLM(
            canned_responses={
                "reverse": expected_code,
            }
        )

        task = TaskSpecification(
            task_id="reverse-string",
            description="Implement `reverse_string` that reverses a string.",
            language="python",
            entry_point="reverse_string",
            examples=[
                IOExample(args=("hello",), expected="olleh"),
                IOExample(args=("Codex",), expected="xedoC"),
            ],
        )

        pipeline = SynthesisPipeline(llm, config=SynthesisConfig(max_iterations=1))
        outcome = pipeline.run(task)

        candidate_details = outcome.details.get("candidate")
        self.assertIsNotNone(candidate_details)
        assert candidate_details is not None
        self.assertEqual(candidate_details["attempt"], 1)
        self.assertEqual(candidate_details["origin"], "canned")
        self.assertEqual(candidate_details["code"].strip(), expected_code.strip())

        best_candidate = pipeline.best_candidate
        self.assertIsNotNone(best_candidate)
        assert best_candidate is not None
        self.assertEqual(best_candidate.code.strip(), expected_code.strip())
        self.assertEqual(best_candidate.origin, "canned")

    def test_syntax_error_is_reported_as_compilation_failure(self) -> None:
        invalid_code = "def reverse_string(value)\n    return value[::-1]\n"
        llm = MockLLM(
            canned_responses={
                "reverse": invalid_code,
            }
        )

        task = TaskSpecification(
            task_id="reverse-string",
            description="Implement `reverse_string` that reverses a string.",
            language="python",
            entry_point="reverse_string",
            examples=[
                IOExample(args=("hello",), expected="olleh"),
                IOExample(args=("Codex",), expected="xedoC"),
            ],
        )

        pipeline = SynthesisPipeline(llm, config=SynthesisConfig(max_iterations=1))
        outcome = pipeline.run(task)

        self.assertFalse(outcome.passed)
        self.assertEqual(
            outcome.details.get("candidate", {}).get("code"),
            invalid_code.strip(),
        )
        self.assertIn("Compilation error", outcome.feedback)
        self.assertTrue(
            any(message.startswith("Compilation error:") for message in outcome.details["failures"])
        )
        self.assertEqual(outcome.score, 0.0)

        best_candidate = pipeline.best_candidate
        self.assertIsNotNone(best_candidate)
        assert best_candidate is not None
        self.assertEqual(best_candidate.code, invalid_code.strip())


if __name__ == "__main__":
    unittest.main()
