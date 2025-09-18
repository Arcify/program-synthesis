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
        llm = MockLLM(
            canned_responses={
                "reverse": "def reverse_string(value):\n    return value[::-1]\n",
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


if __name__ == "__main__":
    unittest.main()
