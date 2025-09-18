"""Program synthesis pipeline package."""

from .evaluation import evaluate_candidate
from .llm_interface import LLMClient, LLMGeneration, MockLLM
from .plugins.python_executor import PythonExecutor
from .synthesis_loop import SynthesisConfig, SynthesisPipeline
from .tasks import (
    CandidateProgram,
    EvaluationOutcome,
    IOExample,
    SynthesisState,
    TaskSpecification,
)

__all__ = [
    "LLMClient",
    "LLMGeneration",
    "MockLLM",
    "PythonExecutor",
    "SynthesisConfig",
    "SynthesisPipeline",
    "evaluate_candidate",
    "CandidateProgram",
    "EvaluationOutcome",
    "IOExample",
    "SynthesisState",
    "TaskSpecification",
]
