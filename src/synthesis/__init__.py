"""Program synthesis pipeline package."""

from .evaluation import evaluate_candidate
from .llm_interface import LLMClient, LLMGeneration, MockLLM
from .openai_client import OpenAIClient, OpenAIConfig
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
    "OpenAIClient",
    "OpenAIConfig",
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
