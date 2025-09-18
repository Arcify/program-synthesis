"""Interfaces and mock implementations for LLM interactions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class LLMGeneration:
    """Represents a single completion from an LLM."""

    prompt: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> LLMGeneration:
        """Generate a single completion for the given prompt."""

    def bulk_generate(
        self, prompts: Sequence[str], **kwargs: Any
    ) -> List[LLMGeneration]:
        """Generate completions for multiple prompts."""

        return [self.generate(prompt, **kwargs) for prompt in prompts]


class MockLLM(LLMClient):
    """Lightweight mock LLM for local development and tests."""

    def __init__(
        self,
        canned_responses: Optional[Dict[str, str]] = None,
        default_language: str = "python",
    ) -> None:
        self._canned_responses = canned_responses or {}
        self._default_language = default_language

    def generate(self, prompt: str, **kwargs: Any) -> LLMGeneration:
        """Return canned responses or synthesize a naive program stub."""

        for key, response in self._canned_responses.items():
            if key.lower() in prompt.lower():
                return LLMGeneration(prompt=prompt, text=response, meta={"source": "canned"})

        language = kwargs.get("language", self._default_language)
        program = self._synthesize_stub(prompt, language)
        return LLMGeneration(prompt=prompt, text=program, meta={"source": "stub"})

    # ------------------------------------------------------------------
    def _synthesize_stub(self, prompt: str, language: str) -> str:
        """Produce simple deterministic code snippets for demo purposes."""

        if language.lower() == "python":
            return self._python_stub(prompt)
        return f"// TODO: implement solution\n"  # Fallback for other languages

    def _python_stub(self, prompt: str) -> str:
        templates: Iterable[str] = (
            "def synthesized_function(*args, **kwargs):\n    \"\"\"TODO: implement based on prompt.\"\"\"\n    return None\n",
            "def solve(input_value):\n    # Placeholder solution produced by MockLLM\n    return input_value\n",
            "def run(data):\n    total = 0\n    for value in data:\n        total += value\n    return total\n",
        )
        # Pick a template deterministically to keep tests stable
        template_list = list(templates)
        seed = sum(ord(ch) for ch in prompt) % len(template_list)
        return template_list[seed]
