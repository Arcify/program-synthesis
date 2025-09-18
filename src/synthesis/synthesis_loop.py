"""Core orchestration logic for the synthesis pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .evaluation import evaluate_candidate
from .llm_interface import LLMClient
from .plugins.python_executor import PythonExecutor
from .prompts import build_reflection_prompt, build_synthesis_prompt
from .tasks import CandidateProgram, EvaluationOutcome, SynthesisState, TaskSpecification
from .utils import format_score


@dataclass
class SynthesisConfig:
    max_iterations: int = 5
    score_threshold: float = 0.99


class SynthesisPipeline:
    """Coordinates LLM prompting, execution, and feedback loops."""

    def __init__(
        self,
        llm: LLMClient,
        config: Optional[SynthesisConfig] = None,
        executor: Optional[PythonExecutor] = None,
    ) -> None:
        self.llm = llm
        self.config = config or SynthesisConfig()
        self.executor = executor or PythonExecutor()
        self._last_state: Optional[SynthesisState] = None

    def run(self, task: TaskSpecification) -> EvaluationOutcome:
        state = SynthesisState(task=task)
        self._last_state = state
        guidance_override: Optional[str] = None

        best_outcome: Optional[EvaluationOutcome] = None

        for attempt in range(1, self.config.max_iterations + 1):
            prompt = build_synthesis_prompt(task, state, guidance_override)
            generation = self.llm.generate(prompt, language=task.language)
            code = self._extract_code(generation.text)
            candidate = CandidateProgram(
                code=code,
                language=task.language,
                attempt=attempt,
                origin=generation.meta.get("source", "llm"),
            )

            outcome = evaluate_candidate(candidate, task, executor=self.executor)
            state.record(candidate, outcome)
            best_outcome = self._select_best(best_outcome, outcome)

            if outcome.passed or outcome.score >= self.config.score_threshold:
                break

            reflection_prompt = build_reflection_prompt(task, candidate, outcome)
            reflection = self.llm.generate(reflection_prompt)
            guidance_override = reflection.text.strip() or guidance_override

        assert best_outcome is not None  # max_iterations >= 1
        return best_outcome

    # ------------------------------------------------------------------
    @property
    def last_state(self) -> Optional[SynthesisState]:
        return self._last_state

    def _extract_code(self, text: str) -> str:
        """Strip Markdown fences and leading/trailing whitespace."""

        fence = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)
        match = fence.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _select_best(
        self,
        current: Optional[EvaluationOutcome],
        candidate: EvaluationOutcome,
    ) -> EvaluationOutcome:
        if current is None:
            return candidate
        if candidate.score > current.score:
            return candidate
        return current

    def summarize(self, state: Optional[SynthesisState] = None) -> str:
        target = state or self._last_state
        if target is None:
            return "<no synthesis has been run>"
        lines = []
        for entry in target.history:
            lines.append(
                f"attempt {entry['attempt']}: {format_score(entry['passed'], entry['score'])}"
            )
        return "\n".join(lines)
