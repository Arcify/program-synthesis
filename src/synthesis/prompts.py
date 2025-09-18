"""Prompt builders for the program synthesis pipeline."""

from __future__ import annotations

from textwrap import dedent
from typing import Optional

from .tasks import CandidateProgram, EvaluationOutcome, SynthesisState, TaskSpecification


def build_synthesis_prompt(
    task: TaskSpecification,
    state: SynthesisState,
    system_guidelines: Optional[str] = None,
) -> str:
    """Create the prompt used to request a new candidate program from the LLM."""

    guidance = system_guidelines or (
        "You write clean, testable and minimal code."
        " Always return only the program text."
        " Avoid Markdown fence syntax."
    )

    history_fragment = ""
    if state.history:
        last = state.history[-1]
        history_fragment = dedent(
            f"""
            Previous attempt #{last['attempt']} summary:
            - Best score so far: {state.best_score():.2f}
            - Latest feedback: {last['feedback']}
            - Passed: {last['passed']}
            """
        ).strip()

    prompt = dedent(
        f"""
        System guidelines:
        {guidance}

        Task [{task.task_id}] ({task.language}):
        {task.description}

        Entry point: {task.entry_point}
        Constraints: {task.constraints or 'None provided.'}

        Reference IO examples:
        {task.format_examples()}
        """
    ).strip()

    if history_fragment:
        prompt = f"{prompt}\n\n{history_fragment}\n"

    prompt += "\nSynthesize the full program now."
    return prompt


def build_reflection_prompt(
    task: TaskSpecification,
    candidate: CandidateProgram,
    outcome: EvaluationOutcome,
) -> str:
    """Prompt the LLM to reflect on a failed attempt and plan the next one."""

    status = "passed" if outcome.passed else "failed"
    prompt = dedent(
        f"""
        The previous attempt ({status}) with score {outcome.score:.2f} produced:

        ---
        {candidate.code}
        ---

        Feedback:
        {outcome.feedback}

        Provide concise guidance for improving the next attempt. Focus on the root cause of failure and actionable fixes.
        """
    ).strip()
    return prompt
