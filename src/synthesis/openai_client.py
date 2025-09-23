"""OpenAI-backed LLM client for the synthesis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

from .llm_interface import LLMClient, LLMGeneration


@dataclass
class OpenAIConfig:
    """Configuration defaults for the OpenAI client."""

    model: str = "gpt-5"
    temperature: float = 0.2
    max_output_tokens: int = 1200
    system_prompt: Optional[str] = (
        "You are an expert software engineer. Produce only valid code without Markdown fences."
    )
    extra_options: Dict[str, Any] = field(default_factory=dict)


class OpenAIClient(LLMClient):
    """LLMClient adapter that calls OpenAI's Responses API (e.g. GPT-5)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        config: Optional[OpenAIConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        if client is None and OpenAI is None:
            raise ImportError(
                "openai package is required to use OpenAIClient. Install with `pip install openai`."
            )

        self.config = config or OpenAIConfig()
        client_options: Dict[str, Any] = {
            "api_key": api_key,
            "organization": organization,
            "project": project,
        }
        client_options = {k: v for k, v in client_options.items() if v is not None}
        self._client = client or OpenAI(**client_options)  # type: ignore[call-arg]

    def generate(self, prompt: str, **kwargs: Any) -> LLMGeneration:
        """Generate code using the configured OpenAI model."""

        temperature = kwargs.pop("temperature", self.config.temperature)
        max_output_tokens = kwargs.pop("max_output_tokens", self.config.max_output_tokens)
        system_prompt = kwargs.pop("system_prompt", self.config.system_prompt)

        # Merge additional options with per-call kwargs taking precedence.
        options: Dict[str, Any] = {**self.config.extra_options, **kwargs}
        options.pop("language", None)
        if temperature is not None and "temperature" not in options:
            options["temperature"] = temperature
        if max_output_tokens is not None and "max_output_tokens" not in options:
            options["max_output_tokens"] = max_output_tokens

        if self.config.model.startswith("gpt-5"):
            options.pop("temperature", None)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.responses.create(
            model=self.config.model,
            input=messages,
            **options,
        )

        try:
            text = response.output_text  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - API fallback
            text = "".join(
                chunk.content[0].text
                for chunk in getattr(response, "output", [])
                if getattr(chunk, "content", [])
            )

        return LLMGeneration(
            prompt=prompt,
            text=text.strip(),
            meta={
                "model": self.config.model,
                "tokens": getattr(response, "usage", {}),
            },
        )
