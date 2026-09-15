"""The backend contract.

Deliberately NOT upstream's ``opengrad.evaluation.runner.InferenceBackend``: that one passes a
CanonicalEvaluationExample to the backend, and the example carries ``expected_decision``. A backend
could then condition on gold -- in a cache key, a log line, or a helper someone adds later -- with
nothing preventing it and no test catching it. This shape takes only what a model may legitimately
see, so gold leakage is structurally impossible rather than merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"  # stop | length | error
    latency: float = 0.0
    backend_metadata: dict[str, Any] = field(default_factory=dict)


class InferenceBackend(Protocol):
    name: str

    def generate(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        generation_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationResult: ...
