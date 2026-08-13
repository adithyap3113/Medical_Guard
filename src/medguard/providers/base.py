"""Vendor-neutral LLM provider contract.

This module defines what MedGuard needs from a text-generation provider, expressed
without reference to any particular vendor. It imports nothing from Google.

The rest of the application depends on :class:`LLMProvider`, never on a concrete
implementation. That is what makes the generation provider swappable, and what lets
tests inject a fake provider that returns canned text without any network access --
which in turn is what makes the Phase 12 evaluation harness runnable offline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["LLMProvider", "LLMResponse", "ProviderHealth"]


class LLMResponse(BaseModel):
    """One completion, plus the metadata needed for cost and latency accounting.

    Token counts are optional because not every provider reports them and a response
    is still usable without them. They are recorded when available so that Phase 13
    can attribute cost per query instead of estimating it.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    model: str = Field(description="The model that actually served the request.")
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    finish_reason: str | None = Field(
        default=None, description="Normalised to a plain string, never a vendor enum."
    )
    truncated: bool = Field(
        default=False,
        description="True when generation stopped at the token limit. Worth surfacing: "
        "a truncated answer can drop its citations and look unsupported.",
    )


class ProviderHealth(BaseModel):
    """Result of a provider diagnostic check, safe to render in ``medguard doctor``.

    Deliberately separates three states that are easy to conflate:

    * ``configured`` -- a key is present (says nothing about whether it works);
    * ``reachable`` -- the API answered us;
    * ``model_available`` -- *this* key can actually use the configured model.

    The third is the one that cannot be assumed. A model being free-tier eligible in
    the documentation does not mean it is enabled for a given API project, so the
    only honest way to report it is to ask the API and record the answer.
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    configured: bool
    reachable: bool = False
    model: str | None = None
    model_available: bool | None = Field(
        default=None, description="None means 'not checked', not 'available'."
    )
    available_models: list[str] = Field(default_factory=list)
    latency_ms: float | None = Field(default=None, ge=0)
    error: str | None = Field(default=None, description="Redacted, human-readable failure.")

    @property
    def ok(self) -> bool:
        return self.configured and self.reachable and self.model_available is not False


@runtime_checkable
class LLMProvider(Protocol):
    """The generation interface the application depends on.

    A ``Protocol`` rather than an abstract base class: implementations do not need to
    inherit from anything, so a test double is a plain object with the right methods.
    """

    name: str

    def generate(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a completion.

        Raises a :class:`medguard.errors.ProviderError` subclass on failure -- never a
        vendor exception.
        """
        ...

    def health(self, *, smoke_test: bool = False) -> ProviderHealth:
        """Check configuration and connectivity.

        Must not raise: diagnostics report failure in the returned object, because a
        doctor command that crashes on the first broken check cannot tell you about
        the rest of the system.
        """
        ...

    def list_models(self) -> list[str]:
        """Return model ids usable for text generation with the current credentials."""
        ...
