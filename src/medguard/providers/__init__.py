"""LLM provider adapters.

Import :mod:`medguard.providers.base` for the contract. Concrete adapters are imported
explicitly by the composition root, not re-exported here, so that importing this
package never pulls a vendor SDK into memory.
"""

from __future__ import annotations

from medguard.providers.base import LLMProvider, LLMResponse, ProviderHealth

__all__ = ["LLMProvider", "LLMResponse", "ProviderHealth"]
