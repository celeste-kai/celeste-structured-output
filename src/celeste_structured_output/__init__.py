"""
Celeste AI Client - Minimal predefinition AI communication for Alita agents.
"""

from typing import Any, Union

from .base import BaseStructuredClient
from .core import StructuredOutputProvider, StructuredResponse

__version__ = "0.1.0"


def create_structured_client(
    provider: Union[StructuredOutputProvider, str], **kwargs: Any
) -> BaseStructuredClient:
    if isinstance(provider, str):
        provider = StructuredOutputProvider(provider)

    if provider == StructuredOutputProvider.GOOGLE:
        from .providers.google import GoogleStructuredClient

        return GoogleStructuredClient(**kwargs)

    if provider == StructuredOutputProvider.OPENAI:
        from .providers.openai import OpenAIClient

        return OpenAIClient(**kwargs)

    if provider == StructuredOutputProvider.HUGGINGFACE:
        from .providers.huggingface import HuggingFaceStructuredClient

        return HuggingFaceStructuredClient(**kwargs)

    if provider == StructuredOutputProvider.MISTRAL:
        from .providers.mistral import MistralStructuredClient

        return MistralStructuredClient(**kwargs)

    if provider == StructuredOutputProvider.ANTHROPIC:
        from .providers.anthropic import AnthropicStructuredClient

        return AnthropicStructuredClient(**kwargs)

    raise ValueError(f"StructuredOutputProvider {provider} not implemented")


__all__ = [
    "create_structured_client",
    "BaseStructuredClient",
    "StructuredOutputProvider",
    "StructuredResponse",
]
