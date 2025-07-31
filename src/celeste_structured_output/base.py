from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional

from pydantic import BaseModel

from .core.types import AIUsage, StructuredResponse


class BaseStructuredClient(ABC):
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the client, loading credentials from the environment.
        StructuredOutputProvider-specific arguments can be passed via kwargs.
        """
        pass

    @abstractmethod
    async def generate_content(
        self,
        prompt: str,
        response_schema: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> StructuredResponse:
        """Generates a single response."""
        pass

    @abstractmethod
    async def stream_generate_content(
        self,
        prompt: str,
        response_schema: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StructuredResponse]:
        """Streams the response chunk by chunk."""
        pass

    @abstractmethod
    def format_usage(self, usage_data: Any) -> Optional[AIUsage]:
        """Convert provider-specific usage data to standardized AIUsage format."""
        pass
