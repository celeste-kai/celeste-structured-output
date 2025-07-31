"""
Core data types for agent communication.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from .enums import StructuredOutputProvider


class AIUsage(BaseModel):
    """Token usage metrics for AI responses."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int
    total_tokens: int


class StructuredResponse(BaseModel):
    """Response from AI providers."""

    model_config = ConfigDict(frozen=True)

    content: Union[BaseModel, List[BaseModel]]
    usage: Optional[AIUsage] = None
    provider: Optional[StructuredOutputProvider] = None
    metadata: Dict[str, Any] = {}
