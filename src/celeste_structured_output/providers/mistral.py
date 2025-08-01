from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

from mistralai import Mistral
from mistralai.extra.utils.response_format import pydantic_model_from_json
from pydantic import BaseModel

from ..base import BaseStructuredClient
from ..core.config import MISTRAL_API_KEY
from ..core.enums import (
    MistralStructuredModel as MistralModel,
)
from ..core.enums import (
    StructuredOutputProvider,
)
from ..core.types import AIUsage, StructuredResponse


class MistralStructuredClient(BaseStructuredClient):
    def __init__(
        self, model: str | MistralModel = MistralModel.SMALL_LATEST, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.model_name = model.value if isinstance(model, MistralModel) else model

    def format_usage(self, usage_data: Any) -> Optional[AIUsage]:
        """Convert Mistral usage data to AIUsage."""
        if not usage_data:
            return None
        return AIUsage(
            input_tokens=getattr(usage_data, "prompt_tokens", 0),
            output_tokens=getattr(usage_data, "completion_tokens", 0),
            total_tokens=getattr(usage_data, "total_tokens", 0),
        )

    async def generate_content(
        self, prompt: str, response_schema: type[BaseModel], **kwargs: Any
    ) -> StructuredResponse:
        response = await self.client.chat.parse_async(
            response_format=response_schema,
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        usage = self.format_usage(getattr(response, "usage", None))
        content = None
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.parsed

        return StructuredResponse(
            content=content,
            usage=usage,
            provider=StructuredOutputProvider.MISTRAL,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(
        self, prompt: str, response_schema: type[BaseModel], **kwargs: Any
    ) -> AsyncIterator[StructuredResponse]:
        stream = await self.client.chat.parse_stream_async(
            response_format=response_schema,
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        buffer = ""
        usage_data = None
        async for chunk in stream:
            if chunk.data.choices and chunk.data.choices[0].delta.content:
                buffer += chunk.data.choices[0].delta.content
            if chunk.data.usage:
                usage_data = chunk.data.usage

        if buffer:
            content = pydantic_model_from_json(
                json.loads(buffer),
                response_schema,
            )
            yield StructuredResponse(
                content=content,
                provider=StructuredOutputProvider.MISTRAL,
                metadata={"model": self.model_name, "is_stream_chunk": True},
            )

        if usage_data:
            yield StructuredResponse(
                content=None,
                usage=self.format_usage(usage_data),
                provider=StructuredOutputProvider.MISTRAL,
                metadata={"model": self.model_name, "is_final_usage": True},
            )
