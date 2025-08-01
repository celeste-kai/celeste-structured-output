from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Optional, get_args, get_origin

from huggingface_hub import InferenceClient
from pydantic import BaseModel

from ..base import BaseStructuredClient
from ..core.config import HUGGINGFACE_TOKEN
from ..core.enums import HuggingFaceModel, StructuredOutputProvider
from ..core.types import AIUsage, StructuredResponse


class HuggingFaceStructuredClient(BaseStructuredClient):
    def __init__(
        self, model: str | HuggingFaceModel = HuggingFaceModel.GEMMA_2_2B, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[misc, safe-super]
        self.model_name = model.value if isinstance(model, HuggingFaceModel) else model
        self.client = InferenceClient(model=self.model_name, token=HUGGINGFACE_TOKEN)

    def format_usage(self, usage_data: Any) -> Optional[AIUsage]:
        """Convert HuggingFace usage data to AIUsage."""
        if not usage_data:
            return None
        return AIUsage(
            input_tokens=getattr(usage_data, "prompt_tokens", 0),
            output_tokens=getattr(usage_data, "completion_tokens", 0),
            total_tokens=getattr(usage_data, "total_tokens", 0),
        )

    def _parse_content(self, data: Any, schema: BaseModel) -> Any:
        if get_origin(schema) is list:
            model = get_args(schema)[0]
            return [model.model_validate(item) for item in data]
        return schema.model_validate(data)

    async def generate_content(
        self, prompt: str, response_schema: BaseModel, **kwargs: Any
    ) -> StructuredResponse:
        messages = [{"role": "user", "content": prompt}]
        kwargs.setdefault("response_format", {"type": "json_object"})
        response = await asyncio.to_thread(
            self.client.chat_completion, messages=messages, **kwargs
        )
        usage = self.format_usage(getattr(response, "usage", None))
        data = json.loads(response.choices[0].message.content)
        content = self._parse_content(data, response_schema)
        return StructuredResponse(
            content=content,
            usage=usage,
            provider=StructuredOutputProvider.HUGGINGFACE,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(  # type: ignore[override, misc]
        self, prompt: str, response_schema: BaseModel, **kwargs: Any
    ) -> AsyncIterator[StructuredResponse]:
        messages = [{"role": "user", "content": prompt}]
        kwargs.setdefault("response_format", {"type": "json_object"})
        kwargs["stream"] = True
        stream = await asyncio.to_thread(
            self.client.chat_completion, messages=messages, **kwargs
        )
        buffer = ""
        usage_data = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                buffer += chunk.choices[0].delta.content
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = self.format_usage(chunk.usage)
        if buffer:
            data = json.loads(buffer)
            content = self._parse_content(data, response_schema)
            yield StructuredResponse(
                content=content,
                provider=StructuredOutputProvider.HUGGINGFACE,
                metadata={"model": self.model_name, "is_stream_chunk": True},
            )
        if usage_data:
            yield StructuredResponse(
                content=None,
                usage=usage_data,
                provider=StructuredOutputProvider.HUGGINGFACE,
                metadata={"model": self.model_name, "is_final_usage": True},
            )
