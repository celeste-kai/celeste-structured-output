from typing import Any, AsyncIterator, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel

from ..base import BaseStructuredClient
from ..core.config import GOOGLE_API_KEY
from ..core.enums import GoogleStructuredModel, StructuredOutputProvider
from ..core.types import AIUsage, StructuredResponse


class GoogleStructuredClient(BaseStructuredClient):
    def __init__(
        self, model: str = GoogleStructuredModel.FLASH_LITE, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = model

    def format_usage(self, usage_data: Any) -> Optional[AIUsage]:
        """Convert Gemini usage data to AIUsage."""
        if not usage_data:
            return None
        return AIUsage(
            input_tokens=getattr(usage_data, "prompt_token_count", 0),
            output_tokens=getattr(usage_data, "candidates_token_count", 0),
            total_tokens=getattr(usage_data, "total_token_count", 0),
        )

    async def generate_content(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        **kwargs: Any,
    ) -> StructuredResponse:
        config = kwargs.pop("config", {})

        config["response_mime_type"] = "application/json"
        config["response_schema"] = response_schema

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config),
        )

        # Extract usage information if available
        usage = None
        if hasattr(response, "usage_metadata"):
            usage = self.format_usage(response.usage_metadata)

        # Return parsed content if using response_schema, otherwise return text
        content = response.parsed

        return StructuredResponse(
            content=content,
            usage=usage,
            provider=StructuredOutputProvider.GOOGLE,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        **kwargs: Any,
    ) -> AsyncIterator[StructuredResponse]:
        config = kwargs.pop("config", {})

        config["response_mime_type"] = "application/json"
        config["response_schema"] = response_schema

        last_usage_metadata = None
        has_yielded_content = False

        async for chunk in self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config),
        ):
            # Track usage metadata
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                last_usage_metadata = chunk.usage_metadata

            # When using structured output, Google returns the parsed object directly
            if hasattr(chunk, "parsed") and chunk.parsed:
                has_yielded_content = True
                yield StructuredResponse(
                    content=chunk.parsed,
                    provider=StructuredOutputProvider.GOOGLE,
                    metadata={"model": self.model_name, "is_stream_chunk": True},
                )

        # Yield final usage information if we have it and content was streamed
        if last_usage_metadata and has_yielded_content:
            yield StructuredResponse(
                content=None,  # No content in final usage response
                usage=self.format_usage(last_usage_metadata),
                provider=StructuredOutputProvider.GOOGLE,
                metadata={"model": self.model_name, "is_final_usage": True},
            )
