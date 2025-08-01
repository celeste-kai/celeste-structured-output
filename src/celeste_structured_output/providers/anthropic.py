from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from pydantic import BaseModel

from ..base import BaseStructuredClient
from ..core.config import ANTHROPIC_API_KEY
from ..core.enums import AnthropicStructuredModel as AnthropicModel
from ..core.enums import StructuredOutputProvider
from ..core.types import AIUsage, StructuredResponse

MAX_TOKENS = 1024


class AnthropicStructuredClient(BaseStructuredClient):
    def __init__(
        self,
        model: str | AnthropicModel = AnthropicModel.CLAUDE_3_7_SONNET,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = model.value if isinstance(model, AnthropicModel) else model

    def format_usage(self, usage_data: Any) -> Optional[AIUsage]:
        """Convert Anthropic usage data to AIUsage."""
        if not usage_data:
            return None
        return AIUsage(
            input_tokens=usage_data.input_tokens,
            output_tokens=usage_data.output_tokens,
            total_tokens=usage_data.input_tokens + usage_data.output_tokens,
        )

    async def generate_content(
        self, prompt: str, response_schema: BaseModel, **kwargs: Any
    ) -> StructuredResponse:
        max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
        tools = [
            {
                "name": "structured_output",
                "description": "Return a JSON object matching the provided schema",
                "input_schema": response_schema.model_json_schema(),
            }
        ]

        response = await self.client.messages.create(
            max_tokens=max_tokens,
            messages=[MessageParam(role="user", content=prompt)],
            model=self.model_name,
            tools=tools,
            tool_choice="auto",
            **kwargs,
        )

        tool_use = next(
            (b.input for b in response.content if getattr(b, "type", "") == "tool_use"),
            None,
        )
        content = response_schema.model_validate(tool_use or {})

        return StructuredResponse(
            content=content,
            usage=self.format_usage(response.usage),
            provider=StructuredOutputProvider.ANTHROPIC,
            metadata={"model": self.model_name},
        )

    async def stream_generate_content(
        self, prompt: str, response_schema: BaseModel, **kwargs: Any
    ) -> AsyncIterator[StructuredResponse]:
        max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
        tools = [
            {
                "name": "structured_output",
                "description": "Return a JSON object matching the provided schema",
                "input_schema": response_schema.model_json_schema(),
            }
        ]

        async with self.client.messages.stream(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[MessageParam(role="user", content=prompt)],
            tools=tools,
            tool_choice="auto",
            **kwargs,
        ) as stream:
            snapshot: dict[str, Any] | None = None
            async for event in stream:
                if getattr(event, "type", "") == "input_json":
                    snapshot = event.snapshot
                    yield StructuredResponse(
                        content=response_schema.model_validate(snapshot),
                        provider=StructuredOutputProvider.ANTHROPIC,
                        metadata={"model": self.model_name, "is_stream_chunk": True},
                    )

            final_message = await stream.get_final_message()
            usage = self.format_usage(final_message.usage if final_message else None)
            if usage:
                yield StructuredResponse(
                    content=None,
                    usage=usage,
                    provider=StructuredOutputProvider.ANTHROPIC,
                    metadata={"model": self.model_name, "is_final_usage": True},
                )
