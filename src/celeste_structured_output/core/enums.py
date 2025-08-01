"""
Core enumerations for Celeste AI Client.
"""

from enum import Enum


class StructuredOutputProvider(Enum):
    """AI provider enumeration for multi-provider agent support."""

    GOOGLE = "google"
    OPENAI = "openai"
    MISTRAL = "mistral"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class GoogleStructuredModel(Enum):
    """Google model enumeration for provider-specific model selection."""

    FLASH_LITE = "gemini-2.5-flash-lite-preview-06-17"
    FLASH = "gemini-2.5-flash"
    PRO = "gemini-2.5-pro"


class OpenAIStructuredModel(Enum):
    """OpenAI model enumeration for provider-specific model selection."""

    GPT_O3 = "o3-2025-04-16"
    GPT_O4_MINI = "o4-mini-2025-04-16"
    GPT_4_1 = "gpt-4.1-2025-04-14"


class MistralStructuredModel(Enum):
    """Mistral AI model enumeration for provider-specific model selection."""

    SMALL_LATEST = "mistral-small-latest"
    MEDIUM_LATEST = "mistral-medium-latest"
    LARGE_LATEST = "mistral-large-latest"
    CODESTRAL_LATEST = "codestral-latest"


class AnthropicStructuredModel(Enum):
    """Anthropic Claude model enumeration for provider-specific model selection."""

    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_4_SONNET = "claude-sonnet-4-20250514"
    CLAUDE_4_OPUS = "claude-opus-4-20250514"


class HuggingFaceModel(Enum):
    """Hugging Face model enumeration for provider-specific model selection."""

    GEMMA_2_2B = "google/gemma-2-2b-it"
    META_LLAMA_3_1_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MICROSOFT_PHI_4 = "microsoft/phi-4"
    QWEN_2_5_7B_1M = "Qwen/Qwen2.5-7B-Instruct-1M"
    DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1"
