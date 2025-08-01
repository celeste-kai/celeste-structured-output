# mypy: ignore-errors
import asyncio
from datetime import datetime
from typing import cast

import streamlit as st
from pydantic import BaseModel, create_model

from src.celeste_structured_output import (
    StructuredOutputProvider,
    create_structured_client,
)
from src.celeste_structured_output.core.enums import (
    AnthropicStructuredModel,
    GoogleStructuredModel,
    HuggingFaceModel,
    MistralStructuredModel,
    OpenAIStructuredModel,
)

st.title("Celeste Structured output")

PROVIDER_MODEL_MAP = {
    StructuredOutputProvider.GOOGLE.name: GoogleStructuredModel,
    StructuredOutputProvider.OPENAI.name: OpenAIStructuredModel,
    StructuredOutputProvider.MISTRAL.name: MistralStructuredModel,
    StructuredOutputProvider.ANTHROPIC.name: AnthropicStructuredModel,
    StructuredOutputProvider.HUGGINGFACE.name: HuggingFaceModel,
}

with st.sidebar:
    structured_output_provider = st.selectbox(
        "Select provider",
        [p.name for p in list(StructuredOutputProvider)],
        format_func=lambda x: StructuredOutputProvider[x].name,
        key="provider",
    )

    structured_output_model = st.selectbox(
        "Select model",
        [m.name for m in PROVIDER_MODEL_MAP[structured_output_provider]],
        key="model",
    )

# Initialize session state for properties
if "properties" not in st.session_state:
    st.session_state.properties = []

# Structure Builder
st.subheader("Build Your Structure")

# Structure name
structure_name = st.text_input("Structure name", value="MyModel")

# Return type selection
return_type = st.radio("Return type", ["Single object", "List of objects"])

# Add property button
if st.button("Add Property"):
    st.session_state.properties.append({"name": "", "type": "str"})

# Display properties
for i, _prop in enumerate(st.session_state.properties):
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.properties[i]["name"] = st.text_input(
            "Property name",
            key=f"name_{i}",
        )
    with col2:
        st.session_state.properties[i]["type"] = st.selectbox(
            "Type",
            ["str", "int", "datetime"],
            key=f"type_{i}",
        )

model_enum = PROVIDER_MODEL_MAP[structured_output_provider]
client = create_structured_client(
    provider=StructuredOutputProvider[structured_output_provider].value,
    model=model_enum[structured_output_model].value,
)

prompt = st.text_input("Prompt", value=f"Generate a sample {structure_name}")

if st.button("Generate"):
    # Create dynamic model from properties

    fields = {}
    for prop in st.session_state.properties:
        if prop["name"]:
            field_type = {"str": str, "int": int, "datetime": datetime}[prop["type"]]
            fields[prop["name"]] = (field_type, ...)
    DynamicModel = cast(type[BaseModel], create_model(structure_name, **fields))

    # Use list schema if requested

    response_schema = (
        list[DynamicModel] if return_type == "List of objects" else DynamicModel
    )

    with st.spinner("Generating..."):
        output = asyncio.run(
            client.generate_content(
                prompt=prompt,
                response_schema=response_schema,
            )
        )
        if output:
            st.json(output.content)
