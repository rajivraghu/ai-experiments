import io
import mimetypes
from typing import Optional

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image


st.set_page_config(page_title="Gemini Image Generator", page_icon="🖼️", layout="centered")


def generate_image(
    api_key: str,
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    model: str,
) -> tuple[Optional[Image.Image], str]:
    client = genai.Client(api_key=api_key)

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(mime_type=mime_type, data=image_bytes),
                types.Part.from_text(text=prompt),
            ],
        )
    ]

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        image_config=types.ImageConfig(image_size="1K"),
        response_modalities=["IMAGE", "TEXT"],
    )

    output_image = None
    text_chunks: list[str] = []

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        if not chunk.parts:
            continue

        first_part = chunk.parts[0]
        if first_part.inline_data and first_part.inline_data.data:
            data = first_part.inline_data.data
            output_image = Image.open(io.BytesIO(data)).convert("RGB")
        elif chunk.text:
            text_chunks.append(chunk.text)

    return output_image, "".join(text_chunks).strip()


st.title("Reference Image + Prompt → Generated Image")
st.write(
    "Upload a reference image, provide a prompt, and generate a new image with Gemini."
)

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your API key")
    model = st.text_input("Model", value="gemini-3.1-flash-image-preview")

uploaded_file = st.file_uploader(
    "Upload reference image", type=["png", "jpg", "jpeg", "webp"]
)
prompt = st.text_area(
    "Prompt",
    value='Add a text overlay "Dhurandhar 2 Movie Review" in dark yellow font.',
    height=120,
)

generate_clicked = st.button("Generate Image", type="primary")

if generate_clicked:
    if not api_key:
        st.error("Please enter your Gemini API key.")
    elif not uploaded_file:
        st.error("Please upload a reference image.")
    elif not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        image_bytes = uploaded_file.getvalue()
        guessed_mime = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0]
        mime_type = guessed_mime or "image/jpeg"

        with st.spinner("Generating image..."):
            try:
                generated_image, generated_text = generate_image(
                    api_key=api_key,
                    prompt=prompt.strip(),
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    model=model.strip(),
                )
            except Exception as exc:
                st.exception(exc)
            else:
                st.subheader("Generated Result")
                if generated_image:
                    st.image(generated_image, caption="Generated Image", use_container_width=True)
                else:
                    st.warning("Model response did not include an image.")

                if generated_text:
                    st.markdown("**Model text response:**")
                    st.write(generated_text)
