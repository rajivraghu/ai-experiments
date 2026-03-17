import io
import mimetypes
import struct
from typing import Optional

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image


st.set_page_config(page_title="Gemini Media Generator", page_icon="🪄", layout="centered")


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
            output_image = Image.open(io.BytesIO(first_part.inline_data.data)).convert("RGB")
        elif chunk.text:
            text_chunks.append(chunk.text)

    return output_image, "".join(text_chunks).strip()


def parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    bits_per_sample = 16
    rate = 24000

    parts = mime_type.split(";")
    for part in parts:
        part = part.strip()
        if part.lower().startswith("rate="):
            try:
                rate = int(part.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        elif part.startswith("audio/L"):
            try:
                bits_per_sample = int(part.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    params = parse_audio_mime_type(mime_type)
    bits_per_sample = params["bits_per_sample"]
    sample_rate = params["rate"]

    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data


def generate_audio(
    api_key: str,
    script_text: str,
    voice_name: str,
    model: str,
) -> tuple[Optional[bytes], Optional[str], str]:
    client = genai.Client(api_key=api_key)

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=script_text)],
        )
    ]

    config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
    )

    audio_bytes = None
    audio_mime_type = None
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
            raw_data = first_part.inline_data.data
            raw_mime_type = first_part.inline_data.mime_type or "audio/wav"
            file_extension = mimetypes.guess_extension(raw_mime_type)
            if file_extension is None:
                audio_bytes = convert_to_wav(raw_data, raw_mime_type)
                audio_mime_type = "audio/wav"
            else:
                audio_bytes = raw_data
                audio_mime_type = raw_mime_type
        elif chunk.text:
            text_chunks.append(chunk.text)

    return audio_bytes, audio_mime_type, "".join(text_chunks).strip()


st.title("Gemini Media Generator")
st.write("Generate images from a reference image or generate audio from a script.")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your API key")
    mode = st.radio("Menu", ["Image", "Audio"])

if mode == "Image":
    st.subheader("Reference Image + Prompt → Generated Image")
    image_model = st.text_input("Image Model", value="gemini-3.1-flash-image-preview")
    uploaded_file = st.file_uploader(
        "Upload reference image", type=["png", "jpg", "jpeg", "webp"]
    )
    prompt = st.text_area(
        "Prompt",
        value='Add a text overlay "Dhurandhar 2 Movie Review" in dark yellow font.',
        height=120,
    )

    if st.button("Generate Image", type="primary"):
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
                        model=image_model.strip(),
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
else:
    st.subheader("Script → Generated Audio")
    audio_model = st.text_input("Audio Model", value="gemini-2.5-flash-preview-tts")
    voice_name = st.text_input("Voice Name", value="Zephyr")
    script_text = st.text_area("Script", height=220)

    if st.button("Generate Audio", type="primary"):
        if not api_key:
            st.error("Please enter your Gemini API key.")
        elif not script_text.strip():
            st.error("Please enter a script for speech generation.")
        else:
            with st.spinner("Generating audio..."):
                try:
                    audio_bytes, audio_mime_type, generated_text = generate_audio(
                        api_key=api_key,
                        script_text=script_text.strip(),
                        voice_name=voice_name.strip() or "Zephyr",
                        model=audio_model.strip(),
                    )
                except Exception as exc:
                    st.exception(exc)
                else:
                    if audio_bytes:
                        st.audio(audio_bytes, format=audio_mime_type or "audio/wav")
                        st.download_button(
                            "Download audio",
                            data=audio_bytes,
                            file_name="generated_audio.wav",
                            mime=audio_mime_type or "audio/wav",
                        )
                    else:
                        st.warning("Model response did not include audio.")

                    if generated_text:
                        st.markdown("**Model text response:**")
                        st.write(generated_text)
