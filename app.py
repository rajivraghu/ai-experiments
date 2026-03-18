import io
import json
import mimetypes
import struct
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests
import streamlit as st
from google import genai
from google.genai import types
from moviepy import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips
from PIL import Image, ImageOps


SEARCHAPI_URL = "https://www.searchapi.io/api/v1/search"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_AUDIO_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_VOICE = "Zephyr"
VIDEO_SIZE = (1280, 720)
MIN_IMAGE_DIMENSION = 1280
IMAGE_SEARCH_RESULT_LIMIT = 10

SAMPLE_JSON = json.dumps(
    [
        {
            "title": "గేమ్ ఛేంజర్: రామ్ చరణ్ మెగా మాస్ టీజర్ విడుదల",
            "summary": "మెగా పవర్ స్టార్ రామ్ చరణ్ మరియు శంకర్ కాంబినేషన్‌లో వస్తున్న 'గేమ్ ఛేంజర్' టీజర్ ఎట్టకేలకు విడుదలైంది. ఇందులో చరణ్ రెండు విభిన్న పాత్రల్లో కనిపిస్తూ అభిమానులను ఆకట్టుకుంటున్నారు.",
            "image_search_keywords": "Ram Charan Game Changer movie teaser HD poster",
            "article_url": "https://www.123telugu.com/mnews/ram-charan-shankar-game-changer-teaser-out.html",
        },
        {
            "title": "పుష్ప 2: 'ది రూల్' ట్రైలర్ సెన్సేషన్",
            "summary": "అల్లు అర్జున్ 'పుష్ప 2: ది రూల్' ట్రైలర్ ప్రపంచవ్యాప్తంగా ప్రకంపనలు సృష్టిస్తోంది. సుకుమార్ దర్శకత్వంలో రూపొందిన ఈ విజువల్ వండర్ పుష్పరాజ్ ప్రపంచాన్ని మరింత పెద్దగా చూపిస్తోంది.",
            "image_search_keywords": "Allu Arjun Pushpa 2 The Rule trailer launch HD image",
            "article_url": "https://www.telugucinema.com/news/pushpa-2-trailer-release-date-updates",
        },
    ],
    ensure_ascii=False,
    indent=2,
)


@dataclass
class NewsItem:
    title: str
    summary: str
    image_search_keywords: str
    article_url: str


@dataclass
class DownloadedImage:
    data: bytes
    mime_type: str
    width: int
    height: int
    source_url: str
    source_page: str


@dataclass
class StoryResult:
    item: NewsItem
    source_image: DownloadedImage
    generated_image_bytes: bytes
    generated_audio_bytes: bytes
    generated_audio_mime: str
    clip_bytes: bytes
    clip_filename: str
    generated_text: str
    tts_text: str



def get_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def generate_image(
    api_key: str,
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    model: str,
) -> tuple[Optional[bytes], str]:
    client = get_gemini_client(api_key)
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

    output_image_bytes = None
    text_chunks: list[str] = []
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        if not chunk.parts:
            continue
        first_part = chunk.parts[0]
        if first_part.inline_data and first_part.inline_data.data:
            output_image_bytes = first_part.inline_data.data
        elif chunk.text:
            text_chunks.append(chunk.text)

    return output_image_bytes, "".join(text_chunks).strip()


def parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    bits_per_sample = 16
    rate = 24000

    for part in mime_type.split(";"):
        cleaned = part.strip()
        if cleaned.lower().startswith("rate="):
            try:
                rate = int(cleaned.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        elif cleaned.startswith("audio/L"):
            try:
                bits_per_sample = int(cleaned.split("L", 1)[1])
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
    client = get_gemini_client(api_key)
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=script_text)])]
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
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
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


def parse_json_items(raw_json: str) -> list[NewsItem]:
    parsed = json.loads(raw_json)
    if not isinstance(parsed, list):
        raise ValueError("The JSON input must be a list of objects.")

    items: list[NewsItem] = []
    required_keys = {"title", "summary", "image_search_keywords", "article_url"}
    for index, entry in enumerate(parsed, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Item {index} must be a JSON object.")
        missing = [key for key in required_keys if not str(entry.get(key, "")).strip()]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Item {index} is missing required fields: {joined}")
        items.append(
            NewsItem(
                title=str(entry["title"]).strip(),
                summary=str(entry["summary"]).strip(),
                image_search_keywords=str(entry["image_search_keywords"]).strip(),
                article_url=str(entry["article_url"]).strip(),
            )
        )
    return items


def search_images(searchapi_key: str, query: str) -> list[dict[str, Any]]:
    params = {
        "engine": "google_images",
        "q": query,
        "api_key": searchapi_key,
        "num": IMAGE_SEARCH_RESULT_LIMIT,
    }
    response = requests.get(SEARCHAPI_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    images = payload.get("images", [])
    if not isinstance(images, list):
        return []
    return images


def normalize_image_bytes(image_bytes: bytes, fallback_format: str = "JPEG") -> tuple[bytes, str, int, int]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    buffer = io.BytesIO()
    image.save(buffer, format=fallback_format, quality=95)
    mime_type = "image/jpeg" if fallback_format.upper() == "JPEG" else f"image/{fallback_format.lower()}"
    return buffer.getvalue(), mime_type, width, height


def download_best_image(searchapi_key: str, query: str) -> DownloadedImage:
    results = search_images(searchapi_key, query)
    candidate_errors: list[str] = []

    for result in results:
        original = result.get("original") or {}
        image_url = str(original.get("link") or "").strip()
        width = int(original.get("width") or 0)
        height = int(original.get("height") or 0)
        if not image_url or max(width, height) < MIN_IMAGE_DIMENSION:
            continue

        try:
            response = requests.get(
                image_url,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            mime_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
            normalized_bytes, normalized_mime, actual_width, actual_height = normalize_image_bytes(
                response.content
            )
        except Exception as exc:
            candidate_errors.append(f"{image_url}: {exc}")
            continue

        if max(actual_width, actual_height) < MIN_IMAGE_DIMENSION:
            candidate_errors.append(
                f"{image_url}: downloaded image was only {actual_width}x{actual_height}"
            )
            continue

        return DownloadedImage(
            data=normalized_bytes,
            mime_type=normalized_mime or mime_type,
            width=actual_width,
            height=actual_height,
            source_url=image_url,
            source_page=str((result.get("source") or {}).get("link") or ""),
        )

    details = "\n".join(candidate_errors[:5])
    raise ValueError(
        "No downloadable image with a dimension over 1280 pixels was found for the search keywords."
        + (f"\n{details}" if details else "")
    )


def build_overlay_prompt(title: str) -> str:
    return textwrap.dedent(
        f"""
        Using the provided reference image, create a polished entertainment-news visual.
        Add the exact title text in Telugu as a bold readable headline:
        {title}

        Keep the subject recognizable, use strong contrast, and make the title look like a premium movie-news thumbnail.
        """
    ).strip()


def create_cover_frame(image_bytes: bytes, output_path: Path) -> None:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    canvas = Image.new("RGB", VIDEO_SIZE, color=(12, 12, 12))
    fitted = ImageOps.contain(image, VIDEO_SIZE)
    offset_x = (VIDEO_SIZE[0] - fitted.width) // 2
    offset_y = (VIDEO_SIZE[1] - fitted.height) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    canvas.save(output_path, format="PNG")


def set_clip_duration(clip: ImageClip, duration: float) -> ImageClip:
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    return clip.set_duration(duration)


def set_clip_audio(clip: ImageClip, audio_clip: AudioFileClip) -> ImageClip:
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio_clip)
    return clip.set_audio(audio_clip)


def create_video_from_image_and_audio(
    image_bytes: bytes,
    audio_bytes: bytes,
    audio_mime_type: str,
    video_format: str,
    output_path: Path,
) -> None:
    workspace = output_path.parent
    frame_path = workspace / "frame.png"
    create_cover_frame(image_bytes, frame_path)

    audio_extension = mimetypes.guess_extension(audio_mime_type or "audio/wav") or ".wav"
    audio_path = workspace / f"track{audio_extension}"
    audio_path.write_bytes(audio_bytes)

    audio_clip = AudioFileClip(str(audio_path))
    image_clip = ImageClip(str(frame_path))
    video_clip = set_clip_audio(set_clip_duration(image_clip, audio_clip.duration), audio_clip)

    try:
        if video_format == "webm":
            video_clip.write_videofile(
                str(output_path),
                fps=24,
                codec="libvpx-vp9",
                audio_codec="libvorbis",
                logger=None,
            )
        else:
            video_clip.write_videofile(
                str(output_path),
                fps=24,
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
    finally:
        video_clip.close()
        image_clip.close()
        audio_clip.close()


def concatenate_story_videos(video_paths: list[Path], video_format: str, output_path: Path) -> None:
    video_clips = [VideoFileClip(str(path)) for path in video_paths]
    final_clip = concatenate_videoclips(video_clips, method="compose")
    try:
        if video_format == "webm":
            final_clip.write_videofile(
                str(output_path),
                fps=24,
                codec="libvpx-vp9",
                audio_codec="libvorbis",
                logger=None,
            )
        else:
            final_clip.write_videofile(
                str(output_path),
                fps=24,
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
    finally:
        final_clip.close()
        for clip in video_clips:
            clip.close()


def process_story_items(
    items: list[NewsItem],
    gemini_api_key: str,
    searchapi_key: str,
    image_model: str,
    audio_model: str,
    voice_name: str,
    video_format: str,
    status_placeholder: Any,
    progress_bar: Any,
) -> tuple[list[StoryResult], bytes, str]:
    results: list[StoryResult] = []
    generated_clip_paths: list[Path] = []

    with tempfile.TemporaryDirectory() as workspace_dir:
        workspace = Path(workspace_dir)
        total_steps = max(len(items) * 4 + 1, 1)
        completed_steps = 0

        def advance(message: str) -> None:
            nonlocal completed_steps
            completed_steps += 1
            progress_bar.progress(min(completed_steps / total_steps, 1.0), text=message)
            status_placeholder.info(message)

        for index, item in enumerate(items, start=1):
            advance(f"[{index}/{len(items)}] Searching and downloading image for: {item.title}")
            source_image = download_best_image(searchapi_key, item.image_search_keywords)

            advance(f"[{index}/{len(items)}] Generating Gemini overlay image for: {item.title}")
            generated_image_bytes, generated_text = generate_image(
                api_key=gemini_api_key,
                prompt=build_overlay_prompt(item.title),
                image_bytes=source_image.data,
                mime_type=source_image.mime_type,
                model=image_model,
            )
            if not generated_image_bytes:
                raise ValueError(f"Gemini did not return an image for item {index}: {item.title}")
            generated_image_bytes, _, _, _ = normalize_image_bytes(generated_image_bytes)

            advance(f"[{index}/{len(items)}] Generating Telugu TTS audio for: {item.title}")
            generated_audio_bytes, generated_audio_mime, tts_text = generate_audio(
                api_key=gemini_api_key,
                script_text=item.summary,
                voice_name=voice_name,
                model=audio_model,
            )
            if not generated_audio_bytes:
                raise ValueError(f"Gemini did not return audio for item {index}: {item.title}")

            advance(f"[{index}/{len(items)}] Rendering video clip for: {item.title}")
            clip_filename = f"story_{index:02d}.{video_format}"
            clip_path = workspace / clip_filename
            create_video_from_image_and_audio(
                image_bytes=generated_image_bytes,
                audio_bytes=generated_audio_bytes,
                audio_mime_type=generated_audio_mime or "audio/wav",
                video_format=video_format,
                output_path=clip_path,
            )
            generated_clip_paths.append(clip_path)

            results.append(
                StoryResult(
                    item=item,
                    source_image=source_image,
                    generated_image_bytes=generated_image_bytes,
                    generated_audio_bytes=generated_audio_bytes,
                    generated_audio_mime=generated_audio_mime or "audio/wav",
                    clip_bytes=clip_path.read_bytes(),
                    clip_filename=clip_filename,
                    generated_text=generated_text,
                    tts_text=tts_text,
                )
            )

        advance("Combining all clips into one final video")
        final_filename = f"combined_news_reel.{video_format}"
        final_path = workspace / final_filename
        concatenate_story_videos(generated_clip_paths, video_format, final_path)
        final_video_bytes = final_path.read_bytes()

    progress_bar.progress(1.0, text="Done")
    status_placeholder.success("Processing completed successfully.")
    return results, final_video_bytes, final_filename


def render_app() -> None:
    st.set_page_config(page_title="Gemini Media Studio", page_icon="🎬", layout="wide")
    st.title("Gemini Media Studio")
    st.write(
        "Build image, audio, or full JSON-driven news videos using Gemini + SearchApi image search."
    )

    with st.sidebar:
        st.header("Configuration")
        gemini_api_key = st.text_input("Gemini API Key", type="password")
        searchapi_key = st.text_input("SearchApi.io Key", type="password")
        mode = st.radio("Menu", ["Image", "Audio", "JSON Video Builder"])
        image_model = st.text_input("Image Model", value=DEFAULT_IMAGE_MODEL)
        audio_model = st.text_input("Audio Model", value=DEFAULT_AUDIO_MODEL)
        voice_name = st.text_input("Voice Name", value=DEFAULT_VOICE)

    if mode == "Image":
        st.subheader("Reference Image + Prompt → Generated Image")
        uploaded_file = st.file_uploader(
            "Upload reference image", type=["png", "jpg", "jpeg", "webp"]
        )
        prompt = st.text_area(
            "Prompt",
            value='Add a text overlay "Dhurandhar 2 Movie Review" in dark yellow font.',
            height=120,
        )

        if st.button("Generate Image", type="primary"):
            if not gemini_api_key:
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
                    generated_image_bytes, generated_text = generate_image(
                        api_key=gemini_api_key,
                        prompt=prompt.strip(),
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                        model=image_model.strip(),
                    )
                st.subheader("Generated Result")
                if generated_image_bytes:
                    st.image(generated_image_bytes, caption="Generated Image", use_container_width=True)
                    st.download_button(
                        "Download image",
                        data=generated_image_bytes,
                        file_name="generated_image.jpg",
                        mime="image/jpeg",
                    )
                else:
                    st.warning("Model response did not include an image.")
                if generated_text:
                    st.markdown("**Model text response:**")
                    st.write(generated_text)
    elif mode == "Audio":
        st.subheader("Script → Generated Audio")
        script_text = st.text_area("Script", height=220)

        if st.button("Generate Audio", type="primary"):
            if not gemini_api_key:
                st.error("Please enter your Gemini API key.")
            elif not script_text.strip():
                st.error("Please enter a script for speech generation.")
            else:
                with st.spinner("Generating audio..."):
                    audio_bytes, audio_mime_type, generated_text = generate_audio(
                        api_key=gemini_api_key,
                        script_text=script_text.strip(),
                        voice_name=voice_name.strip() or DEFAULT_VOICE,
                        model=audio_model.strip(),
                    )
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
    else:
        st.subheader("JSON → Images + Telugu Audio + Video Reel")
        st.caption(
            "Each JSON object must include: title, summary, image_search_keywords, and article_url."
        )
        uploaded_json = st.file_uploader("Upload JSON file", type=["json"])
        raw_json = st.text_area("JSON payload", value=SAMPLE_JSON, height=320)
        video_format = st.selectbox("Final video format", ["mp4", "webm"], index=0)

        json_source = raw_json
        if uploaded_json is not None:
            json_source = uploaded_json.getvalue().decode("utf-8")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Preview JSON items"):
                try:
                    preview_items = parse_json_items(json_source)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.success(f"Loaded {len(preview_items)} JSON items.")
                    for index, item in enumerate(preview_items, start=1):
                        st.markdown(f"**{index}. {item.title}**")
                        st.write(item.summary)
                        st.caption(item.article_url)
        with col2:
            run_pipeline = st.button("Generate videos from JSON", type="primary")

        if run_pipeline:
            if not gemini_api_key:
                st.error("Please enter your Gemini API key.")
            elif not searchapi_key:
                st.error("Please enter your SearchApi.io key.")
            else:
                try:
                    items = parse_json_items(json_source)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0.0, text="Starting...")
                    try:
                        results, final_video_bytes, final_filename = process_story_items(
                            items=items,
                            gemini_api_key=gemini_api_key,
                            searchapi_key=searchapi_key,
                            image_model=image_model.strip(),
                            audio_model=audio_model.strip(),
                            voice_name=voice_name.strip() or DEFAULT_VOICE,
                            video_format=video_format,
                            status_placeholder=status_placeholder,
                            progress_bar=progress_bar,
                        )
                    except Exception as exc:
                        status_placeholder.error(str(exc))
                    else:
                        st.success(
                            f"Built {len(results)} individual story videos and one combined {video_format.upper()} reel."
                        )
                        st.download_button(
                            "Download combined video",
                            data=final_video_bytes,
                            file_name=final_filename,
                            mime="video/mp4" if video_format == "mp4" else "video/webm",
                        )
                        for index, result in enumerate(results, start=1):
                            with st.expander(f"Story {index}: {result.item.title}", expanded=index == 1):
                                st.markdown(f"**Article URL:** {result.item.article_url}")
                                st.markdown(
                                    f"**Downloaded source image:** {result.source_image.width}×{result.source_image.height}"
                                )
                                st.caption(f"Search result source page: {result.source_image.source_page}")
                                st.caption(f"Original downloaded image URL: {result.source_image.source_url}")
                                st.image(
                                    result.source_image.data,
                                    caption="Downloaded source image from SearchApi result",
                                    use_container_width=True,
                                )
                                st.image(
                                    result.generated_image_bytes,
                                    caption="Gemini-generated overlay image",
                                    use_container_width=True,
                                )
                                st.audio(
                                    result.generated_audio_bytes,
                                    format=result.generated_audio_mime,
                                )
                                st.video(result.clip_bytes)
                                clip_mime = "video/mp4" if result.clip_filename.endswith(".mp4") else "video/webm"
                                st.download_button(
                                    f"Download clip {index}",
                                    data=result.clip_bytes,
                                    file_name=result.clip_filename,
                                    mime=clip_mime,
                                    key=f"clip-download-{index}",
                                )
                                st.download_button(
                                    f"Download image {index}",
                                    data=result.generated_image_bytes,
                                    file_name=f"story_{index:02d}.jpg",
                                    mime="image/jpeg",
                                    key=f"image-download-{index}",
                                )
                                audio_ext = mimetypes.guess_extension(result.generated_audio_mime) or ".wav"
                                st.download_button(
                                    f"Download audio {index}",
                                    data=result.generated_audio_bytes,
                                    file_name=f"story_{index:02d}{audio_ext}",
                                    mime=result.generated_audio_mime,
                                    key=f"audio-download-{index}",
                                )
                                if result.generated_text:
                                    st.markdown("**Gemini image response text:**")
                                    st.write(result.generated_text)
                                if result.tts_text:
                                    st.markdown("**Gemini TTS response text:**")
                                    st.write(result.tts_text)


if __name__ == "__main__":
    render_app()
