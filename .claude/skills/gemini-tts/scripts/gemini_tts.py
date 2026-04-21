#!/usr/bin/env python3
"""
Gemini TTS Script
Converts text to speech using the Gemini 2.5 Flash TTS API.
Returns a WAV file (raw PCM audio wrapped with WAV headers).

Usage:
    python gemini_tts.py --text "Hello world" --output out.wav
    python gemini_tts.py --prompt "Say in a warm tone: Welcome!" --output out.wav
    python gemini_tts.py --text "Hello" --prompt "Say dramatically:" --output out.wav

API key is read from GEMINI_API_KEY env var.
"""

import argparse
import os
import sys
import wave
import json
import base64
import urllib.request
import urllib.error


MODEL_ID = "gemini-2.5-flash-preview-tts"
DEFAULT_VOICE = "Achird"  # Friendly voice


def write_wav(filename: str, pcm_data: bytes, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)


def build_contents(text: str | None, prompt: str | None) -> str:
    """Combine text and prompt into a single input string."""
    if prompt and text:
        return f"{prompt}\n{text}"
    elif prompt:
        return prompt
    elif text:
        return text
    else:
        raise ValueError("Provide at least --text or --prompt")


def call_gemini_tts(api_key: str, content: str, voice: str) -> bytes:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL_ID}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": content}]}],
        "generationConfig": {
            "responseModalities": ["audio"],
            "temperature": 1,
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": voice
                    }
                }
            },
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            response_body = resp.read()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP Error {e.code}: {e.reason}", file=sys.stderr)
        print(f"Details: {error_body}", file=sys.stderr)
        sys.exit(1)

    result = json.loads(response_body)

    # Navigate to inline audio data
    try:
        part = result["candidates"][0]["content"]["parts"][0]
        inline = part["inlineData"]
        mime_type = inline.get("mimeType", "")
        raw_b64 = inline["data"]
    except (KeyError, IndexError) as e:
        print(f"Unexpected response structure: {e}", file=sys.stderr)
        print(json.dumps(result, indent=2), file=sys.stderr)
        sys.exit(1)

    pcm_bytes = base64.b64decode(raw_b64)
    print(f"Audio received — mime: {mime_type}, size: {len(pcm_bytes)} bytes")
    return pcm_bytes


def main():
    parser = argparse.ArgumentParser(description="Gemini TTS — text to WAV")
    parser.add_argument("--text", help="Plain text to speak")
    parser.add_argument(
        "--prompt",
        help='Style/instruction prompt (e.g. "Say warmly:" or a full director-style prompt)',
    )
    parser.add_argument(
        "--output", default="output.wav", help="Output file path (default: output.wav)"
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Voice name (default: {DEFAULT_VOICE} — Friendly)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if not args.text and not args.prompt:
        parser.error("Provide at least one of --text or --prompt")

    content = build_contents(args.text, args.prompt)
    print(f"Voice: {args.voice}")
    print(f"Input: {content[:120]}{'...' if len(content) > 120 else ''}")

    pcm = call_gemini_tts(api_key, content, args.voice)
    write_wav(args.output, pcm)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
