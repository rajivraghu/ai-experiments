---
name: gemini-tts
description: Convert text to speech using the Google Gemini 2.5 Flash TTS API. Default voice is Achird (Friendly). Accepts plain text, a style/director prompt, or both. Outputs a WAV file. Use when the user asks to generate speech or audio via Gemini TTS, or says "gemini tts", "speak this with Gemini", "generate voice with Gemini API".
---

# Gemini TTS

Converts text → speech using `gemini-2.5-flash-preview-tts` via the Gemini REST API. Default voice: **Achird (Friendly)**. Output: WAV file.

## What the user provides

| Input | Required | Notes |
|---|---|---|
| `--text` | One of these two | Plain text to speak verbatim |
| `--prompt` | One of these two | Style/tone instructions, or a full director-style prompt |
| Both together | Optional | Prompt = instructions, text = transcript |
| API key | Yes | Stored securely via `ask_secrets` into `.env` |
| Output path | Optional | Defaults to `output.wav` |
| Voice | Optional | Defaults to `Achird`. See voice table below |

## Tools

| Tool | When to Use |
|---|---|
| `ask_secrets` | Collect `GEMINI_API_KEY` from the user securely, write to `.env` |
| `bash` | Run `gemini_tts.py` script after loading the env file |
| `deliver` | Deliver the resulting `.wav` file |

## Workflow

1. **API key** — use `ask_secrets` to collect `GEMINI_API_KEY`, write to `/tmp/gemini-tts.env` (or a project `.env`). Then `source` it before running the script.
2. **Determine inputs** — from the user's request, extract `--text` and/or `--prompt`.
3. **Run the script**:
   ```
   source /tmp/gemini-tts.env && python3 /home/user/.skills/gemini-tts/scripts/gemini_tts.py \
     --text "..." \
     --prompt "..." \
     --output /tmp/output.wav \
     --voice Achird
   ```
4. **Deliver** the `.wav` file using the `deliver` tool.

The script handles base64 decoding and WAV header wrapping automatically. Raw API returns PCM 16-bit 24kHz mono — the script wraps it into a valid `.wav`.

## Voice Options (30 available)

Default is **Achird — Friendly**. Other notable ones:

| Voice | Tone |
|---|---|
| Achird | Friendly ← **default** |
| Zephyr | Bright |
| Kore | Firm |
| Aoede | Breezy |
| Sulafat | Warm |
| Achernar | Soft |
| Puck | Upbeat |
| Charon | Informative |
| Fenrir | Excitable |

Full list: Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, Callirrhoe, Autonoe, Enceladus, Iapetus, Umbriel, Algieba, Despina, Erinome, Algenib, Rasalgethi, Laomedeia, Achernar, Alnilam, Schedar, Gacrux, Pulcherrima, Achird, Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager, Sulafat

## Prompt Style Guide

Gemini TTS is director-style. You can influence tone through the prompt:

- Simple: `"Say warmly: Have a wonderful day!"`
- Rich: Include an Audio Profile, Scene, Director's Notes, and Transcript for full control
- Just text: `--text "Hello world"` works fine — Achird voice speaks naturally

## Limitations

- Text-only input, audio-only output
- 32k token context window per session
- Language is auto-detected (30+ languages supported)

<preflight>
Ask the user:
1. What text should be spoken? (or do they have a full prompt?)
2. Do they want a custom tone/style instruction, or plain speech?
3. Do they already have a `GEMINI_API_KEY` ready?
4. Preferred output filename/path? (default: output.wav)

Then use `ask_secrets` to collect the API key if not already stored.
Run the script and deliver the WAV.
</preflight>
