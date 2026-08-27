import os

from ...shared.audio import transcribe_audio
from ...shared.waveform import load_mono_samples, rms_envelope
from .prompts import TYPE_FROM_AUDIO_PROMPT, normalize_style, type_from_transcript
from .renderer import render_style


def generate(output_path: str, audio_path: str, style: str = "rings", **_kwargs):
    chosen = normalize_style(style)
    ok, transcript = transcribe_audio(audio_path, TYPE_FROM_AUDIO_PROMPT)
    if not ok:
        return False, transcript
    text = type_from_transcript(transcript)
    if not text:
        return False, "No spoken words were found in the audio"

    try:
        samples = load_mono_samples(audio_path)
    except ValueError as exc:
        return False, str(exc)
    except Exception as exc:
        print(f"audio-type decode error: {exc}")
        return False, str(exc)

    bins = 360 if chosen == "rings" else 512
    image = render_style(chosen, rms_envelope(samples, bins=bins), text)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return True, f'Artwork generated from: "{text}"'
