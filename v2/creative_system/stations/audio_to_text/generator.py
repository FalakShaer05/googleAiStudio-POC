from ...shared.audio import transcribe_audio
from .prompts import TRANSCRIBE_PROMPT


def generate(output_path: str, audio_path: str, **_kwargs):
    success, transcript = transcribe_audio(audio_path, TRANSCRIBE_PROMPT)
    if not success:
        return False, transcript
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(transcript)
        if not transcript.endswith("\n"):
            handle.write("\n")
    return True, "Transcript generated successfully"
