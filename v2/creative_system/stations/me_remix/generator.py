"""Me Remix — half color photo | half light line art (same effect as Selfie Becoming)."""
from ..selfie_becoming.generator import generate_half_photo_half_line_art


def generate(output_path: str, selfie_path: str, **_kwargs):
    return generate_half_photo_half_line_art(
        output_path=output_path,
        selfie_path=selfie_path,
        style_target_id="me-remix",
        operation="art_generation:creative:me-remix",
    )
