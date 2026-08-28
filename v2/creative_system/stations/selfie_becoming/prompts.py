STYLE_INSTRUCTION = (
    "STYLE TARGET (line-art effect only). Copy the becoming contour treatment: "
    "finished dark ink on the left, lines thinning and fading to light gray on the right. "
    "The reference sits on a FLAT PURE WHITE field. Match that: solid RGB(255,255,255) "
    "behind the lines. Do NOT copy cream, beige, ivory, or paper. "
    "Do NOT draw a checkerboard, gray squares, dither, grid, or transparency preview. "
    "Do not copy the face from this reference."
)

BACKGROUND_LOCK = (
    "BACKGROUND LOCK: solid pure white RGB(255,255,255) only. "
    "No checkerboard. No gray/black squares. No beige, cream, or paper texture. "
    "No transparency preview pattern. Ink on white, nothing else."
)


def build_prompt() -> str:
    return """Convert the uploaded selfie into a minimalist black-and-white contour line drawing of the same person, chest-up, facing forward.

ART EFFECT — "becoming":
- Thin ink contour lines only. No shading, no color, no solid fills, no gray wash except the fade described below.
- Left half of the portrait: finished, dark, clearly defined pen-and-ink lines (eyes, brows, nose, lips, hair, collar).
- From the vertical midline of the face, lines gradually fade: they become thinner and lighter gray toward the right edge, dissolving into the white field.
- Far-right hair and shoulder lines should be almost invisible.
- Preserve the person's identity: face shape, expression, hairstyle, glasses if worn.
- No photography remaining, no halftone, no UI, no title text.

BACKGROUND (critical — read twice):
- Flat solid PURE WHITE only. Every background pixel is RGB(255,255,255).
- Ink strokes sit directly on that white field. No halo, outline, drop shadow, or glow around the lines.
- NEVER draw a checkerboard, chessboard, gray squares, dither, grid, or any pattern that looks like a transparency preview. That pattern is wrong.
- NEVER use beige, cream, ivory, off-white, tan, warm paper, or page texture.

OUTPUT: portrait 3:4, solid pure white background, one subject only."""
