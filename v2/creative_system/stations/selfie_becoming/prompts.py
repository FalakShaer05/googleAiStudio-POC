def build_prompt() -> str:
    return """Convert the uploaded selfie into a minimalist black-and-white contour line drawing of the same person, chest-up, facing forward.

ART EFFECT — "becoming":
- Thin ink contour lines on a solid white background. No shading, no color, no solid fills, no gray wash except the fade described below.
- Left half of the portrait: finished, dark, clearly defined pen-and-ink lines (eyes, brows, nose, lips, hair, collar).
- From the vertical midline of the face, lines gradually fade: they become thinner and lighter gray toward the right edge, dissolving into the white paper.
- Far-right hair and shoulder lines should be almost invisible.
- Preserve the person's identity: face shape, expression, hairstyle, glasses if worn.
- No photography remaining, no halftone, no UI, no title text.

OUTPUT: portrait 3:4, white background, one subject only."""
