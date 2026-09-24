BACKGROUND_LOCK = (
    "BACKGROUND LOCK: solid pure white RGB(255,255,255) behind the whole portrait. "
    "No checkerboard, gray squares, beige, or paper texture."
)

STYLE_INSTRUCTION = (
    "STYLE TARGET — copy this exact ART EFFECT only: "
    "vertical midline split, LEFT half full-color photo cutout, "
    "RIGHT half light-gray pencil sketch, both on pure white, features aligned. "
    "Do NOT copy this person's face or identity — only the half-photo / half-pencil treatment."
)


def build_split_prompt() -> str:
    """One-shot split: photo left | pencil right, locked to the selfie geometry."""
    return """Edit the uploaded selfie into a vertical half-and-half portrait of THE SAME person.

CRITICAL — geometry lock (do not break the face):
- Keep the exact same crop, scale, head position, pose, and camera angle as the selfie.
- Do NOT reframe, zoom, mirror, or redraw the person in a new pose.
- Features must meet seamlessly at the vertical center line (eyes, nose, lips, chin, hair).

ART EFFECT:
- Exact vertical midline through the center of the face.
- LEFT half: keep as the realistic full-color photograph of this person (skin, hair, clothes, jewelry).
- RIGHT half: convert to a refined light-gray PENCIL SKETCH of the matching half
  (thin continuous graphite lines, hair strands, facial features, jewelry, clothing — finished sketch, not sparse outlines).
- Sharp hard cut at the midline — no soft blend.
- Background on BOTH halves: flat pure white RGB(255,255,255). Subject already on white — keep it white.

OUTPUT: portrait 3:4, one subject, white background, perfectly aligned split."""


def build_line_art_prompt() -> str:
    """Kept for imports; split flow uses build_split_prompt()."""
    return build_split_prompt()


def build_prompt() -> str:
    return build_split_prompt()
