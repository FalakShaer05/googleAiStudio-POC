BACKGROUND_LOCK = (
    "BACKGROUND LOCK: solid pure white RGB(255,255,255) across the whole image. "
    "No gray paper, no paper texture, no border, no rectangle, no checkerboard."
)

# Never list specific features/accessories here — the image model treats them as things to draw.
STYLE_INSTRUCTION = (
    "Pencil technique: fine light-gray continuous graphite line art on pure white, "
    "delicate strand-by-strand strokes for hair, soft clean contour lines, "
    "minimal shading — no charcoal blocks, no gray fills, no stipple dots."
)


def build_line_art_prompt() -> str:
    """Full-face in-place trace — then we hard-split color | pencil in post."""
    return """Convert the uploaded selfie into a fine light-gray PENCIL LINE-ART drawing
on pure white. This is a TRACE of the photo, not a new portrait.

GEOMETRY LOCK:
- Trace ONLY what is visible in the photo. Do NOT add or remove features.
- Every line sits exactly on top of the photo detail it replaces — same position,
  size, and angle. Do NOT turn the head, shrink, enlarge, move, or re-center.
- Keep the same pose and viewpoint (front, three-quarter or profile).
- Same person, same gender, same expression. Do not beautify.

PENCIL STYLE:
- Delicate continuous graphite strokes, strand-by-strand hair, soft clean contours.
- Light-gray lines on pure white — no gray paper, no box, no border, no heavy fills.
- Minimal shading. No charcoal blocks. No stipple or noise.

OUTPUT: the same canvas and framing as the input — one aligned pencil portrait of
the same person in the same place."""


def build_split_prompt() -> str:
    return build_line_art_prompt()


def build_sketch_prompt() -> str:
    return build_line_art_prompt()


def build_prompt() -> str:
    return build_line_art_prompt()
