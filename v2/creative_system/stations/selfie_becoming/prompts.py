BACKGROUND_LOCK = (
    "BACKGROUND LOCK: solid pure white RGB(255,255,255) across the whole image. "
    "No gray paper, no paper texture, no border, no rectangle, no checkerboard."
)

# Never list specific features/accessories here — the image model treats them as things to draw.
STYLE_INSTRUCTION = (
    "Pencil technique: fine light-gray continuous graphite line art on pure white, "
    "delicate strand-by-strand strokes for hair, soft clean contour lines, "
    "minimal shading — no charcoal blocks, no gray fills."
)


def build_split_prompt() -> str:
    """In-place edit: left half untouched photo, right half becomes pencil of the SAME pixels."""
    return """IN-PLACE EDIT of the uploaded photo. This is ONE continuous portrait, not two images.

LEFT HALF (x < 50%): DO NOT CHANGE. Leave every pixel of the color photo exactly as it is.

RIGHT HALF (x >= 50%): redraw ONLY this half as a fine light-gray PENCIL LINE-ART drawing,
tracing the photo that is already there:
- Trace ONLY what is visible in the photo. Do NOT add any feature, facial hair, accessory,
  or object that is not in the photo. Do NOT remove anything that is in it.
- Every line sits exactly on top of the photo detail it replaces — same position, size and angle.
- The face must continue seamlessly across the vertical center line so both halves read as
  one person in one pose.
- Keep the same pose and viewpoint (front, three-quarter or profile). Do NOT turn the head,
  shrink, enlarge, move, or re-center the face. Do NOT draw a new portrait.
- Same person, same gender, same expression. Do not beautify.

PENCIL STYLE (right half only):
- Delicate continuous graphite strokes, strand-by-strand hair, soft clean contours.
- Minimal shading, light-gray lines, pure white background — no gray paper, no box, no border.

OUTPUT: the same canvas and framing as the input — left half color photo, right half pencil,
one aligned portrait with a sharp vertical cut exactly in the middle."""


def build_sketch_prompt() -> str:
    return build_split_prompt()


def build_line_art_prompt() -> str:
    return build_split_prompt()


def build_prompt() -> str:
    return build_split_prompt()
