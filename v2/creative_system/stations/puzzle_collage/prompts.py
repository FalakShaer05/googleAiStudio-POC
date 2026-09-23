BACKGROUND_LOCK = (
    "BACKGROUND LOCK: solid pure white RGB(255,255,255) only. "
    "No beige, cream, ivory, off-white, tan, or warm paper. "
    "No checkerboard. No gray wash. Ink on white, nothing else."
)


def build_prompt() -> str:
    return """Convert the uploaded image into a coloring-book page of the SAME scene.

ART EFFECT — fillable line art:
- Clean dark ink contour lines only (fine-liner / pen-and-ink).
- Every area that was colored in the photo becomes an EMPTY region the user can color in.
- No photographic color, no grayscale shading, no hatching fills, no gray wash.
- Lines stay complete on every side — do not fade, dissolve, or drop out.
- Keep the same composition, subjects, objects, and crop so the page is recognizable.

BACKGROUND (critical — read twice):
- Flat solid PURE WHITE only. Every background and open interior pixel is RGB(255,255,255).
- Ink strokes sit directly on that white field. No halo, outline, drop shadow, or glow.
- NEVER use beige, cream, ivory, off-white, tan, warm paper, or page texture.
- NEVER draw a checkerboard, grid, or transparency preview.
- No title, caption, UI, watermark, or extra border frame.

OUTPUT: full-page coloring sheet on solid pure white, same aspect as the upload, one scene only."""
