"""Origami system prompt — lock target camera; wrap open-sheet print."""


def build_prompt(user_prompt: str = "", layout_hint: str = "") -> str:
    extra = (user_prompt or "").strip()
    extra_block = f"\nADDITIONAL USER DIRECTION:\n{extra}\n" if extra else ""
    layout_block = f"\n{layout_hint}\n" if layout_hint else ""

    return f"""IN-PLACE edit of IMAGE 1. Wrap IMAGE 2's print onto it. Do not redesign.

IMAGE 1 — TARGET FOLDED MODEL (LOCKED CAMERA + SHAPE)
- This is the ONLY allowed final pose, silhouette, subject, orientation, camera angle, scale, and centering.
- Three-quarter / product view as shown — keep that exact viewpoint.
- FORBIDDEN: top-down views, orthographic flat layouts, different animals/objects, reinvented silhouettes.
- Ignore IMAGE 1 paper color — geometry only.

IMAGE 2 — COLOR PLATE (print to wrap)
- Visitor sheet colors/regions. Wrap onto IMAGE 1's existing faces.
- Keep exact hues and region placement (not random facet painting).

IMAGE 3 — CREASE PATTERN (topology hint only, small)
- Fold structure only. Do NOT copy its top-down framing into the output camera.

ORIENTATION (critical):
- Output camera = IMAGE 1 camera. Never the top-down square view of IMAGE 2/3.
- Same subject as IMAGE 1 (e.g. frog stays frog in the same crouch/angle).

COLOR:
- From IMAGE 2 only. No invent, no enhance, no copying IMAGE 1 colors.
- Contiguous regions stay contiguous after wrapping.
{layout_block}
BACKGROUND: solid flat cream (#F5F0E6) only — real transparency is added in code later.
No text, watermarks, captions, UI, or extra objects.
{extra_block}"""


def build_trailing_instruction(layout_hint: str = "") -> str:
    hint = f" {layout_hint}" if layout_hint else ""
    return (
        "CRITICAL: Identical camera/pose/subject to IMAGE 1 — never top-down. "
        "Wrap IMAGE 2 print onto IMAGE 1 faces only."
        f"{hint} Cream background. No text."
    )
