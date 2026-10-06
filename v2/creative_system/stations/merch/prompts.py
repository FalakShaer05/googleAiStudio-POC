"""Prompts for Gemini merch mockups (apparel) and sticker layouts."""
from __future__ import annotations

from typing import Any, Dict


def build_prompt(product: Dict[str, Any]) -> str:
    mode = product["mode"]
    label = product["label"]
    pid = product.get("id") or ""

    if mode == "mockup" and pid == "cap":
        return """Create a realistic merchandise product photo of a CAP.

IMAGE 1 = BLANK PRODUCT TEMPLATE. Keep this exact cap: shape, color, mesh, brim, camera angle, lighting, white studio background.
IMAGE 2 = SOURCE ARTWORK to print. Use this design exactly — do not invent a different character or restyle it.

Print placement (match a POD store listing):
- Cover the ENTIRE solid front panel / crown of the cap with the artwork, edge to edge on that panel.
- Leave the VISOR / BRIM completely blank and untouched — no artwork, text, or color on the brim at all.
- Keep the artwork's natural proportions — do NOT stretch or squash faces/text.
- Softly warp the print to the cap's curved front panel only.
- Do not place art on the mesh sides/back or on the brim.
- Plain white background. No UI, price tags, watermarks, or extra props."""

    if mode == "mockup" and pid == "tshirt":
        return """Create a realistic merchandise product photo of a T-SHIRT.

IMAGE 1 = BLANK PRODUCT TEMPLATE. Keep this exact shirt: color, shape, sleeves, collar, folds, camera angle, lighting, white studio background.
IMAGE 2 = SOURCE ARTWORK to print. Use this design exactly — do not invent a different character or restyle it.

CRITICAL — print size (this is the main requirement):
- The artwork must be a BIG full-front print, similar scale to a full-bleed tote print but on the shirt torso.
- Width: almost seam-to-seam on the FRONT body — leave only a small margin (~5–10%) before each armhole. A tiny centered stamp is WRONG.
- Height: from just below the collar down to the lower torso / near the hem area — a large tall print, not a thin strip.
- The print should dominate the shirt front so it looks like a store “print out” listing, NOT a small chest logo.

Boundaries (do not violate):
- FRONT panel only — do NOT wrap onto sleeves, do NOT cross the shoulder seams onto the sleeves, do NOT print the collar or back.
- Keep artwork proportions — do NOT stretch or squash faces/text. Cover/crop to fill the large print area; never distort.
- Soft fabric warp / fold shading is OK.
- Plain white background. No UI, price tags, watermarks, or props.

If the print looks small or floating in the middle of the shirt, the result is incorrect — make it much larger."""

    if mode == "mockup" and pid == "hoodie":
        return """Create a realistic merchandise product photo of a HOODIE.

IMAGE 1 = BLANK PRODUCT TEMPLATE. Keep this exact hoodie: color, hood, pocket, sleeves, camera angle, lighting, white studio background.
IMAGE 2 = SOURCE ARTWORK to print. Use this design exactly — do not invent a different character or restyle it.

CRITICAL — print size (this is the main requirement):
- The artwork must be a BIG full-front chest print, same visual weight as a full-bleed tote print.
- Width: almost seam-to-seam on the FRONT body — leave only a small margin (~5–10%) before each armhole. A tiny centered stamp is WRONG.
- Height: from below the neckline/hood opening down to just above the kangaroo pocket — fill that whole chest band.
- The print should dominate the hoodie front like a store “print out” listing, NOT a small logo.

Boundaries (do not violate):
- FRONT chest only — do NOT wrap onto sleeves, hood exterior, or the pocket; do NOT cross shoulder seams onto sleeves.
- Keep artwork proportions — do NOT stretch or squash. Cover/crop to fill the large print area; never distort.
- Soft fabric warp is OK.
- Plain white background. No UI, price tags, watermarks, or props.

If the print looks small or floating in the middle of the hoodie, the result is incorrect — make it much larger."""

    if mode == "mockup" and pid == "tote":
        return """Create a realistic merchandise product photo of a TOTE BAG.

IMAGE 1 = BLANK PRODUCT TEMPLATE. Keep this exact tote: fabric color, handles, seams, camera angle, lighting, white studio background.
IMAGE 2 = SOURCE ARTWORK to print. Use this design exactly — do not invent a different character or restyle it.

Print placement (match a POD store listing):
- Cover the ENTIRE front face of the bag edge-to-edge (full bleed on the bag panel).
- Keep the artwork's natural proportions — do NOT stretch or squash. Use cover/crop to fill the panel.
- Leave only the handles/straps unprinted.
- Plain white background. No UI, price tags, watermarks, or props."""

    if mode == "sticker_single":
        return f"""Create a professional die-cut sticker product photo for "{label}".

SOURCE ARTWORK (IMAGE 1) is the ONLY design — keep colors, character, and details exact.
STYLE REFERENCE (IMAGE 2) is layout/finish only: thick white sticker contour on plain white.

Requirements:
- ONE sticker of the source artwork with a thick even white die-cut border.
- Plain white background. No props, text, UI, or watermark.
- Do not invent a different character."""

    if mode == "sticker_fan":
        return f"""Create a professional "{label}" pack product photo.

SOURCE ARTWORK (IMAGE 1) is the ONLY design — keep it exact.
STYLE REFERENCE (IMAGE 2) is layout only: identical die-cut stickers fanned on white.

Requirements:
- Exactly 5 identical stickers of the source artwork with thick white die-cut borders.
- Fan / slight overlap. Plain white background. No props, text, or UI."""

    if mode == "sticker_sheet":
        return f"""Create a professional "{label}" product photo.

SOURCE ARTWORK (IMAGE 1) is the ONLY design — keep it exact.
STYLE REFERENCE (IMAGE 2) is layout only: same sticker repeated in size rows on white.

Requirements:
- Sticker sheet of ONLY the source artwork.
- About 3 rows, larger on top and smaller below (roughly 4 / 5 / 8).
- Thick white die-cut border on each. Plain white background. No props or text."""

    return f"""Create a merch product photo for "{label}" from the source artwork. White background. No UI or text."""
