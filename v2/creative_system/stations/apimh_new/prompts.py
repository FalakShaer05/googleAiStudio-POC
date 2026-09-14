"""A Place in My Heart — Explorer / Dreamer map prints."""

TEXT_COLOR = "#1A1A1A"
DREAMER_PIN = "#E85A2D"
ACCENT_GREEN = "#8FBF3F"

TYPES = ("explorer", "dreamer")

TYPE_META = {
    "explorer": {
        "label": "THE EXPLORER",
        "tagline": "Bold. Driven. Always discovering.",
        "heart": (
            "anatomical heart die-cut with a FEW raised cream/white paper partitions "
            "(top vessel separator + one curving chamber wall)"
        ),
        "pin": "solid black map pin with a small white center dot",
    },
    "dreamer": {
        "label": "THE DREAMER",
        "tagline": "Intuitive. Connected. Full of feeling.",
        "heart": "classic stylized valentine heart (single solid cutout, no internal ridges)",
        "pin": f"warm orange ({DREAMER_PIN}) map pin with a small white heart icon in its head",
    },
}

# A few raised die-cut walls — not flat, not a vein tree.
EXPLORER_CUTOUT_LOCK = (
    "EXPLORER CUTOUT LOCK (required — match the product sample): Anatomical OUTER silhouette "
    "plus a FEW raised cream/off-white paper partitions inside (3D die-cut walls with shallow "
    "drop shadow), typically: "
    "(1) a short separator between the top vessel pipes, and "
    "(2) one longer curving chamber wall from upper-left into the body. "
    "About 2–3 partitions total. Map fills the recessed openings between those walls, "
    "including vessel lobes. "
    "FORBIDDEN: a flat empty heart with ZERO internal lining. "
    "FORBIDDEN: thick branching vein/artery trees, coral-like vascular overlays, or many "
    "tiny medical pockets drawn on top of the map."
)

MAP_FIDELITY_LOCK = (
    "MAP FIDELITY LOCK (highest priority): Inside the heart, show a LITERAL crop/mask of "
    "IMAGE 1 / IMAGE 2 — the user's uploaded map screenshot. "
    "Keep the SAME roads, blocks, parks, water, highway colors, street labels, sector names, "
    "and POI icons in the SAME relative positions. "
    "Scale/position the map so important labels near the center stay fully readable — "
    "do not chop them with thick internal paper. "
    "Fill the entire heart aperture edge-to-edge with that map, including vessel lobes. "
    "Do NOT redraw, restyle, simplify, or invent a different city. "
    "Do NOT replace it with the layout-sample map, Miami, Wynwood, a generic downtown, "
    "or any 'pretty map' from memory. If unsure, copy IMAGE 1 pixels more literally."
)


def normalize_type(map_type: str | None) -> str:
    value = (map_type or "explorer").strip().lower()
    return value if value in TYPES else "explorer"


def format_coords(latitude: float | None, longitude: float | None) -> str:
    if latitude is None or longitude is None:
        return ""
    lat_hem = "N" if latitude >= 0 else "S"
    lng_hem = "E" if longitude >= 0 else "W"
    return f"{abs(latitude):.4f}° {lat_hem}, {abs(longitude):.4f}° {lng_hem}"


def style_instruction(map_type: str) -> str:
    chosen = normalize_type(map_type)
    meta = TYPE_META[chosen]
    explorer_note = (
        " REQUIRED: a few raised cream paper partitions (top vessel split + one curving "
        "chamber wall) — not flat, and not a branching vein tree."
        if chosen == "explorer"
        else ""
    )
    return (
        f"LAYOUT / HEART-SHAPE / PIN STYLE TARGET only ({meta['label']}). "
        "Copy ONLY: die-cut heart silhouette, shallow inner shadow on the cut edge, "
        f"and pin style ({meta['pin']}). "
        f"Heart silhouette must be a {meta['heart']}.{explorer_note} "
        f"If a title is present in the print, include a short {ACCENT_GREEN} lime-green "
        "hand-painted brush-stroke underline beneath it. "
        "Outside the heart and text use a FLAT solid pure white (#FFFFFF) backdrop only — "
        "never cream/beige card stock, never a gray/white checkerboard, never striped borders. "
        "CRITICAL: IGNORE and DO NOT COPY any map streets, water, parks, pins, or place names "
        "from this reference — it is a different sample city. "
        "The map fill MUST come only from the user map images (IMAGE 1 / IMAGE 2)."
    )


def layout_lock(map_type: str) -> str:
    chosen = normalize_type(map_type)
    meta = TYPE_META[chosen]
    explorer_extra = f" {EXPLORER_CUTOUT_LOCK}" if chosen == "explorer" else ""
    return (
        f"LAYOUT LOCK ({meta['label']}): Heart cutout dominates the upper ~60% of the frame. "
        f"Silhouette must clearly read as a {meta['heart']} — not the other type. "
        "Background outside the heart and typography must be flat solid #FFFFFF "
        "(no beige card, no checkerboard pattern, no striped borders). "
        f"Keep generous empty margins; no busy chrome or UI.{explorer_extra}"
    )


def map_lock(map_type: str = "explorer") -> str:
    chosen = normalize_type(map_type)
    explorer_extra = f" {EXPLORER_CUTOUT_LOCK}" if chosen == "explorer" else ""
    return f"{MAP_FIDELITY_LOCK}{explorer_extra}"


def build_prompt(
    message: str,
    location_label: str,
    latitude: float | None,
    longitude: float | None,
    map_type: str = "explorer",
) -> str:
    chosen = normalize_type(map_type)
    meta = TYPE_META[chosen]
    coords = format_coords(latitude, longitude)
    location = (location_label or "").strip().upper()
    title = (message or "").strip()
    text_lines = []
    n = 1
    if title:
        text_lines.append(f'{n}. Title: "{title}"')
        n += 1
    if location:
        text_lines.append(f'{n}. Location: "{location}"')
        n += 1
    if coords:
        text_lines.append(f'{n}. Coordinates: "{coords}"')
        n += 1
    if text_lines:
        text_block = (
            "EXACT TEXT TO RENDER (print only these strings — never add font names, type labels, or extra words):\n"
            + "\n".join(text_lines)
        )
        layout_text = (
            "- Heart cutout occupies most of the upper frame; text block sits in the lower third, centered.\n"
            "- Keep comfortable gaps between the heart and any text lines that are present.\n"
        )
        if title:
            layout_text += (
                "- Title sits close beneath the heart with a comfortable gap.\n"
                f"- Directly under the title, draw a short hand-painted {ACCENT_GREEN} "
                "brush-stroke underline (organic tapered ends, ~30–40% of the title width, "
                "centered) — this green accent line is required whenever a title is present.\n"
            )
        if location or coords:
            layout_text += "- Location and coordinates sit closer together below the green underline (or below the title if no underline).\n"
        typography = "TYPOGRAPHY (style only — do NOT print any of these descriptions as visible text):\n"
        if title:
            typography += (
                f"- Title line: large elegant handwritten signature script, weight 400, color {TEXT_COLOR}\n"
                f"- Underline: short {ACCENT_GREEN} lime-green hand-painted brush stroke with "
                "organic tapered edges — not a thin straight geometric rule\n"
            )
        if location:
            typography += f"- Location line: smaller geometric sans-serif, uppercase, wide letter-spacing, semi-bold, color {TEXT_COLOR}\n"
        if coords:
            typography += f"- Coordinates line: clean sans-serif, semi-bold, moderate letter-spacing, color {TEXT_COLOR}\n"
    else:
        text_block = (
            "TEXT: Omit all typography under the heart — no title, no location, no coordinates, "
            "no placeholder words."
        )
        layout_text = (
            "- Heart cutout is the sole focus; leave the lower area empty (no text).\n"
        )
        typography = "TYPOGRAPHY: none — do not invent slogans or labels.\n"
    explorer_treatment = (
        f"""
- {EXPLORER_CUTOUT_LOCK}
- Partitions are raised cream/white paper walls with shallow bevel — not ink lines and not vein drawings.
- Soft shallow shadow on the outer cut edge and on each internal partition.
"""
        if chosen == "explorer"
        else """
- Single solid valentine-heart aperture only — no internal chamber ridges.
- Soft shallow inner shadow along the outer cut edge.
- Map fills the entire heart opening edge-to-edge with no blank gaps.
"""
    )
    return f"""Create a square "A Place in My Heart" keepsake print: a die-cut heart map on a flat solid white (#FFFFFF) field.

TYPE: {meta['label']} — {meta['tagline']}

MAP SOURCE (mandatory — do not invent geography):
- IMAGE 1 and IMAGE 2 are the SAME user map screenshot of THIS exact place.
- Fill the heart by cropping / masking that screenshot — literal geography, not a redraw.
- Preserve recognizable roads, parks, water, labels, and icons from IMAGE 1.
- The last image is a blurred layout sample of a DIFFERENT place. Use it only for heart/pin style. Never copy its map.

{MAP_FIDELITY_LOCK}

MAP TREATMENT:
- Mask IMAGE 1 into a VERY LARGE {meta['heart']} shape — roughly 55–65% of the canvas height, dominant in the upper portion.
- Soft shallow inner shadow / bevel along the heart cut edge only (depth cue). Do NOT fill the rest of the canvas with cream or beige paper.
- Continuity: the map texture runs continuously through the whole silhouette — no missing tiles, no empty vessel tubes, no gray/cream patches where streets should be.
- Keep the uploaded map's own colors and detail (Google-Maps-like look is fine). Do NOT convert to a two-tone red graphic poster. Do NOT invent a prettier substitute map.
- Place a {meta['pin']} at the marker already on IMAGE 1 (or near the visual center of IMAGE 1 if there is no marker). Restyle only the pin; do not move geography.
- You may hide Google UI chrome / compass / watermarks at the edges, but keep the street map content intact.
{explorer_treatment}
BACKGROUND:
- Flat solid pure white (#FFFFFF) everywhere outside the heart and the text block.
- FORBIDDEN: gray/white checkerboard, transparency preview pattern, cream/beige paper fill, striped side borders, page drop-shadow box.
- Do NOT illustrate transparency — leave a plain white field; real alpha is added afterward.

{text_block}

LAYOUT:
{layout_text}- Do NOT print "THE EXPLORER", "THE DREAMER", the taglines, or an OR divider.

{typography}
No extra slogans. No font names on the artwork. No selection checkmarks. No lime selection borders.
OUTPUT: square 1:1 PNG — heart map + typography on flat solid white (#FFFFFF). No checkerboard."""
