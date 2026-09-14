"""A Place in My Heart — Explorer / Dreamer map prints."""

TEXT_COLOR = "#1A1A1A"
CREAM = "#F3EDE3"
DREAMER_PIN = "#E85A2D"

TYPES = ("explorer", "dreamer")

TYPE_META = {
    "explorer": {
        "label": "THE EXPLORER",
        "tagline": "Bold. Driven. Always discovering.",
        "heart": (
            "simplified anatomical heart silhouette with a FEW subtle cream-paper "
            "ridges (mostly top vessels + one light chamber hint)"
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

# Restrained Explorer cut: suggest anatomy without cropping away the map.
EXPLORER_CUTOUT_LOCK = (
    "EXPLORER CUTOUT LOCK: Anatomical OUTER silhouette. Optional: 1–2 VERY THIN cream "
    "paper ridges only (a light top-vessel suggestion). "
    "CRITICAL — NO CROPPED / MISSING MAP PIECES: the USER map must fill EVERY opening "
    "inside the silhouette, including the top vessel/aorta lobes. "
    "Do NOT leave blank cream, gray, or empty tubes at the top. "
    "Do NOT cover large map regions with thick cream bridges or solid vessel fills. "
    "Ridges, if any, are hairline paper strips — the map stays continuous and readable. "
    "Shadows: soft and shallow. Aim for elegant product art with maximum map visibility."
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
        " Sparse thin ridges only; map must fill vessel lobes — no blank cropped tops."
        if chosen == "explorer"
        else ""
    )
    return (
        f"LAYOUT / PAPER / HEART-SHAPE / PIN STYLE TARGET only ({meta['label']}). "
        "Copy ONLY: cream card stock, die-cut heart silhouette, shallow inner shadow, "
        f"and pin style ({meta['pin']}). "
        f"Heart silhouette must be a {meta['heart']}.{explorer_note} "
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
        "Cream paper fills the rest of the card with a soft die-cut inner shadow. "
        f"Keep generous margins; no busy chrome or UI.{explorer_extra}"
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
    location = (location_label or "WYNWOOD, FLORIDA").strip().upper()
    coord_block = (
        f'3. Coordinates: "{coords}"'
        if coords
        else "3. Omit the coordinates line."
    )
    explorer_treatment = (
        f"""
- {EXPLORER_CUTOUT_LOCK}
- Top vessels are OPEN map windows (map visible inside), never solid cream stubs.
- Avoid thick cream slabs that crop/hide map pieces. Prefer one continuous map fill.
- Soft shallow bevel only.
"""
        if chosen == "explorer"
        else """
- Single solid valentine-heart aperture only — no internal chamber ridges.
- Soft shallow inner shadow along the outer cut edge.
- Map fills the entire heart opening edge-to-edge with no blank gaps.
"""
    )
    return f"""Create a square "A Place in My Heart" keepsake print: a die-cut heart map on cream paper.

TYPE: {meta['label']} — {meta['tagline']}

MAP SOURCE (mandatory — do not invent geography):
- IMAGE 1 and IMAGE 2 are the SAME user map screenshot of THIS exact place.
- Fill the heart by cropping / masking that screenshot — literal geography, not a redraw.
- Preserve recognizable roads, parks, water, labels, and icons from IMAGE 1.
- The last image is a blurred layout sample of a DIFFERENT place. Use it only for paper/heart/pin style. Never copy its map.

{MAP_FIDELITY_LOCK}

MAP TREATMENT:
- Mask IMAGE 1 into a VERY LARGE {meta['heart']} shape — roughly 55–65% of the canvas height, dominant in the upper portion.
- Present it as a paper die-cut: textured cream ({CREAM}) card stock surrounds the openings; the USER map shows through the cut regions only.
- Continuity: the map texture runs continuously through the whole silhouette — no missing tiles, no empty vessel tubes, no gray/cream patches where streets should be.
- Keep the uploaded map's own colors and detail (Google-Maps-like look is fine). Do NOT convert to a two-tone red graphic poster. Do NOT invent a prettier substitute map.
- Place a {meta['pin']} at the marker already on IMAGE 1 (or near the visual center of IMAGE 1 if there is no marker). Restyle only the pin; do not move geography.
- You may hide Google UI chrome / compass / watermarks at the edges, but keep the street map content intact.
{explorer_treatment}
EXACT TEXT TO RENDER (print only these strings — never add font names, type labels, or extra words):
1. Title: "{message}"
2. Location: "{location}"
{coord_block}

LAYOUT:
- Heart cutout occupies most of the upper frame; text block sits in the lower third, centered on cream paper.
- Title sits close beneath the heart with a comfortable gap.
- Location and coordinates sit closer together below the title.
- Do NOT print "THE EXPLORER", "THE DREAMER", the taglines, or an OR divider.

TYPOGRAPHY (style only — do NOT print any of these descriptions as visible text):
- Title line: large elegant handwritten signature script, weight 400, color {TEXT_COLOR}
- Location line: smaller geometric sans-serif, uppercase, wide letter-spacing, semi-bold, color {TEXT_COLOR}
- Coordinates line: clean sans-serif, semi-bold, moderate letter-spacing, color {TEXT_COLOR}

No extra slogans. No font names on the artwork. No selection checkmarks. No lime selection borders.
OUTPUT: square 1:1 finished print on textured cream paper."""
