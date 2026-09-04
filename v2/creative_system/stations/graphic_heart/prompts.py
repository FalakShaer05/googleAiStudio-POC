ART_COLOR = "#AF3329"
TEXT_COLOR = "#33363F"

STYLE_INSTRUCTION = (
    "LAYOUT AND PALETTE TARGET only (blurred on purpose). "
    "Copy the overall composition: a VERY LARGE heart dominating the upper ~60% of the frame, cream paper, "
    f"two-tone graphic style ({ART_COLOR} city blocks on cream streets), compact typography stack tucked under the heart, "
    f"a short hand-painted {ART_COLOR} brush-stroke divider with generous breathing room above and below, "
    f"tiny {ART_COLOR} heart icon at the bottom. "
    "Typography on the reference: large flowing signature script title, "
    "smaller semi-bold all-caps sans-serif location with wide letter-spacing, "
    "semi-bold sans-serif coordinates in art red. "
    "IGNORE every street, highway, pin location, and every word printed on this reference. "
    "Those belong to a different sample place. Streets, pin, and labels come only from the "
    "user map images and the text prompt."
)


def layout_lock() -> str:
    return (
        "LAYOUT LOCK: Heart must be very large (~60% of frame height). "
        "Title script must be large and prominent. "
        "Divider must be a short hand-painted brush stroke with clear space above and below — not a thin straight line. "
        "Location text must be smaller than the title and semi-bold. Coordinates must be semi-bold."
    )


def map_lock() -> str:
    return (
        "HARD LOCK: The heart must be filled with the user map's actual street geometry. "
        "Do not reuse the sample/style-target map. Do not invent a generic downtown grid. "
        "Do not copy Wynwood, Miami, or any other place from the layout reference."
    )


def format_coords(latitude: float | None, longitude: float | None) -> str:
    if latitude is None or longitude is None:
        return ""
    lat_hem = "N" if latitude >= 0 else "S"
    lng_hem = "E" if longitude >= 0 else "W"
    return f"{abs(latitude):.4f}° {lat_hem}, {abs(longitude):.4f}° {lng_hem}"


def build_prompt(message: str, location_label: str, latitude: float | None, longitude: float | None) -> str:
    coords = format_coords(latitude, longitude)
    location = (location_label or "WYNWOOD, FLORIDA").strip().upper()
    coord_block = (
        f'3. Coordinates: "{coords}"'
        if coords
        else "3. Omit the coordinates line."
    )
    return f"""Create a square keepsake print: a HEART-SHAPED stylized city map with typography beneath.

MAP SOURCE (mandatory):
- IMAGE 1 and IMAGE 2 are the same user map of THIS place. Trace its actual roads, blocks, water, parks, and any pin already on it.
- The street pattern inside the heart MUST be recognizably that snapshot, only recolored.
- The last image is a blurred layout sample of a DIFFERENT place. Do not copy its streets, pin, or text.

MAP TREATMENT:
- Crop/mask IMAGE 1 into a VERY LARGE heart shape — roughly 55–65% of the canvas height, wide and dominant in the upper portion with modest top margin only.
- The heart should feel substantially bigger than a medium-sized heart; maximize map area inside the heart.
- Restyle as high-contrast two-tone graphics: city blocks filled {ART_COLOR}, off-white/cream streets and background.
- Place a solid black location-pin icon at the marker already on IMAGE 1 (or at the visual center of IMAGE 1 if there is no marker).
- Do not keep Google UI, logos, compass, or the original full-color basemap look.

EXACT TEXT TO RENDER (print only these strings — never add font names, style labels, or extra words):
1. Title: "{message}"
2. Location: "{location}"
{coord_block}

LAYOUT (match the reference proportions):
- Heart occupies most of the upper frame; text block sits in the lower third, centered.
- Title sits close beneath the heart point with comfortable but not excessive gap.
- Brush-stroke divider: short (~30–40% of text width), hand-painted look, NOT a thin geometric line.
- Add clear vertical spacing above AND below the brush stroke (roughly equal padding on both sides).
- Location and coordinates sit closer together below the divider; bottom heart icon with modest margin beneath coordinates.

TYPOGRAPHY (style only — do NOT print any of these descriptions as visible text):
- Title line: large elegant handwritten signature script — noticeably bigger than all other text, weight 400, color {TEXT_COLOR}
- Divider: short {ART_COLOR} brush-stroke mark with organic tapered edges, generous whitespace above and below
- Location line: smaller geometric sans-serif, uppercase, wide letter-spacing, semi-bold (weight 600), color {TEXT_COLOR}
- Coordinates line: clean sans-serif, semi-bold (weight 600), moderate letter-spacing, color {ART_COLOR}
- Small solid {ART_COLOR} heart icon at the bottom center

No extra slogans. No font names on the artwork. No "Graphic Heart" logo. No Next button. No city-skyline UI chrome.
OUTPUT: square 1:1 finished print on textured off-white paper."""
