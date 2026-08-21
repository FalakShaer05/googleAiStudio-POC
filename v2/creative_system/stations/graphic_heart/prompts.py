STYLE_INSTRUCTION = (
    "LAYOUT AND PALETTE TARGET only (blurred on purpose). "
    "Copy the overall composition: large heart in the upper half, cream paper, "
    "rust-red/cream two-tone graphic style, centered typography stack under the heart, "
    "thin rust rule, tiny heart icon at the bottom. "
    "IGNORE every street, highway, pin location, and every word printed on this reference. "
    "Those belong to a different sample place. Streets, pin, and labels come only from the "
    "user map images and the text prompt."
)


def map_lock() -> str:
    return (
        "HARD LOCK: The heart must be filled with the user map's actual street geometry. "
        "Do not reuse the sample/style-target map. Do not invent a generic downtown grid. "
        "Do not copy Wynwood, Miami, or any other place from the layout reference."
    )


def build_prompt(message: str, location_label: str, latitude: float | None, longitude: float | None) -> str:
    coords = ""
    if latitude is not None and longitude is not None:
        lat_hem = "N" if latitude >= 0 else "S"
        lng_hem = "E" if longitude >= 0 else "W"
        coords = f"{abs(latitude):.4f}° {lat_hem}, {abs(longitude):.4f}° {lng_hem}"
    location = (location_label or "THE PLACE").strip().upper()
    coord_line = f'Third line: "{coords}" in smaller dark red/brown sans-serif.' if coords else "Omit a coordinate line if none were provided."
    return f"""Create a square keepsake print: a HEART-SHAPED stylized city map with typography beneath.

MAP SOURCE (mandatory):
- IMAGE 1 and IMAGE 2 are the same user map of THIS place. Trace its actual roads, blocks, water, parks, and any pin already on it.
- The street pattern inside the heart MUST be recognizably that snapshot, only recolored.
- The last image is a blurred layout sample of a DIFFERENT place. Do not copy its streets, pin, or text.

MAP TREATMENT:
- Crop/mask IMAGE 1 into a large heart shape in the upper half of a cream square.
- Restyle as high-contrast two-tone graphics: muted rust-red city blocks, off-white/cream streets.
- Place a solid black location-pin icon at the marker already on IMAGE 1 (or at the visual center of IMAGE 1 if there is no marker).
- Do not keep Google UI, logos, compass, or the original full-color basemap look.

TEXT (centered under the heart, in this exact order — use these strings, not the sample's text):
- Top line, elegant black cursive: "{message}"
- A thin rust-red horizontal rule
- Next line, black wide-tracked all-caps sans-serif: "{location}"
- {coord_line}
- A small solid red heart icon at the bottom center.

No extra slogans. No "Graphic Heart" logo. No Next button. No city-skyline UI chrome.
OUTPUT: square 1:1 finished print on textured off-white paper."""
