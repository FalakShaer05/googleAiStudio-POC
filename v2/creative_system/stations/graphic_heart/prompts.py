def build_prompt(message: str, location_label: str, latitude: float | None, longitude: float | None) -> str:
    coords = ""
    if latitude is not None and longitude is not None:
        lat_hem = "N" if latitude >= 0 else "S"
        lng_hem = "E" if longitude >= 0 else "W"
        coords = f"{abs(latitude):.4f}° {lat_hem}, {abs(longitude):.4f}° {lng_hem}"
    location = (location_label or "THE PLACE").strip().upper()
    coord_line = f'Third line: "{coords}" in smaller dark red/brown sans-serif.' if coords else "Omit a coordinate line if none were provided."
    return f"""Create a square keepsake print: a HEART-SHAPED stylized city map with typography beneath.

MAP TREATMENT:
- Image 1 is a real map snapshot of the chosen place. Use its actual street grid.
- Crop/mask the map into a large heart shape in the upper half of a cream square.
- Restyle the map as high-contrast two-tone graphics: muted rust-red city blocks, off-white/cream streets.
- Place a solid black location-pin icon at the marked point inside the heart.
- Do not keep Google UI, logos, or the original full-color basemap look.

TEXT (centered under the heart, in this exact order):
- Top line, elegant black cursive: "{message}"
- A thin rust-red horizontal rule
- Next line, black wide-tracked all-caps sans-serif: "{location}"
- {coord_line}
- A small solid red heart icon at the bottom center.

No extra slogans. No "Graphic Heart" logo. No Next button. No city-skyline UI chrome.
OUTPUT: square 1:1 finished print on textured off-white paper."""
