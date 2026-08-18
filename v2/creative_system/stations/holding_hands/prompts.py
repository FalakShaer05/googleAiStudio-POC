from datetime import datetime


def _format_year(date_text: str) -> str:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(date_text.strip(), fmt).strftime("%Y")
        except ValueError:
            continue
    digits = "".join(ch for ch in date_text if ch.isdigit())
    if len(digits) >= 4:
        return digits[-4:]
    return date_text


def _pretty_date(date_text: str) -> str:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            parsed = datetime.strptime(date_text.strip(), fmt)
            return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}"
        except ValueError:
            continue
    return date_text


STYLE_INSTRUCTION = (
    "LOCKED STYLE TARGET. Clone this print EXACTLY for composition, negative space, "
    "hand pose, line-and-wash rendering, name-into-wrist calligraphy, year placement, "
    "date overlay, bottom heart ornament, cream background, and empty corners. "
    "Do NOT add leaves, branches, gold sparkles, watercolor splatter, or extra decoration. "
    "Do NOT copy the names, date, or skin tones from this reference. "
    "Swap in the user names, user date, and skin tones from the two person photos."
)


def build_prompt(name_a: str, name_b: str, date_text: str, caption: str = "") -> str:
    pretty_date = _pretty_date(date_text)
    year = _format_year(date_text)
    caption_line = (
        f'If there is room under the heart ornament, you may add this short caption '
        f'in the same thin black script: "{caption}".'
        if caption else
        "Do not invent an extra caption, quote, or location line."
    )
    return f"""Reproduce the locked style-target print as a finished square keepsake.
The output must look like the SAME artwork with only names, date, and skin tones changed.

ART STYLE — thin black line-and-wash watercolor:
- Solid warm cream / off-white background. Generous empty margins. Almost no paper grain.
- Delicate consistent black ink outlines. Soft blended watercolor skin (pale peach to warm tan).
- Shading only at joints, between fingers, and along palm contours. No heavy realism.
- Minimalist, clean, sentimental wedding-stationery look.

HAND POSE — copy the style target, not a handshake and not a tight finger-knot:
- Two hands meet from the top-left and top-right, forming a tall V.
- The RIGHT hand sits slightly in front. Its fingers point downward and rest on the other hand.
- The LEFT hand sits behind. Its fingers curl around the side and bottom of the front hand.
- Gentle intimate clasp. Fingers are visible and separated — not a dense interlaced knot.
- Wrists stay as thin ink lines until they bloom into watercolor at the hands.

NAMES ARE THE ARM LINES (critical):
- Top left, thin elegant calligraphy: "{name_a}". The tail of the last letter continues as one unbroken line that becomes the outer edge of the LEFT wrist.
- Top right, matching calligraphy: "{name_b}". The lead-in stroke of the first letter continues as one unbroken line that becomes the outer edge of the RIGHT wrist.
- No separate decorative rules next to the names. The letterforms ARE the wrists.

DATE BLOCK — below the hands, never overlapping them:
- Large widely-spaced semi-transparent serif year "{year}" sitting UNDER the clasped hands, not behind them. Each digit a different muted pastel (blue-gray, tan, terracotta, lavender) at about 25% opacity.
- Over those digits, the full date "{pretty_date}" in one thin black cursive script (month, day, and year all the same script — do not mix serif into the date).
- Centered under that: a small solid tan heart flanked by two thin horizontal lines that each end in a small dot.

SKIN / IDENTITY:
- Image 1 = left person / name {name_a}. Image 2 = right person / name {name_b}.
- Match each photo's real skin tone, age impression, and hand character. Do not exaggerate a dark/light contrast. Draw illustrated hands — never paste the photos.

FORBIDDEN (these belong to a different design — never include them):
- Botanical leaves, branches, foliage in any corner
- Gold glitter, sparkles, speckles, watercolor splatter
- Year numerals overlapping or showing through the hands
- Mixed fonts on the date line (cursive month + serif day)
- Dense anatomical finger-knot / fully interlaced grip
- Photography, collage, UI chrome, frames, watermarks, extra people

TEXT MUST READ EXACTLY:
- Name left: {name_a}
- Name right: {name_b}
- Date: {pretty_date}
{caption_line}

OUTPUT: one finished square art print, 1:1. No mockup, no device, no watermark."""
