"""
Character generation prompt constants for different hobbies and activities.
"""
from typing import Optional

# Hobby-based character prompts
HOBBY_PROMPTS = {
    "football": """A full-body, hand-drawn cartoon-style caricature with slightly exaggerated features and proportionally larger heads. Each character is actively playing football (soccer), wearing detailed football outfits — jerseys, shorts, socks, cleats, and optional accessories like shin guards, headbands, or captain armbands — in bright and vibrant colors. They are engaged in grounded, realistic poses such as dribbling, passing, shooting, tackling, or celebrating a goal, with clear contact to the ground to avoid any floating appearance. Use bold black outlines, vivid comic-style coloring, and a humorous, energetic aesthetic inspired by traditional marker and watercolor caricatures. The background must be plain white with no patterns or scenery. Maintain full visibility of each subject from head to toe — do not crop, remove, or merge any person. Keep the exaggeration subtle and flattering, not overly distorted.""",

    "basketball": """A full-body, hand-drawn cartoon-style caricature with slightly exaggerated features and proportionally larger heads. Each character is actively playing basketball, wearing detailed basketball outfits — jerseys, shorts, sneakers, and optional accessories like headbands, wristbands, or knee pads — in bright and vibrant colors. They are engaged in grounded, realistic poses such as dribbling, shooting, dunking, passing, or celebrating a basket, with clear contact to the court surface to avoid any floating appearance. Use bold black outlines, vivid comic-style coloring, and a humorous, energetic aesthetic inspired by traditional marker and watercolor caricatures. The background must be plain white with no patterns or scenery. Maintain full visibility of each subject from head to toe — do not crop, remove, or merge any person. Keep the exaggeration subtle and flattering, not overly distorted.""",

    "baseball": """A full-body, hand-drawn cartoon-style caricature with slightly exaggerated features and proportionally larger heads. Each character is actively playing baseball, wearing detailed baseball outfits — jerseys, pants, cleats, caps or helmets, and optional accessories like gloves, bats, catcher's gear, or wristbands — in bright and vibrant colors. They are engaged in grounded, realistic poses such as pitching, batting, catching, sliding, or celebrating a home run, with clear contact to the ground to avoid any floating appearance. Use bold black outlines, vivid comic-style coloring, and a humorous, dynamic aesthetic inspired by traditional marker and watercolor caricatures. The background must be plain white with no patterns or scenery. Maintain full visibility of each subject from head to toe — do not crop, remove, or merge any person. Keep the exaggeration subtle and flattering, not overly distorted.""",

    "cricket": """A full-body, hand-drawn cartoon-style caricature with slightly exaggerated features and proportionally larger heads. Each character is actively playing cricket, wearing detailed cricket outfits — jerseys, pants, cricket shoes, helmets, pads, gloves, and optional accessories like caps, arm guards, or cricket bats — in bright and vibrant colors. They are engaged in grounded, realistic poses such as batting, bowling, fielding, catching, or celebrating a wicket, with clear contact to the ground to avoid any floating appearance. Use bold black outlines, vivid comic-style coloring, and a humorous, dynamic aesthetic inspired by traditional marker and watercolor caricatures. The background must be plain white with no patterns or scenery. Maintain full visibility of each subject from head to toe — do not crop, remove, or merge any person. Keep the exaggeration subtle and flattering, not overly distorted.""",

    "skateboarding": """A full-body, hand-drawn cartoon-style caricature with slightly exaggerated features and proportionally larger heads. Each character is actively skateboarding, wearing detailed skate outfits — t-shirts, hoodies, shorts or pants, sneakers, and optional accessories like helmets, knee pads, elbow pads, or wristbands — in bright and vibrant colors. They are engaged in grounded, realistic poses such as riding, performing tricks, jumping, grinding, or celebrating a successful move, with clear contact between the skateboard and the ground to avoid any floating appearance. Use bold black outlines, vivid comic-style coloring, and a humorous, energetic aesthetic inspired by traditional marker and watercolor caricatures. The background must be plain white with no patterns or scenery. Maintain full visibility of each subject and skateboard from end to end — do not crop, remove, or merge any part. Keep the exaggeration subtle and flattering, not overly distorted."""
}

# Compositing prompt for merging characters on background
COMPOSITING_PROMPT = """Use the attached background exactly as it is — same colors, shapes, and watercolor effect.
Do not change or redraw the background.

CRITICAL CHARACTER PRESERVATION:
- Merge all attached character images into this background, keeping each person EXACTLY as they appear in their original image.
- Each character must maintain their EXACT face, pose, outfit, colors, and equipment from their original image.
- Do NOT add any extra objects, equipment, or elements to any character.
- Each character should only have the equipment/objects they had in their original image (e.g., if a character has a soccer ball, only that soccer ball; if a character has a softball, only that softball).
- Do NOT mix equipment between characters (e.g., do not add a soccer ball to a baseball player).
- Do NOT add any new sports equipment, balls, or objects that were not in the original character images.

CHARACTER PLACEMENT:
- Place the characters naturally around the large "LOVE" text in the background.
- Make the parents slightly larger than the children for natural proportions.
- Position characters so they don't overlap inappropriately.

STRICT RULES:
✅ Keep the background identical every time.
✅ Do not add, remove, or alter any person.
✅ Do NOT add any extra objects, equipment, or elements.
✅ Each character keeps ONLY what they had in their original image.
✅ Only adjust positioning and size slightly to blend them naturally."""

FIFA_WORLD_CUP_PROMPT = """THREE INPUT IMAGES — ROLES (read carefully before generating):

IMAGE 1 = USER PHOTO → sole source for the player's FACE and IDENTITY
IMAGE 2 = JERSEY PHOTO → sole source for KIT/JERSEY design only (optional)
IMAGE 3 = TRADING CARD TEMPLATE → sole source for CARD DESIGN/LAYOUT only

PRIORITY ORDER (when instructions conflict):
1. Identity from Image 1 (highest — face must match Image 1 exactly)
2. Card design from Image 3 (100% template fidelity for frame, layout, typography, badges)
3. Jersey kit from Image 2 (dress Image 1's person in this exact jersey when provided)

CRITICAL — TEMPLATE FIDELITY (Image 3):
Reproduce Image 3 at 100% accuracy for:
- Card dimensions, aspect ratio, and outer frame
- Background colors, gradients, patterns, and textures
- Border style, corner shapes, foil/holographic effects, and shadows
- Logo placement, crest positions, sponsor areas, flags, and badges
- Typography style, font weight, label positions, stat bars, and number styling
- All decorative elements, icons, dividers, and graphic ornaments

DO NOT invent a new card design. DO NOT simplify, modernize, crop, or restyle the template.
Only replace: (1) the portrait face/body in the template with the person from Image 1, and (2) text/stat values in existing slots.

JERSEY (Image 2, if provided):
Apply this exact jersey onto the person from Image 1. Image 2 is clothing reference ONLY — never copy any face from Image 2.

OUTPUT:
One finished trading card visually identical to Image 3, with Image 1's face in the portrait slot and updated player text."""

FIFA_IDENTITY_LOCK = """CRITICAL — IDENTITY LOCK (HIGHEST PRIORITY — Image 1):
The output player's face MUST be recognizably the SAME PERSON as Image 1 (user photo).

PRESERVE from Image 1:
- Face shape, eye shape/color, eyebrows, nose, lips, jawline, cheekbones
- Skin tone and complexion
- Hair color, hair style, facial hair
- Distinctive features (moles, scars, glasses if worn, etc.)
- Approximate age appearance

FORBIDDEN:
- Do NOT keep or copy the face shown in the trading card template (Image 3) — replace it entirely with Image 1
- Do NOT use any face or head from the jersey image (Image 2)
- Do NOT blend, average, or morph faces between images
- Do NOT beautify, age-shift, or generate a generic stock athlete face

Composite Image 1's real face into the template's portrait area. The result must look like the user from Image 1 wearing the kit, inside the card from Image 3."""

FIFA_TEMPLATE_STRICT_RULES = """FINAL CHECKLIST:
- Face in output = Image 1 user photo (NOT the template face, NOT a new face)
- Card design in output = Image 3 template at 100% fidelity
- Jersey in output = Image 2 kit on Image 1's person (if jersey provided)
- If unsure about the face, match Image 1 more closely
- If unsure about the card design, match Image 3 more closely"""

FIFA_POSITIONS = {
    "GK": "Goalkeeper",
    "RB": "Right Back",
    "LB": "Left Back",
    "CB": "Center Back",
    "RWB": "Right Wing Back",
    "LWB": "Left Wing Back",
    "CDM": "Central Defensive Midfielder",
    "CM": "Central Midfielder",
    "CAM": "Central Attacking Midfielder",
    "RM": "Right Midfielder",
    "LM": "Left Midfielder",
    "RW": "Right Winger",
    "LW": "Left Winger",
    "CF": "Center Forward",
    "ST": "Striker",
}


def format_fifa_position(position_code: str) -> str:
    code = (position_code or "").strip().upper()
    if not code:
        return ""
    label = FIFA_POSITIONS.get(code)
    if label:
        return f"{code} — {label}"
    return code

FIFA_TEAMS = [
    "Argentina",
    "Australia",
    "Brazil",
    "Canada",
    "Colombia",
    "Ecuador",
    "France",
    "Germany",
    "Japan",
    "Mexico",
    "Morocco",
    "Netherlands",
    "New Zealand",
    "Paraguay",
    "Portugal",
    "Qatar",
    "Saudi Arabia",
    "Senegal",
    "South Africa",
    "South Korea",
    "Spain",
    "Switzerland",
    "Tunisia",
    "United States",
    "Uruguay",
]


def build_fifa_card_context(
    profile: Optional[dict] = None,
    is_ai_stats: bool = True,
    stats: Optional[dict] = None,
) -> str:
    """Build prompt text for player profile and stats to render on the trading card."""
    profile = profile or {}
    stats = stats or {}
    lines = []

    profile_parts = []
    if profile.get("club_team"):
        profile_parts.append(f"Club/Team: {profile['club_team']}")
    if profile.get("first_name") or profile.get("last_name"):
        full_name = " ".join(
            part for part in [profile.get("first_name", "").strip(), profile.get("last_name", "").strip()] if part
        )
        if full_name:
            profile_parts.append(f"Name: {full_name}")
    if profile.get("jersey_number"):
        profile_parts.append(f"Jersey #: {profile['jersey_number']}")
    if profile.get("age"):
        profile_parts.append(f"Age: {profile['age']}")
    if profile.get("height_cm"):
        profile_parts.append(f"Height: {profile['height_cm']} cm")
    if profile.get("weight_kg"):
        profile_parts.append(f"Weight: {profile['weight_kg']} kg")

    if profile_parts:
        lines.append("PLAYER PROFILE (fill into the template's existing name/info slots only — do not change card design):")
        lines.extend(f"- {part}" for part in profile_parts)

    position_code = (stats.get("position") or "").strip().upper()
    if position_code:
        position_text = format_fifa_position(position_code)
        lines.append(
            f"PLAYER POSITION: {position_text}. "
            f'Render the position abbreviation "{position_code}" exactly on the trading card in the template\'s position slot. '
            "Pose and kit presentation should match this role naturally."
        )

    if is_ai_stats:
        lines.append(
            "PLAYER STATS: Use AI-generated FIFA-style stats that fit the player's appearance and position. "
            "Render overall rating and attributes (PAC, SHO, PAS, DRI, DEF, PHY) in the template's existing stat areas only."
        )
    else:
        stat_lines = []
        if stats.get("rating"):
            stat_lines.append(f"Overall Rating: {stats['rating']}")
        for key, label in [
            ("pace", "PAC"),
            ("shooting", "SHO"),
            ("passing", "PAS"),
            ("dribbling", "DRI"),
            ("defending", "DEF"),
            ("physical", "PHY"),
        ]:
            if stats.get(key) is not None and str(stats.get(key)).strip() != "":
                stat_lines.append(f"{label}: {stats[key]}")
        if stat_lines:
            lines.append("PLAYER STATS (use these exact values in the template's existing stat slots only):")
            lines.extend(f"- {part}" for part in stat_lines)

    return "\n".join(lines)

