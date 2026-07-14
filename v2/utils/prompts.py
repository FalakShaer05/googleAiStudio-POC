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
- Side card icons/badges on the left and right edges of the frame (must remain visible)

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

Composite Image 1's real face into the template's portrait area. The result must look like the user from Image 1 wearing the kit, inside the card from Image 3.

If the user wears glasses, keep the same glasses. If they have a beard, mustache, or distinctive hair, preserve them exactly. Do not remove eyewear or facial hair."""

FIFA_IDENTITY_PORTRAIT_PROMPT = """Create a chest-up football player portrait photo.

IMAGE 1 = USER PHOTO — the face in your output MUST be the same person as Image 1.
Copy exactly: face shape, eyes, eyebrows, nose, lips, jawline, skin tone, hair, facial hair, glasses, and every distinctive feature.

If Image 2 (jersey) is provided, dress this same person in that exact kit only.
Image 2 is clothing reference ONLY — never copy any face from Image 2.

Use a simple neutral studio background. No trading card frame, no stats, no text overlays.
Do NOT generate a different person. Do NOT beautify, slim, age-shift, or genericize the face."""

FIFA_IDENTITY_FINAL_REMINDER = """FINAL IDENTITY CHECK (mandatory before output):
The portrait face on the card MUST still be the exact person from the original user photo (Image 1).
If the face looks like the template's original player or a new/generic athlete, you MUST correct it to match Image 1.
Glasses, beard, hairline, and skin tone from Image 1 must remain visible and recognizable."""

FIFA_CARD_COMPOSITE_PROMPT = """TWO INPUT IMAGES — ROLES:

IMAGE 1 = LOCKED PLAYER PORTRAIT — the face is already correct. Do NOT change this person's face.
IMAGE 2 = TRADING CARD TEMPLATE — copy card design/layout at 100% fidelity.

Place Image 1's portrait into Image 2's portrait slot.
Keep Image 1's exact face, glasses, beard, hair, and skin tone unchanged.
Copy Image 2's frame, colors, typography, stats areas, and badges exactly.
Only update player text/stats in the template's existing slots."""

FIFA_TEMPLATE_STRICT_RULES = """FINAL CHECKLIST:
- Face in output = Image 1 user photo (NOT the template face, NOT a new face)
- Card design in output = Image 3 template at 100% fidelity
- Side card icons/badges on the left and right edges must match Image 3 exactly
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
    include_stats: bool = True,
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

    if profile_parts:
        lines.append("PLAYER PROFILE (fill into the template's existing name/info slots only — do not change card design):")
        lines.extend(f"- {part}" for part in profile_parts)

    if not include_stats:
        lines.append(
            "PLAYER STATS: Do NOT generate or update any stats, ratings, position abbreviation, or attribute values. "
            "Preserve the template's original stat areas exactly as shown in the card template."
        )
        return "\n".join(lines)

    position_code = (stats.get("position") or "").strip().upper()
    if not is_ai_stats and position_code:
        position_text = format_fifa_position(position_code)
        lines.append(
            f"PLAYER POSITION: {position_text}. "
            f'Render the position abbreviation "{position_code}" exactly on the trading card in the template\'s position slot. '
            "Pose and kit presentation should match this role naturally."
        )

    if is_ai_stats:
        lines.append(
            "PLAYER STATS: Use AI-generated FIFA-style stats that fit the player's appearance. "
            "Choose a realistic position abbreviation (GK, RB, LB, CB, CDM, CM, CAM, RM, LM, RW, LW, CF, ST) "
            "and render overall rating plus attributes (PAC, SHO, PAS, DRI, DEF, PHY) in the template's existing stat areas only."
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


# --- Birthday Card mode ---

BIRTHDAY_STATION_PREFILLED_PROMPTS = {
    "vangogh": """Create a personalized birthday celebration card in Vincent van Gogh "The Starry Night" post-impressionist style.

ART STYLE (MANDATORY — follow exactly every time):
- Heavy impasto oil-painting texture with bold, visible swirling brushstrokes
- Deep swirling blue and teal night sky with glowing yellow stars and orbs
- Dark cypress tree silhouette on the left side
- Rolling hills in the mid-ground
- Foreground field of glowing yellow circular flowers (sunflower-like orbs) on green grass
- High contrast between deep blues and vibrant yellows — same palette as Starry Night
- The entire image must look like one cohesive painted artwork, not a photo collage

COMPOSITION LAYOUT:
- IMAGE 1 = CELEBRANT (birthday person) — integrate their uploaded photo centrally (waist-up), painted with the same impasto brushstroke texture as the background
- The celebrant is the focal point — largest and most prominent figure in the center
- If participant photos are uploaded, paint EACH participant as a smaller recognizable portrait beside the celebrant
- Also render participant personal messages as handwritten-style text near those people (left/right)
- Bottom center: large bold cursive script text "Happy Birthday [Name]" with a dark outline for legibility
- Include celebrant age near the name if provided

DECORATIVE ELEMENTS:
- Glowing five-pointed stars flanking the celebrant's head
- Small festive accents: party hats, balloons, or stars woven into the swirling sky
- Relationship-appropriate badge overlays when applicable (e.g. "THE BEST!" medal, "Best Mom" / "Best Dad" ribbon badge)
- Small pink heart icons near participant messages

IDENTITY PRESERVATION (CRITICAL):
- The celebrant's face MUST come from the uploaded CELEBRANT PHOTO — never invent a different person
- Preserve exact face shape, eyes, glasses, hair color, hairstyle, skin tone, and facial features
- Apply Van Gogh brushstroke TEXTURE to the portrait — do NOT replace the face with a generic person or a layout-reference face
- Do NOT beautify, age-shift, or alter distinctive features
- Every uploaded participant photo face must also remain recognizable

TEXT RENDERING:
- Main heading: elegant bold cursive/script font with dark stroke outline
- Participant notes: warm handwritten-style font on left and right sides of the celebrant
- All text must be clearly legible against the painted background

OUTPUT:
- ONE unified print-ready birthday card portrait (3:4 aspect ratio)
- Single cohesive painting — NOT a collage, NOT separate panels
- Rich, festive, celebratory atmosphere throughout""",

    "klimt": """Create a personalized birthday celebration card in Gustav Klimt Art Nouveau / gold-leaf mosaic style.

ART STYLE (MANDATORY — follow exactly every time):
- Ornate gold leaf patterns, mosaic textures, and rich warm tones
- Decorative floral and geometric motifs inspired by "The Kiss" and "Portrait of Adele Bloch-Bauer I"
- Shimmering gold backgrounds with intricate decorative borders and frames
- Elegant Art Nouveau composition with flat decorative planes and luminous gold accents
- The entire image must look like one cohesive Klimt-inspired painting

COMPOSITION LAYOUT:
- IMAGE 1 = CELEBRANT (birthday person) — integrate their uploaded photo centrally (waist-up), framed in ornate gold decorative patterns
- The celebrant is the focal point — largest and most prominent figure in the center
- If participant photos are uploaded, paint EACH participant as a smaller recognizable portrait beside the celebrant
- Also render participant personal messages as elegant handwritten-style text near those people (left/right)
- Bottom center: ornate gold script text "Happy Birthday [Name]" integrated into the mosaic border design
- Include celebrant age near the name if provided

DECORATIVE ELEMENTS:
- Gold leaf mosaic circles, rectangles, and spiral patterns surrounding the portrait
- Floral motifs, decorative borders, and Art Nouveau ornamental corners
- Relationship-appropriate badge overlays when applicable (e.g. "THE BEST!" medal, "Best Mom" / "Best Dad" ribbon)
- Small heart and star accents in gold and warm tones

IDENTITY PRESERVATION (CRITICAL):
- The celebrant's face MUST come from the uploaded CELEBRANT PHOTO — never invent a different person
- Preserve exact face shape, eyes, glasses, hair color, hairstyle, skin tone, and facial features
- Apply Klimt decorative STYLE around the portrait — do NOT replace the face with a generic person or a layout-reference face
- Do NOT beautify, age-shift, or alter distinctive features
- Every uploaded participant photo face must also remain recognizable

TEXT RENDERING:
- Main heading: elegant gold script/cursive font integrated into the mosaic design
- Participant notes: refined handwritten-style font on left and right sides
- All text must be clearly legible against the gold and warm-toned background

OUTPUT:
- ONE unified print-ready birthday card portrait (3:4 aspect ratio)
- Single cohesive Klimt-inspired artwork — NOT a collage, NOT separate panels
- Luxurious, celebratory, elegant atmosphere throughout""",

    "custom": """Create a personalized birthday celebration card using your custom art direction described below.

COMPOSITION LAYOUT:
- IMAGE 1 = CELEBRANT (birthday person) — integrate their uploaded photo centrally as the focal point
- If participant photos are uploaded, paint EACH participant as a smaller recognizable portrait beside the celebrant
- Participant messages appear as handwritten-style text near those people / on the LEFT and RIGHT sides
- Bottom center: prominent "Happy Birthday [Name]" text in a style matching your art direction
- Include celebrant age near the name if provided

IDENTITY PRESERVATION (CRITICAL):
- The celebrant's face MUST come from the uploaded CELEBRANT PHOTO
- Preserve exact face shape, eyes, glasses, hair, skin tone, and distinctive features
- Do NOT beautify, age-shift, generate a generic face, or reuse any face from a layout reference
- Every uploaded participant photo face must also remain recognizable

DECORATIVE ELEMENTS:
- Festive celebratory accents appropriate to your chosen style
- Relationship-appropriate badges when applicable (e.g. "THE BEST!" medal, relationship ribbons)

OUTPUT:
- ONE unified print-ready birthday card portrait (3:4 aspect ratio)
- Single cohesive artwork — NOT a collage or separate panels
- Original scene in your custom style — never a near-copy of a layout reference

CUSTOM ART DIRECTION (edit this section):
Describe your desired art style, color palette, brushwork/texture, background scenery, typography style, and decorative elements here.""",
}

BIRTHDAY_STYLE_LABELS = {
    "klimt": "Klimt Inspired Birthday Card",
    "vangogh": "Van Gogh Inspired Birthday Card",
    "custom": "Create Your Own Style",
}

BIRTHDAY_IDENTITY_LOCK = """CRITICAL — USER PHOTOS WHEN PROVIDED:
- If a CELEBRANT PHOTO is attached, that face MUST be the main central portrait and stay recognizably the same person
- Preserve from the celebrant photo: face shape, eyes, eyebrows, nose, lips, jawline, skin tone, hair, glasses, facial hair, age appearance
- Apply art style as TEXTURE/brushwork ONLY — never replace, beautify, or swap an uploaded face
- If PARTICIPANT PHOTOS are attached, EACH of those people MUST appear as painted portraits (recognizable faces)
- If NO celebrant photo is uploaded, center the card on the celebrant NAME/AGE and composition — do not invent a fake celebrity-like face for them
- FORBIDDEN: keeping any face shown only in an inspiration/reference image"""

BIRTHDAY_REFERENCE_LAYOUT_INSTRUCTIONS = """LAYOUT REFERENCE — LOOSE CREATIVE INSPIRATION ONLY:
The last attached image is a soft mood / arrangement idea. Be creative. Do NOT treat it as a template.

Take ONLY light inspiration, such as:
- A general sense that there is a main person and supporting messages or people around them
- A festive birthday-card vibe
- Optional decorative accents like badges, hearts, or title text near the bottom

You have FULL creative freedom to:
- Invent a completely new background, scenery, colors, and composition
- Redesign typography, badge shapes, and decorative elements however you like
- Change proportions, camera framing, and pose freely
- Reinterpret the station art style boldly and uniquely each time

Must NOT do:
- Copy or closely recreate the layout reference image
- Reuse its face(s), background details, brush patterns, color layout, or exact text
- Make an output that a viewer would say "this is basically the same as the reference"

Still required: use the uploaded celebrant (and any participant photos) as the real people in the scene."""

BIRTHDAY_RELATIONSHIPS = [
    "Mother",
    "Father",
    "Grand Mother",
    "Grand Father",
    "Sibling",
    "Child",
    "Relative",
    "Spouse / Partner",
    "Boyfriend / Girlfriend",
    "Best Friend",
    "Friend",
    "Colleague",
    "Others",
]


def get_birthday_station_prompt(style: str) -> str:
    """Return the prefilled station prompt for UI and generation defaults."""
    style_key = (style or "vangogh").strip().lower()
    return BIRTHDAY_STATION_PREFILLED_PROMPTS.get(
        style_key, BIRTHDAY_STATION_PREFILLED_PROMPTS["vangogh"]
    )


def build_birthday_participant_context(participants: list) -> str:
    """Build participant people + message instructions for the generation prompt."""
    if not participants:
        return "PARTICIPANTS: None — celebrant-only card. Only Image 1 (celebrant) must appear as a person."

    lines = [
        "PARTICIPANTS (each uploaded participant photo MUST appear as a recognizable painted person in the card):",
    ]
    positions = ["LEFT of celebrant", "RIGHT of celebrant", "lower LEFT", "lower RIGHT", "upper LEFT", "upper RIGHT"]
    for idx, p in enumerate(participants):
        relationship = (p.get("relationship") or "Participant").strip()
        name = (p.get("name") or "").strip()
        message = (p.get("message") or "").strip()
        has_photo = bool(p.get("photo_path") or p.get("has_photo"))
        pos = positions[idx] if idx < len(positions) else f"near celebrant ({idx + 1})"
        who = name or relationship
        if has_photo:
            lines.append(
                f"- Participant {idx + 1} ({who}, {relationship}): MUST show this person's uploaded photo face "
                f"as a painted portrait {pos} of the celebrant (smaller than the celebrant)."
            )
        else:
            lines.append(
                f"- Participant {idx + 1} ({who}, {relationship}): no photo uploaded — render message text only {pos}."
            )
        if message:
            lines.append(f'  Message text near them: "{message}"{" — " + name if name else ""}')
        badge_hint = _birthday_relationship_badge(relationship)
        if badge_hint:
            lines.append(f"  Badge/ribbon idea: {badge_hint}")

    return "\n".join(lines)


def _birthday_relationship_badge(relationship: str) -> str:
    rel = (relationship or "").strip().lower()
    if rel in {"mother", "father", "grand mother", "grand father"}:
        label = relationship.strip()
        return f'"Best {label}" ribbon badge and/or "THE BEST!" medal'
    if rel == "child":
        return '"THE BEST!" medal or "Best Parent" ribbon if appropriate'
    return ""


def build_birthday_card_context(
    celebrant_name: str,
    celebrant_age: str,
    participants: list,
    card_text: Optional[str] = None,
) -> str:
    """Build dynamic card context appended to the station prompt."""
    lines = [
        "CARD DETAILS:",
        f'- Celebrant name: {celebrant_name or "Celebrant"}',
    ]
    if celebrant_age and celebrant_age.strip():
        lines.append(f"- Celebrant age: {celebrant_age.strip()}")
    lines.append(f'- Main heading text: "Happy Birthday {celebrant_name or "Celebrant"}"')
    if celebrant_age and celebrant_age.strip():
        lines.append(f'- Include age "{celebrant_age.strip()}" near the celebrant name in the design')

    if card_text and card_text.strip():
        lines.append(f'- Additional card message: "{card_text.strip()}"')

    lines.append("")
    lines.append(build_birthday_participant_context(participants))
    lines.append("")
    lines.append(BIRTHDAY_IDENTITY_LOCK)
    return "\n".join(lines)


def build_birthday_generation_prompt(
    style: str,
    celebrant_name: str,
    celebrant_age: str,
    participants: list,
    user_prompt: Optional[str] = None,
    card_text: Optional[str] = None,
    has_reference: bool = False,
    participant_photo_count: int = 0,
    has_celebrant_photo: bool = True,
) -> str:
    """Assemble the full generation prompt: station base + dynamic context."""
    style_key = (style or "vangogh").strip().lower()
    base_prompt = (user_prompt or "").strip() or get_birthday_station_prompt(style_key)
    context = build_birthday_card_context(
        celebrant_name=celebrant_name,
        celebrant_age=celebrant_age,
        participants=participants,
        card_text=card_text,
    )

    image_plan = ["IMAGE ROLES (read carefully before drawing):"]
    if has_celebrant_photo:
        image_plan.append(
            "CELEBRANT PHOTO is provided — that face MUST be the central celebration person."
        )
    else:
        image_plan.append(
            "NO celebrant photo was uploaded — feature celebrant by name/age in title and layout. "
            "Do not invent a specific real-person face for them."
        )
    if participant_photo_count:
        image_plan.append(
            f"{participant_photo_count} PARTICIPANT PHOTO(S) are provided — each face MUST appear as a painted person."
        )
    if has_reference:
        image_plan.append(
            "An INSPIRATION image may be attached last — soft vibe only. Be creative. Do not copy it or its faces."
        )
    if has_celebrant_photo:
        image_plan.append(
            "SUCCESS CHECK: Viewers must recognize the uploaded celebrant face. Keep the card original and creative."
        )
    else:
        image_plan.append(
            "SUCCESS CHECK: Celebrant name is clearly featured. Keep the card original and creative."
        )

    parts = [
        "\n".join(image_plan),
        base_prompt,
    ]
    if has_reference:
        parts.append(BIRTHDAY_REFERENCE_LAYOUT_INSTRUCTIONS)
    parts.append(context)
    return "\n\n".join(parts)

