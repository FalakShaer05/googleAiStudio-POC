PLATE_LABEL = (
    "BASE ARTWORK TO EDIT. Keep this exact canvas, cream background, composition, arm angles, "
    "clasped-hand pose, line weight, and soft digital shading. Only repaint the two hands and arms."
)

HAND_MAP_LABEL = (
    "HAND OWNERSHIP MAP for IMAGE 1. Same composition. Every BLUE area (forearm, palm, thumb, "
    "fingers, and nails) belongs to PERSON A. Every ORANGE area belongs to PERSON B. "
    "Use this map only to decide whose hand each part is. Never draw blue or orange in the output."
)

PERSON_A_LABEL = (
    "PERSON A HAND PHOTO. Paint every BLUE area of the map as this person's real hand."
)

PERSON_B_LABEL = (
    "PERSON B HAND PHOTO. Paint every ORANGE area of the map as this person's real hand."
)


def build_prompt() -> str:
    return """Edit IMAGE 1 so its two illustrated hands become the two real hands in IMAGE 3 and IMAGE 4.

WHAT MUST STAY IDENTICAL TO IMAGE 1:
- Canvas size, cream background, and the position, size, and angle of both arms.
- The exact clasped pose and every finger position. Do not add, remove, or move fingers.
- Illustration style: clean thin black outlines, smooth soft digital shading.
- Both forearms stay full, solid skin color all the way up to their ends. Never fade the arms into white or cream.
- Empty cream everywhere outside the hands and arms.

WHO OWNS EACH PART — follow IMAGE 2 exactly:
- BLUE areas = PERSON A (IMAGE 3). ORANGE areas = PERSON B (IMAGE 4).
- Every finger, fingertip, nail, and knuckle takes the skin, nails, and jewelry of the person whose color it is in the map.
- Nail polish, rings, and bracelets appear ONLY on their owner's areas. Never put one person's nail polish or rings on the other person's fingers.
- Where the hands overlap, keep a clear visible difference between the two skin tones.

WHAT MUST COME FROM THE PHOTOS (critical — do not keep the template hands):
- Each person's exact skin tone and undertone, as seen in their photo. Do not lighten or darken it.
- Each hand's real character: hand size, finger thickness, knuckle shape, wrist thickness.
- Each person's nails: natural shape and length. If painted in the photo, use that polish color. If natural, draw natural nails.
- Visible features: rings, bracelets, watches, tattoos, freckles, arm hair.
- If a photo shows a whole person instead of a hand, infer that person's skin tone and hand character from it.
- Redraw the photos in the illustration style. Never paste photo pixels, backgrounds, or lighting.

DO NOT:
- Add any text, letters, names, numbers, dates, lines, swashes, or decorations.
- Use any blue or orange from the map.
- Add a frame, border, paper texture, shadow, or any background color other than the flat cream.
- Change the crop or zoom.

OUTPUT: the edited IMAGE 1 with only the hands changed. Same 4:5 canvas."""
