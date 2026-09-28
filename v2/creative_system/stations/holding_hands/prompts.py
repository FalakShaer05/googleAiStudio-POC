PLATE_LABEL = (
    "BASE ARTWORK TO EDIT. Keep this exact canvas, cream background, composition, arm angles, "
    "clasped-hand pose, line weight, and soft digital shading. Only repaint the two hands and arms."
)

HAND_MAP_LABEL = (
    "OWNERSHIP GUIDE for IMAGE 1 — temporary diagram only, NOT artwork. "
    "Cool-tinted zones = PERSON A. Warm-tinted zones = PERSON B. "
    "Read ownership from the zones; then DISCARD every guide tint. "
    "The finished image must contain ZERO guide blue and ZERO guide orange — "
    "only real skin tones from the person photos."
)

PERSON_A_LABEL = (
    "PERSON A PHOTO — the only source for skin, nails, and jewelry on every cool-tinted "
    "(PERSON A) zone of the ownership guide."
)

PERSON_B_LABEL = (
    "PERSON B PHOTO — the only source for skin, nails, and jewelry on every warm-tinted "
    "(PERSON B) zone of the ownership guide."
)

COLOR_LOCK = (
    "COLOR LOCK (mandatory final check).\n"
    "IMAGE 2 is a temporary ownership diagram. Its blue and orange fills are labels, not paint.\n"
    "Scan the finished hands: if any patch is blue, cyan, cobalt, orange, amber, or map-colored, "
    "you FAILED — repaint that patch with the matching person's real skin tone from their photo.\n"
    "Correct output = natural human skin only (plus black outlines, real nail polish, real jewelry).\n"
    "Never leave guide colors, neon tints, or ownership-map hues anywhere on the hands or arms."
)


def build_prompt() -> str:
    return """Edit IMAGE 1 so its two illustrated hands become the two real hands in IMAGE 3 and IMAGE 4.

WHAT MUST STAY IDENTICAL TO IMAGE 1:
- Canvas size, cream background, and the position, size, and angle of both arms.
- The exact clasped pose and every finger position. Do not add, remove, or move fingers.
- Illustration style: clean thin black outlines, smooth soft digital shading.
- Both forearms stay full, solid natural skin color all the way up to their ends. Never fade the arms into white or cream.
- Empty cream everywhere outside the hands and arms.

WHO OWNS EACH PART — follow IMAGE 2 as a zone diagram only:
- Cool-tinted guide zones = PERSON A (IMAGE 3). Warm-tinted guide zones = PERSON B (IMAGE 4).
- IMAGE 2's blue/orange fills are ownership labels ONLY — never copy, tint, stain, or leave those hues in the art.
- Every finger, fingertip, nail, and knuckle takes the skin, nails, and jewelry of the person who owns that zone.
- Nail polish, rings, and bracelets appear ONLY on their owner's zones. Never put one person's nail polish or rings on the other person's fingers.
- Where the hands overlap, keep a clear visible difference between the two real photo skin tones — not guide colors.

WHAT MUST COME FROM THE PHOTOS (critical — do not keep the template hands):
- Each person's exact skin tone and undertone, as seen in their photo. Do not lighten, darken, cool, or warm it toward the guide.
- Each hand's real character: hand size, finger thickness, knuckle shape, wrist thickness.
- Each person's nails: natural shape and length. If painted in the photo, use that polish color. If natural, draw natural nails.
- Visible features: rings, bracelets, watches, tattoos, freckles, arm hair.
- If a photo shows a whole person instead of a hand, infer that person's skin tone and hand character from it.
- Redraw the photos in the illustration style. Never paste photo pixels, backgrounds, or lighting.

DO NOT:
- Add any text, letters, names, numbers, dates, lines, swashes, or decorations.
- Paint, tint, wash, or outline any hand/arm area with blue, cyan, cobalt, orange, amber, or any IMAGE 2 guide color.
- Keep template peach/pink skin when the photo shows a different tone.
- Add a frame, border, paper texture, shadow, or any background color other than the flat cream.
- Change the crop or zoom.

OUTPUT: the edited IMAGE 1 with only the hands changed — natural photo-matched skin, zero guide blue/orange. Same 4:5 canvas."""
