"""
Character generation prompt constants for different hobbies and activities.
"""

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

