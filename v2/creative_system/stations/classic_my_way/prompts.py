def build_prompt(user_prompt: str) -> str:
    return f"""IMAGE 1 is the ORIGINAL ART / TEMPLATE (blank / base design).
IMAGE 2 is the USER ART image with their art placed on that template.

Use IMAGE 1 to understand and lock the template. Enhance IMAGE 2 only.

Enhance the user's artwork while strictly preserving the original template and creative intent.
Do NOT change the template's orientation, size, aspect ratio, framing, layout, or proportions. Do NOT rotate, crop, stretch, replace, redesign, or remove any template elements.
Preserve all user-created text, drawings, colors, objects, and elements. Do not change the wording or meaning of any text.
Only refine the user's additions by improving:

Placement, alignment, and spacing
Line quality and shapes
Color harmony, contrast, and vibrancy
Overall balance and visual cohesion
Cleanliness and print quality
Make subtle, professional improvements while keeping the artwork clearly recognizable as the user's original creation.
Enhance — do not redesign. The final result must have the exact same orientation, dimensions, and canvas boundaries as the original.

USER PROMPT:
{user_prompt.strip()}

Follow the USER PROMPT for how to enhance, while still obeying all preserve rules above."""
