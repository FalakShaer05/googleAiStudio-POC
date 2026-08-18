def build_prompt(user_prompt: str) -> str:
    return f"""IMAGE 1 is the user's source artwork (a famous painting, drawing, or photo).

Edit THAT image according to the user's instructions. Keep the work recognizable unless the prompt asks otherwise. Preserve the original composition, canvas shape, and core subject as a starting point, then apply the requested changes (clothing, props, background, style mashups, text, etc.).

USER PROMPT:
{user_prompt.strip()}

RULES:
- Produce a finished artwork, not a UI mockup, editor chrome, or before/after split.
- Do not add watermarks, buttons, or toolbars.
- Follow the user prompt closely; invent extra gimmicks only if they clearly support the request.
- Output a single edited image."""
