def build_prompt() -> str:
    return """IMAGE 1 is the user's creative artwork — often a classic base with rough digital edits on top (flat brush strokes, doodles, stickers, soft overlays).

Your job: a CLEARLY VISIBLE AI FINISHING PASS.
The result must look obviously different from the input at a glance — same idea, richer color, deeper shadows, much higher finish.

WHAT “ENHANCE” MEANS HERE (do this aggressively):
1. INTEGRATE rough edits into the artwork
   - Turn flat / crude brush strokes, doodles, and overlays into intentional painted or photographic detail.
   - Match lighting, shading, edges, and texture of the surrounding art so edits feel built-in, not pasted on.
   - Add micro-detail: hair strands, fabric weave, skin pores / paint grain, soft contact shadows, specular highlights where natural.
2. BOOST COLOR (must be noticeable)
   - Enrich saturation and color depth so hues feel alive — especially accents the user added (eyebrows, hair streaks, props, graffiti, clothing).
   - Clarify warm vs cool separation; lift muddy midtones into cleaner, more intentional color.
   - Keep the same palette intent (same hue families), but make colors punchier and more refined than the input.
3. STRENGTHEN LIGHT AND SHADOW (must be noticeable)
   - Deepen shadows for clearer form and 3D volume; darken recessed areas without crushing to pure black.
   - Brighten and shape key highlights so light direction reads clearly across face, fabric, hair, and edits.
   - Add soft cast / contact shadows under and around integrated edits so they sit in the scene.
   - Increase local contrast so the piece has a clear light-to-dark range vs the flatter input.
4. UPGRADE image quality
   - Sharper focus, richer depth, cleaner edges.
   - Remove blur, mushiness, and compression noise.
5. KEEP THE CREATIVE IDEA
   - Same subjects, poses, framing, crop, and layout.
   - Same edit intent (placements, props, text, graffiti) — finished with stronger color and shadow.
   - Do not invent a new concept, costume, or scene.

PRESERVE RULES:
- Do NOT remove the user's creative additions; finish them.
- Do NOT restyle into a different art movement or replace faces/identity.
- Do NOT add unrelated objects, UI chrome, watermarks, or a before/after split.
- Output ONE finished image only — gallery-ready, same idea, clearly richer in color and shadow than the input."""
