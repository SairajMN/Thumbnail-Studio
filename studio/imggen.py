"""OpenRouter image-to-image generation for reference-image chat edits ("make it like that").

Uses `openai/gpt-image-2` by default and falls back to `minimax/minimax-h3-image`
then `openai/gpt-image-2.5-sunburst` only when the default fails, respecting `budget_usd`.
"""

from __future__ import annotations

import base64
import io
import time
from collections.abc import Callable
from typing import Any

import httpx
from PIL import Image, ImageOps

from studio import covers, design, nebius
from studio.config import settings

OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images"

# Optional test hook: (model, prompt, frame_bytes, ref_bytes) -> (png_bytes, cost_usd)
FAKE: Callable[[str, str, bytes, bytes], tuple[bytes, float]] | None = None

BASE_PROMPT = """You are given two input images and video context:
- IMAGE 1 (Subject Frame): The real person, product, or scene from the video.
- IMAGE 2 (Reference Thumbnail): The exact visual style, layout, color palette, lighting, typography, and graphic composition to replicate.

GOAL:
Generate a 1080x1920 vertical 9:16 social/reel thumbnail cover that replicates IMAGE 2's exact layout, design structure, background, color grading, lighting, decorative elements, badge/pill placement, and text styling—while replacing the subject with the exact person/product from IMAGE 1 and updating the text from the VIDEO CONTEXT below.

STRICT RULES (100% FIDELITY):
1. SUBJECT IDENTITY (FROM IMAGE 1):
   - Preserve the exact face, facial expression, skin tone, hair, outfit, logos on clothing, pose, and any held product/object from IMAGE 1 with 100% likeness.
   - Cleanly segment the subject from IMAGE 1's original background and composite them into IMAGE 2's exact subject position, scale, and framing.
2. LAYOUT & STYLE CLONING (FROM IMAGE 2):
   - Copy IMAGE 2's exact composition, background style, typography, text outline/shadow, CTA pill/badge geometry, and color palette.
   - NEVER place any text, pill, badge, icon, or graphic over the subject's face or eyes.
3. TEXT & HOOK:
   - Use the exact headline copy below, spelled 100% accurately, crisp, high-contrast, with zero gibberish or extra watermarks."""


def _build_prompt(c: dict[str, Any], message: str) -> str:
    sp = c.get("spec") or {}
    pill = " ".join(x for x in (sp.get("pill_left"), sp.get("pill_right")) if x)
    lines = [BASE_PROMPT, f"HEADLINE COPY:\n- Small setup line: {sp.get('line1') or ''}\n- Big main headline: {sp.get('line2') or ''}\n- Highlight word: {sp.get('highlight') or ''}\n- CTA pill: {pill}"]
    if c.get("context"):
        lines.append(f"VIDEO CONTEXT:\n{c['context'][:1500]}")
    if message.strip():
        lines.append(f"USER INSTRUCTION:\n{message.strip()}")
    return "\n\n".join(lines)


def _decode_output(body: dict[str, Any]) -> bytes:
    if "data" in body:
        items = body.get("data") or []
        if not items:
            raise RuntimeError("Model returned no image")
        if items[0].get("b64_json"):
            return base64.b64decode(items[0]["b64_json"])
        url = items[0].get("url") or ""
    else:
        images = body["choices"][0]["message"].get("images") or []
        if not images:
            raise RuntimeError("Model returned no image")
        url = images[0].get("image_url", {}).get("url") or images[0].get("url") or ""
    if url.startswith("data:image/"):
        return base64.b64decode(url.split(",", 1)[1])
    if url.startswith(("http://", "https://")):
        r = httpx.get(url, timeout=60)
        r.raise_for_status()
        return r.content
    raise RuntimeError("Unsupported image output format")


def _call_one(model: str, prompt: str, frame_bytes: bytes, ref_bytes: bytes) -> tuple[bytes, float]:
    if FAKE is not None:
        return FAKE(model, prompt, frame_bytes, ref_bytes)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "aspect_ratio": "9:16",
        "input_references": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(raw).decode()}"}}
            for raw in (frame_bytes, ref_bytes)
        ],
    }
    if "gpt-image" in model:
        payload["moderation"] = "low"
    r = httpx.post(
        OPENROUTER_IMAGES_URL,
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter ({r.status_code}): {r.text[:200]}")
    body = r.json()
    img_bytes = _decode_output(body)
    usage = body.get("usage") or {}
    raw_cost = usage.get("cost") if usage.get("cost") is not None else body.get("total_cost")
    cost = round(float(raw_cost), 6) if raw_cost is not None else settings.imggen_est_usd
    return img_bytes, cost


def generate_from_ref(cid: str, frame_bytes: bytes, ref_bytes: bytes, message: str) -> tuple[str, float, str]:
    """Generate a new cover matching the reference image, saving cover-cut.png & cover-plain.png."""
    c = covers.load(cid)
    prompt = _build_prompt(c, message)
    candidates = [settings.imggen_model, *(m for m in settings.imggen_fallbacks if m != settings.imggen_model)]
    errors: list[str] = []

    for idx, model in enumerate(candidates):
        c = covers.load(cid)
        used = covers.spent(c)
        if used + settings.imggen_est_usd > c["budget_usd"]:
            raise covers.CoverError(
                f"budget cap reached (${used:.4f} of ${c['budget_usd']:.2f} used)", 402
            )
        t0 = time.monotonic()
        try:
            out_bytes, cost_usd = _call_one(model, prompt, frame_bytes, ref_bytes)
            call = nebius.Call(step="edit", model=model, ok=True, usd=cost_usd, ms=int((time.monotonic() - t0) * 1000))
            with covers.LOCK:
                cur = covers.load(cid)
                cur["calls"].append(call.dict())
                covers.save(cur)
            img = ImageOps.fit(Image.open(io.BytesIO(out_bytes)).convert("RGB"), (design.W, design.H), Image.LANCZOS)
            folder = covers.folder(cid)
            for name in covers.VERSIONS:
                img.save(folder / f"cover-{name}.png", optimize=True)
            covers.update(cid, version=cur["version"] + 1)
            note = "Generated to match your reference image." + (
                f" (fallback from {settings.imggen_model})" if idx > 0 else ""
            )
            return model, cost_usd, note
        except Exception as e:
            call = nebius.Call(
                step="edit", model=model, ok=False, usd=0.0,
                ms=int((time.monotonic() - t0) * 1000), error=str(e)[:200],
            )
            with covers.LOCK:
                cur = covers.load(cid)
                cur["calls"].append(call.dict())
                covers.save(cur)
            errors.append(f"{model}: {e}")

    raise covers.CoverError("no image model succeeded: " + " | ".join(errors), 502)
