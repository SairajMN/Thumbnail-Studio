# AGENT.md — Thumbnail Studio Architecture & Codebase Guide

> **For Coding Agents & Repo Scanners**: This document is the single source of truth for how **Thumbnail Studio** works, how its modules fit together, what invariants must be preserved, and how to extend or integrate it safely.

---

## 1. What This Project Is

**Thumbnail Studio** is a context-aware reel and video thumbnail generator built with **FastAPI**, **Pillow**, **OpenCV**, **NumPy**, and **Nebius Token Factory** (OpenAI-compatible JSON-schema LLM/vision calls).

It takes:
- A **video keyframe** (PNG/JPEG/WebP), and
- **Video context** (title, caption, transcript, timeline notes, game/product/offer metadata — plain text or JSON),

and produces **1080×1920** covers (plus **1080×1350** Feed, **1080×1080** Square, and **1280×720** YouTube exports) in both:
- **`cut`** (`cover-cut.png`): subject segmented out with a smooth white sticker outline and soft glow.
- **`plain`** (`cover-plain.png`): the original frame graded and framed without cutting out the subject.

It also includes [`agent.py`](agent.py), a drop-in post-export integration layer designed for **video editor / video generator agents** to call right after the user clicks **Export**.

---

## 2. Core Invariants (Do Not Break)

1. **Pre-call Budget Guardrail**: Every Nebius model call in `studio/covers.py:_ask()` checks `spent(c) + worst_case_usd <= budget_usd` (default `$0.15`) *before* sending the request. Never bypass `_ask()` for model calls.
2. **Zero-Cost Code-Only Redraws**: Editing `Spec` fields (`PATCH /api/covers/{id}` or `covers.edit()`) redraws `cover-cut.png` and `cover-plain.png` deterministically in Python (`studio/design.py:draw()`) with **zero** LLM cost.
3. **Face-Safe Layout**: `studio/design.py` ensures headlines, pills, washes, rings, arrows, PEGI labels, and waves never overlap detected faces (`studio/frame.py:faces()`).
4. **Local File-Based State**: Each cover lives in `data/covers/<id>/` (`cover.json`, `frame.png`, `mask.npy`, `cover-cut.png`, `cover-plain.png`, `previews/`, `exports/`, `refs/`). Writes to `cover.json` are guarded by `studio/covers.py:LOCK`.
5. **Test Isolation**: `tests/conftest.py` overrides `DATA_DIR` to a temp directory and clears `NEBIUS_API_KEY`. Tests use `nebius.FAKE` to stub model responses without network calls.

---

## 3. Repository & Module Map

| File | Responsibility |
|---|---|
| [`agent.py`](agent.py) | Standalone post-export orchestrator (`VideoExportContext`, `ThumbnailResult`, `ThumbnailAgent`, `generate_after_export()`, CLI). Supports both in-process (`studio.covers`) and HTTP microservice modes without modifying `studio/`. |
| [`studio/app.py`](studio/app.py) | FastAPI HTTP server & routes (`/api/covers`, `/api/covers/{id}/chat`, `/api/covers/{id}/export/{size}.png`, `/api/covers/{id}/ref/{name}`, `/api/designs`, `/api/palettes`, `/api/options`). |
| [`studio/covers.py`](studio/covers.py) | Cover lifecycle (`create`, `run`, `load`, `save`, `update`, `edit`, `undo`, `draw`, `export`, `previews`, `add_images`, `clear_images`, `out`). Orchestrates `Read → Look → Write → Draw` and enforces `budget_usd`. |
| [`studio/frame.py`](studio/frame.py) | Computer vision & OCR (free, local): YuNet face detection (`faces`), U²-Net-p subject segmentation with connected-component & morphological cleanup (`subject_mask`, `_clean_mask`), k-means palette extraction (`palette`), Tesseract OCR (`text`). |
| [`studio/design.py`](studio/design.py) | `Spec` Pydantic model and deterministic 1080×1920 renderer (`draw`). Handles `pan_x`, `pan_y`, `zoom`, cutout compositing, anti-aliased sticker outlines, cards/panels, collages, typography, badges, QR codes, and quality checks. |
| [`studio/designs.py`](studio/designs.py) | Registry of all **18 cover designs** (`DESIGNS`), background generators (`studio`, `pattern`, `flood`, `gradient`, `spotlight`, `space`, `sunset`, `blur`, `frame`), decorative shapes, and pedestals. |
| [`studio/palette.py`](studio/palette.py) | Colour presets (`PRESETS`), user-saved palettes (`saved`, `save`, `forget`), WCAG contrast/luminance helpers, and role resolution (`bg`, `big`, `small`, `highlight`, `pill`, `glow`). |
| [`studio/chat.py`](studio/chat.py) | Conversational editor (`say`, `understand`, `Plan`). Handles simple edits in code for free; routes complex text edits to `EDIT_MODELS` (`glm-5.3-flash` → `glm-5.3`) and reference-image edits (`"make it like that"`) to `VISION_EDIT_MODELS` (`kimi-k3` → `kimi-k2.6` → `qwen3.8-27b`). |
| [`studio/nebius.py`](studio/nebius.py) | Nebius Token Factory OpenAI-compatible client (`ask`), token/cost estimation (`estimate_in`, `usd`), model registry (`MODELS`), and `FAKE` test hook. |
| [`studio/kit.py`](studio/kit.py) | Brand kit helpers (PEGI rating detection & label assets under `studio/kit/`). |
| [`studio/config.py`](studio/config.py) | `Settings` (`nebius_api_key`, `budget_usd`, `data_dir`) loaded from `.env` / environment variables. |
| [`studio/static/index.html`](studio/static/index.html) | Single-file chat + live preview UI (vanilla JS/CSS, no build step). |
| [`tests/`](tests/) | `pytest` suite (`test_studio.py`, `test_agent.py`, `test_chat_ref.py`). |

---

## 4. End-to-End Pipeline & Features

### A. 4-Stage Cover Generation (`studio/covers.py:run`)
1. **Read (code, free)**: `frame.read()` extracts faces (`YuNet`), subject cutout mask (`U²-Net-p`), dominant colours, and on-screen text (`Tesseract`).
2. **Look (vision model)**: `LOOK_MODELS` (`kimi-k3` → `kimi-k2.6` → `qwen3.8-27b` → `glm-5.3` → `glm-5.3-flash`) identifies `game`, `seen`, `background`/`accent` hex colours, `product_box`, `hook`, and top 3 `designs`.
3. **Write (text model)**: `WRITE_MODELS` (`glm-5.3` → `glm-5.2` → `glm-5.1` → `glm-5.3-flash`) writes 3 `Words` headline options (`line1`, `line2`, `highlight`, `pill_left`, `pill_right`, `name_tag`, `arrow`, `offer`, `why`).
4. **Draw (code, free)**: `covers.draw()` renders `cover-cut.png` and `cover-plain.png` and runs automated face-overlap, 3:4 grid crop, and contrast checks.

### B. Enhanced Cut-out Pipeline (`studio/frame.py` + `studio/design.py`)
- `_clean_mask()` in `studio/frame.py` applies morphological opening (`5×5` ellipse) to sever thin bridges to background objects, filters out disconnected islands below `max(40, 8% of largest component)` via `cv2.connectedComponentsWithStats`, closes internal holes (`7×7` ellipse), and feathers the support edge.
- `CUT_MODES` in `studio/design.py` performs a second-pass connected-component filter on the 1080×1920 canvas (`area >= max(900, 5% of largest component)`), smooths the subject alpha, and blends a Gaussian-smoothed (`sigma=2.0`) white sticker outline instead of a hard binary threshold.

### C. Single Best Thumbnail Default + On-Demand 18-Format Picker (`studio/static/index.html`)
- After generation completes (`status === "done"`), the UI goes directly to `S.step = "ready"` showing **one best cover** with the **With cut-out / Without cut-out** toggle.
- All 18 designs (`/api/covers/{id}/previews`) are only fetched and rendered on demand when the user clicks **All 18 formats** or asks in chat (e.g., `"show all formats"`, `"more options"`).

### D. Interactive Cursor Drag-to-Slide & Zoom Fit (`pan_x`, `pan_y`, `zoom`)
- `Spec` in `studio/design.py` includes `pan_x` (`-1.0..1.0`), `pan_y` (`-1.0..1.0`), and `zoom` (`0.5..2.5`).
- `_card()` (panel/card layouts) and full-frame `draw()` apply these transforms to both `cut` and `plain` outputs.
- In `studio/static/index.html`, pointer events (`pointerdown`/`pointermove`/`pointerup`) on `#shot` let the user click, hold, and slide the image, plus `−` / `+` / `Reset` buttons in `#fitbar` trigger `PATCH /api/covers/{id}`.

### E. Reference-Image Generation & Style Transfer (`"make it like that"`)
- Users can attach a reference image **both at the start of the chat** (`context` step, alongside the frame and video context) **and in follow-up chat edits** (`ready` step) via the **`+`** button (`#refPick`), **drag-and-drop** onto `#log` / `#composer`, or **clipboard paste** (`Ctrl/Cmd+V`) into `#text`.
- **Two generation modes**:
  1. **Frame only (no reference image)**: Runs the normal 4-stage pipeline (`Read → Look [Kimi] → Write [GLM] → Draw [PIL]`), producing both **With cut-out** (`cover-cut.png`) and **Without cut-out** (`cover-plain.png`) versions.
  2. **Frame + Reference image**: `POST /api/covers` accepts an optional `ref_image` file (saved to `data/covers/<id>/refs/ref-1.png`) and `covers.run()` calls `studio/imggen.py:generate_from_ref()` using OpenRouter (`/api/v1/images`) with `openai/gpt-image-2` by default and falling back to `minimax/minimax-h3-image` → `openai/gpt-image-2.5-sunburst`, enforcing `budget_usd`. The model's USD cost is recorded in `calls` (`step="generate"` / `"edit"`) and displayed in the UI's **Checks and costs** panel. Covers generated from a reference image set `ref_generated=True` and render as a **single version** (no `With cut-out` / `Without cut-out` tabs or split downloads).
- `POST /api/covers/{id}/chat` accepts `multipart/form-data` (`message` + `image`) in addition to JSON (`{"message": "..."}`), allowing users to attach a new reference image and generate a fresh variation at any time (falling back to Nebius `VISION_EDIT_MODELS` when `OPENROUTER_API_KEY` is not set).

### F. Video Editing Agent Integration & Video Context Input ([`agent.py`](agent.py))
- **For Video Editing Agents**: Call `generate_after_export(VideoExportContext(...))` in [`agent.py`](agent.py) right after video export, or `POST /api/covers` directly with `frame` (`UploadFile`), `context` (`Form` string — plain text or JSON), and optional `ref_image` (`UploadFile`).
- **Expected Video Context Schema** (`VideoExportContext.to_context_json()`):
  ```json
  {
    "title": "Video title or working headline",
    "caption": "Social post caption / hook",
    "transcript": "Spoken transcript from the video",
    "timeline_notes": "Editor timeline markers, key moments, or visual cues",
    "game": "Game or product name (optional)",
    "offer": "Promo/price tag e.g. '20% OFF' (optional)",
    "metadata": {}
  }
  ```
- **UI Video Context Box** ([`studio/static/index.html`](studio/static/index.html)): During the `context` step, the chat composer acts as a dedicated Video Context input box where users can paste plain video notes **or** the Video Editing Agent's JSON payload directly (with a **Paste Agent JSON** chip that pre-fills the schema above). `studio/covers.py:_context()` parses both JSON and plain text seamlessly.

### G. Unified Standalone `imggen/` App (`imggen/`)
- [`imggen/`](imggen/) is bundled in this same repository: a standalone black-and-white OpenRouter image-generation studio (`imggen/app.py`, `imggen/prompt.py`, `imggen/static/index.html`, `imggen/tests/test_app.py`) with `openai/gpt-image-2` default + `minimax/minimax-h3-image` / `openai/gpt-image-2.5-sunburst` fallbacks, model dropdown, `/api/history`, and hard `BUDGET_USD` + `MAX_CALLS` guardrails.

---

## 5. Running & Testing

```bash
# Run server
uv run uvicorn studio.app:app --port 8300

# Run full test suite (all tests run offline via nebius.FAKE)
uv run pytest -q
```
