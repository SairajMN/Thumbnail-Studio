# Thumbnail Studio

Reel covers that make people watch. Give it the frame you picked and what the video is about; it gives back a
1080×1920 cover in the game's own look, with a hook built on the video's content.

## How a cover is made

| Step | Who | Cost |
|---|---|---|
| Read the frame: faces, the subject's cut-out, main colours, on-screen text | code (YuNet, U²-Net-p, Tesseract) | free |
| Look: which game, its colours, where the product is, the most curious moment in the video | **Kimi K3** on Nebius (sees the frame) | ~$0.007 |
| Write: setup line with one highlighted word, punch line, pill, whether to point at the product | **GLM 5.3** on Nebius | ~$0.002 |
| Draw: punch-in and grade, subject pop (outline and glow), ring and arrow on the product, headline, PEGI label; then checks | code | free |

Every model call is checked against the budget (`BUDGET_USD`, $0.15) **before** it is made, at its worst case, so a
cover never costs more. Fallbacks stay on Nebius: looking goes Kimi K2.6, Qwen 3.8 27B, then GLM given only what code
read; writing goes GLM 5.2, 5.1, 5.3 Flash. GLM 5.3 Flash is never sent the frame: Nebius drops its images.

Nothing is ever drawn over a face: not a word, the pill, the wash behind the words, the ring, the arrow, the PEGI
label or the wave. Each cover is checked for that, for the profile grid's 3:4 crop, for contrast and for size.

## Run it

```bash
brew install tesseract            # once, for on-screen text
cp .env.example .env              # then put your Nebius key in NEBIUS_API_KEY
uv sync
uv run uvicorn studio.app:app --port 8300
```

Open http://127.0.0.1:8300, choose the frame, paste the context (caption, transcript, notes; plain text or JSON) and
make the cover. Edit the words, colours or effects afterwards: that redraws it with no model call.

## API

- `POST /api/covers` (form: `frame` image, `context` text) starts a cover; `GET /api/covers/{id}` until `status` is `done`
- `PATCH /api/covers/{id}` with any of `line1`, `line2`, `highlight`, `pill_left`, `pill_right`, `name_tag`, `bg`, `accent`, `pop`, `punch_in`, `arrow`, `wave`
- `GET /api/covers/{id}/cover.png` (`?download=1` to save it), `GET /api/options` for the models, prices and budget

Covers are kept as files in `data/covers/<id>/`.

## Tests

```bash
uv run pytest
```

## Credits

- Face detection: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), OpenCV Zoo, MIT
- Cut-out: [U²-Net-p](https://github.com/xuebinqin/U-2-Net), Apache-2.0 (the file [rembg](https://github.com/danielgatis/rembg) distributes)
- PEGI labels and ratings: copied from Guideline Copilot's brand kit (`studio/kit`)
