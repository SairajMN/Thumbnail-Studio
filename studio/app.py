"""Thumbnail Studio: `uv run uvicorn studio.app:app --port 8300`, then open http://127.0.0.1:8300."""

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from studio import covers

app = FastAPI(title="Thumbnail Studio", version="0.1.0")
PAGE = Path(__file__).with_name("static") / "index.html"


class EditIn(BaseModel):
    line1: str | None = None
    line2: str | None = None
    highlight: str | None = None
    pill_left: str | None = None
    pill_right: str | None = None
    name_tag: str | None = None
    bg: str | None = None
    accent: str | None = None
    punch_in: bool | None = None
    arrow: bool | None = None
    wave: bool | None = None


def _err(e: covers.CoverError) -> HTTPException:
    return HTTPException(e.status, str(e))


@app.get("/", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(PAGE)


@app.get("/api/options")
def options() -> dict:
    return covers.options()


@app.post("/api/covers", status_code=201)
async def create_cover(tasks: BackgroundTasks, frame: UploadFile = File(...), context: str = Form("")) -> dict:
    """Start a cover from the picked frame and what is known about the video (text or JSON)."""
    data = await frame.read(covers.MAX_FRAME_MB * (1 << 20) + 1)
    try:
        c = covers.create(data, context)
    except covers.CoverError as e:
        raise _err(e) from e
    tasks.add_task(covers.run, c["id"])
    return covers.out(c)


@app.get("/api/covers/{cover_id}")
def get_cover(cover_id: str) -> dict:
    try:
        return covers.out(covers.load(cover_id))
    except covers.CoverError as e:
        raise _err(e) from e


@app.patch("/api/covers/{cover_id}")
def edit_cover(cover_id: str, body: EditIn) -> dict:
    """Change the words, colours or effects; both versions are drawn again at once, at no cost. An empty string
    removes the highlight, pill or name tag."""
    changes = {k: (v or None) if isinstance(v, str) and k not in ("line1", "line2") else v
               for k, v in body.model_dump(exclude_none=True).items()}
    try:
        return covers.out(covers.edit(cover_id, changes))
    except covers.CoverError as e:
        raise _err(e) from e


@app.get("/api/covers/{cover_id}/{name}.png")
def cover_file(cover_id: str, name: str, download: bool = False) -> FileResponse:
    """`cut` (subject cut out), `plain` (the frame as it is) or `frame` (the upload)."""
    if name not in (*covers.VERSIONS, "frame"):
        raise HTTPException(404, "not found")
    try:
        p = covers.folder(cover_id) / (f"cover-{name}.png" if name in covers.VERSIONS else "frame.png")
    except covers.CoverError as e:
        raise _err(e) from e
    if not p.exists():
        raise HTTPException(404, "not drawn yet")
    return FileResponse(p, media_type="image/png", filename=f"cover-{cover_id[:8]}-{name}.png",
                        content_disposition_type="attachment" if download else "inline")
