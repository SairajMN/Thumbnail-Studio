import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from agent import ThumbnailAgent, VideoExportContext, generate_after_export
from studio import covers
from studio.design import Spec


def _png_bytes(color: tuple[int, int, int] = (30, 80, 160), size: tuple[int, int] = (320, 480)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_context_validation_and_json(tmp_path: Path):
    with pytest.raises(ValueError, match="frame_path or frame_bytes"):
        VideoExportContext(title="No frame")

    frame = tmp_path / "frame.png"
    frame.write_bytes(_png_bytes())

    with pytest.raises(ValueError, match="at most 3"):
        VideoExportContext(frame_path=frame, extra_frames=[frame, frame, frame, frame])

    ctx = VideoExportContext(
        frame_path=frame,
        title="PS5 Unboxing",
        caption="First look",
        transcript="Wait until you see the controller",
        game="Astro Bot",
        metadata={"duration_sec": 42},
    )
    parsed = json.loads(ctx.to_context_json())
    assert parsed["title"] == "PS5 Unboxing"
    assert parsed["game"] == "Astro Bot"
    assert parsed["metadata"]["duration_sec"] == 42


def test_direct_mode_generate_refine_and_export(monkeypatch, tmp_path: Path):
    frame = tmp_path / "exported_frame.png"
    frame.write_bytes(_png_bytes())

    def fake_run(cid: str) -> None:
        spec = Spec(
            template="key_art",
            design="key_art",
            line1="WAIT FOR",
            line2="THIS DROP",
            highlight="THIS",
            game="Astro Bot",
        )
        covers.update(
            cid,
            status="done",
            stage=None,
            recommended=["key_art", "spotlight_unbox"],
            headlines=[{"line1": "WAIT FOR", "line2": "THIS DROP", "why": "Direct hook"}],
            headline=0,
            why="Direct hook",
            spec=spec.model_dump(),
        )
        covers.draw(cid)

    monkeypatch.setattr(covers, "run", fake_run)

    ctx = VideoExportContext(
        frame_path=frame,
        title="Astro Bot Reveal",
        transcript="Watch till the end!",
        export_sizes=["reel", "youtube"],
        preferred_design="spotlight_unbox",
    )
    result = generate_after_export(ctx)

    assert result.status == "done"
    assert result.chosen_spec is not None
    assert result.chosen_spec["design"] == "spotlight_unbox"
    assert Path(result.exports["reel"]).exists()
    assert Path(result.exports["youtube"]).exists()

    agent = ThumbnailAgent()
    updated = agent.refine(result.cover_id, edits={"line1": "DON'T MISS"})
    assert updated["spec"]["line1"] == "DON'T MISS"


def test_direct_mode_raises_on_failed_cover(monkeypatch):
    def fake_run(cid: str) -> None:
        covers.update(cid, status="failed", error="Budget exceeded")

    monkeypatch.setattr(covers, "run", fake_run)

    ctx = VideoExportContext(frame_bytes=_png_bytes(), title="Fail test")
    with pytest.raises(RuntimeError, match="Budget exceeded"):
        generate_after_export(ctx)


def test_http_mode_generate_and_download(monkeypatch, tmp_path: Path):
    png = _png_bytes()
    polls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/api/covers":
            return httpx.Response(201, json={"id": "abc123", "status": "queued"})
        if request.method == "GET" and path == "/api/covers/abc123":
            polls["count"] += 1
            status = "queued" if polls["count"] == 1 else "done"
            return httpx.Response(
                200,
                json={
                    "id": "abc123",
                    "status": status,
                    "used_usd": 0.0085,
                    "recommended": ["cinema"],
                    "spec": {"design": "cinema", "line1": "TOP", "line2": "PICK"},
                    "headlines": [],
                    "why": "Strong visual",
                    "checks": {"cut": []},
                    "cut_available": True,
                },
            )
        if request.method == "PATCH" and path == "/api/covers/abc123":
            body = json.loads(request.content.decode())
            return httpx.Response(
                200,
                json={
                    "id": "abc123",
                    "status": "done",
                    "used_usd": 0.0085,
                    "recommended": ["cinema"],
                    "spec": {"design": body.get("design", "cinema"), "line1": "TOP", "line2": "PICK"},
                    "headlines": [],
                    "why": "Strong visual",
                    "checks": {"cut": []},
                    "cut_available": True,
                },
            )
        if request.method == "GET" and path.startswith("/api/covers/abc123/export/"):
            return httpx.Response(200, content=png)
        return httpx.Response(404, json={"detail": "not found"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def mock_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", mock_client)

    ctx = VideoExportContext(
        frame_bytes=png,
        title="HTTP test",
        preferred_design="offer",
        export_sizes=["reel", "square"],
    )
    out_dir = tmp_path / "exports"
    result = generate_after_export(ctx, base_url="http://127.0.0.1:8300", output_dir=out_dir)

    assert result.cover_id == "abc123"
    assert result.chosen_spec["design"] == "offer"
    assert result.used_usd == pytest.approx(0.0085)
    assert Path(result.exports["reel"]).read_bytes() == png
    assert Path(result.exports["square"]).read_bytes() == png
